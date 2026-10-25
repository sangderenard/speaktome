"""Thin, explicit OpenRGB SDK shell for full-controller RGB frames.

Importing this module performs no network discovery and no hardware writes.
Create a connection explicitly with ``connect()`` or invoke the CLI.  The CLI
is dry-run by default; mutating commands require ``--apply``.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

RGB = tuple[int, int, int]


class OpenRGBShellError(RuntimeError):
    """A concise, user-facing configuration or API error."""


@dataclass(frozen=True)
class ZoneLayout:
    index: int
    name: str
    start: int
    count: int
    addresses: tuple[int, ...] = ()
    frame_indices: tuple[int, ...] = ()

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class DomainAddress:
    """One system address projected into the dense law domain.

    ``domain_index`` is the compiler-facing coordinate.  ``frame_index`` is
    the slot in OpenRGB's full-device frame.  The remaining fields preserve
    the controller's own address and region identity without asking the
    symbolic law to understand either representation.
    """

    domain_index: int
    frame_index: int
    system_address: int
    region_index: int
    region_name: str
    region_position: int


def validate_rgb(value: Sequence[int]) -> RGB:
    if len(value) != 3:
        raise ValueError("RGB values must contain exactly three channels")
    channels = tuple(value)
    if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in channels):
        raise TypeError("RGB channels must be integers")
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("RGB channels must be in the inclusive range 0..255")
    return channels  # type: ignore[return-value]


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _zone_layouts(device: Any) -> tuple[ZoneLayout, ...]:
    layouts: list[ZoneLayout] = []
    device_leds = list(device.leds)
    unused = set(range(len(device_leds)))
    domain_start = 0
    for index, zone in enumerate(device.zones):
        addresses = tuple(int(led.id) for led in zone.leds)
        frame_indices: list[int] = []
        for led in zone.leds:
            object_matches = [
                frame_index for frame_index in unused
                if device_leds[frame_index] is led
            ]
            address_matches = [
                frame_index for frame_index in unused
                if int(device_leds[frame_index].id) == int(led.id)
            ]
            matches = object_matches or address_matches
            if not matches:
                raise OpenRGBShellError(
                    f"Zone {index} ({zone.name!r}) address {int(led.id)} "
                    "does not map to an unused full-frame slot"
                )
            frame_index = matches[0]
            unused.remove(frame_index)
            frame_indices.append(frame_index)
        count = len(addresses)
        layouts.append(ZoneLayout(
            index,
            str(zone.name),
            domain_start,
            count,
            addresses,
            tuple(frame_indices),
        ))
        domain_start += count
    if domain_start != len(device_leds) or unused:
        raise OpenRGBShellError(
            f"Zones describe {domain_start} unique LEDs but the controller reports "
            f"{len(device_leds)}"
        )
    return tuple(layouts)


class FullFrameController:
    """Minimal full-frame interface around one openrgb-python Device.

    ``client`` and ``device`` are public escape hatches.  For the ASRock
    Polychrome USB controller, use ``send_frame`` instead of zone/LED SDK
    setters: only the full-controller update reaches its per-LED packet path.
    """

    def __init__(self, client: Any, device: Any, color_factory: Callable[[int, int, int], Any]):
        self.client = client
        self.device = device
        self._color_factory = color_factory
        self._layouts = _zone_layouts(device)

    @property
    def led_count(self) -> int:
        return len(self.device.leds)

    @property
    def zones(self) -> tuple[ZoneLayout, ...]:
        return self._layouts

    @property
    def address_domain(self) -> tuple[DomainAddress, ...]:
        return tuple(
            DomainAddress(
                domain_index=zone.start + region_position,
                frame_index=zone.frame_indices[region_position],
                system_address=zone.addresses[region_position],
                region_index=zone.index,
                region_name=zone.name,
                region_position=region_position,
            )
            for zone in self._layouts
            for region_position in range(zone.count)
        )

    def zone(self, selector: int | str) -> ZoneLayout:
        if isinstance(selector, int):
            try:
                return self._layouts[selector]
            except IndexError as exc:
                raise OpenRGBShellError(f"Zone index {selector} is out of range") from exc
        matches = [zone for zone in self._layouts if zone.name == selector]
        if not matches:
            available = ", ".join(repr(zone.name) for zone in self._layouts)
            raise OpenRGBShellError(f"Unknown zone {selector!r}; available zones: {available}")
        return matches[0]

    def select_mode(self, mode: str) -> None:
        if not mode:
            raise ValueError("Mode must be named explicitly")
        self.device.set_mode(mode)

    def blank_frame(self, background: RGB = (0, 0, 0)) -> list[RGB]:
        color = validate_rgb(background)
        return [color for _ in range(self.led_count)]

    def solid_frame(self, color: RGB) -> list[RGB]:
        return self.blank_frame(validate_rgb(color))

    def pixel_frame(
        self,
        zone_selector: int | str,
        position: int,
        color: RGB,
        background: RGB = (0, 0, 0),
    ) -> list[RGB]:
        zone = self.zone(zone_selector)
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("Pixel position must be an integer")
        if position < 0 or position >= zone.count:
            raise OpenRGBShellError(
                f"Pixel position {position} is outside zone {zone.name!r} (0..{zone.count - 1})"
            )
        frame = self.blank_frame(validate_rgb(background))
        frame[zone.frame_indices[position]] = validate_rgb(color)
        return frame

    def send_frame(self, frame: Iterable[RGB], *, fast: bool = True) -> None:
        colors = [validate_rgb(color) for color in frame]
        if len(colors) != self.led_count:
            raise OpenRGBShellError(
                f"Frame has {len(colors)} colors; controller requires exactly {self.led_count}"
            )
        encoded = [self._color_factory(*color) for color in colors]
        self.device.set_colors(encoded, fast=fast)


def connect(
    *,
    host: str = "127.0.0.1",
    port: int = 6742,
    device_index: int | None = None,
    device_name: str | None = None,
    allow_remote: bool = False,
    client_factory: Callable[..., Any] | None = None,
    color_factory: Callable[[int, int, int], Any] | None = None,
) -> FullFrameController:
    """Connect to an already-running SDK server and select exactly one device."""
    if not allow_remote and not _is_loopback(host):
        raise OpenRGBShellError(
            f"Refusing non-loopback SDK host {host!r}; pass allow_remote=True explicitly"
        )
    if port < 1 or port > 65535:
        raise ValueError("Port must be in the inclusive range 1..65535")
    if device_index is not None and device_name is not None:
        raise ValueError("Select a device by index or exact name, not both")
    if client_factory is None or color_factory is None:
        try:
            from openrgb import OpenRGBClient
            from openrgb.utils import RGBColor
        except ModuleNotFoundError as exc:
            raise OpenRGBShellError(
                "openrgb-python is not installed; install requirements.txt in an isolated environment"
            ) from exc
        client_factory = client_factory or OpenRGBClient
        color_factory = color_factory or RGBColor
    try:
        client = client_factory(address=host, port=port, name="speaktome-openrgb-shell")
    except Exception as exc:
        raise OpenRGBShellError(f"Could not connect to OpenRGB SDK at {host}:{port}: {exc}") from exc
    devices = list(client.devices)
    if not devices:
        raise OpenRGBShellError("The SDK server reported no devices")
    if device_name is not None:
        matches = [device for device in devices if str(device.name) == device_name]
        if len(matches) != 1:
            raise OpenRGBShellError(
                f"Expected one exact device named {device_name!r}; found {len(matches)}"
            )
        device = matches[0]
    else:
        index = 0 if device_index is None else device_index
        if index < 0 or index >= len(devices):
            raise OpenRGBShellError(f"Device index {index} is out of range 0..{len(devices) - 1}")
        device = devices[index]
    return FullFrameController(client, device, color_factory)


def _parse_zone(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6742)
    parser.add_argument("--allow-remote", action="store_true")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--device-index", type=int)
    selector.add_argument("--device-name")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="Print devices and zones; sends no lighting update")

    solid = commands.add_parser("solid", help="Build and optionally send one full solid frame")
    solid.add_argument("--mode", required=True)
    solid.add_argument("--rgb", type=int, nargs=3, required=True, metavar=("R", "G", "B"))
    solid.add_argument("--apply", action="store_true", help="Actually change mode and send the frame")

    pixel = commands.add_parser("pixel", help="Build and optionally send one dim mapping pixel")
    pixel.add_argument("--mode", required=True)
    pixel.add_argument("--zone", required=True, type=_parse_zone)
    pixel.add_argument("--position", required=True, type=int)
    pixel.add_argument("--rgb", type=int, nargs=3, default=(64, 0, 0), metavar=("R", "G", "B"))
    pixel.add_argument(
        "--background", type=int, nargs=3, default=(0, 0, 0), metavar=("R", "G", "B")
    )
    pixel.add_argument("--apply", action="store_true", help="Actually change mode and send the frame")
    return parser


def _print_device_inventory(device: Any, index: int) -> None:
    layouts = _zone_layouts(device)
    print(f"Device [{index}]: {device.name}")
    print(f"LEDs:       {len(device.leds)}")
    print("Zones:")
    for zone in layouts:
        addresses = ",".join(str(value) for value in zone.addresses)
        print(
            f"  [{zone.index}] domain_start={zone.start:3d} count={zone.count:3d} "
            f"name={zone.name} addresses=[{addresses}]"
        )


def _print_inventory(controller: FullFrameController, *, all_devices: bool) -> None:
    devices = list(controller.client.devices)
    if all_devices:
        for index, device in enumerate(devices):
            _print_device_inventory(device, index)
        return
    selected_index = next(
        index for index, device in enumerate(devices) if device is controller.device
    )
    _print_device_inventory(controller.device, selected_index)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        controller = connect(
            host=args.host,
            port=args.port,
            device_index=args.device_index,
            device_name=args.device_name,
            allow_remote=args.allow_remote,
        )
        _print_inventory(controller, all_devices=args.command == "inventory")
        if args.command == "inventory":
            return 0
        if args.command == "solid":
            frame = controller.solid_frame(validate_rgb(args.rgb))
        else:
            frame = controller.pixel_frame(
                args.zone,
                args.position,
                validate_rgb(args.rgb),
                validate_rgb(args.background),
            )
        if not args.apply:
            print(f"Dry run: mode={args.mode!r}, frame_length={len(frame)}; no hardware update sent")
            return 0
        controller.select_mode(args.mode)
        controller.send_frame(frame)
        print(f"Applied mode={args.mode!r} and one full frame of {len(frame)} colors")
        return 0
    except (OpenRGBShellError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
