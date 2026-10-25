"""Direct native HID transport for the ASRock Polychrome USB framebuffer."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char, c_double, c_int, c_int64, c_size_t, c_ubyte, c_uint64, c_ushort, c_void_p
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

from asrock_hid_abi import ASRockHIDExternalZoneEffect, ASRockHIDUniformColor
from openrgb_shell import DomainAddress, OpenRGBShellError, RGB, ZoneLayout, validate_rgb


DEFAULT_NATIVE_DLL = Path(__file__).with_name("native_asrock_hid") / "build" / "Release" / "asrock_polychrome_hid.dll"
DEFAULT_HIDAPI_DLL = Path(r"C:\Users\alber\AppData\Local\Temp\OpenRGB-1.0-81bbe18\OpenRGB Windows 64-bit\hidapi-hotplug.dll")


@dataclass(frozen=True)
class DirectZone:
    name: str
    zone_type: int
    count: int
    maximum_count: int
    rg_swap: bool


NativeZoneEffect = ASRockHIDExternalZoneEffect
NativeUniformColor = ASRockHIDUniformColor


class DirectASRockHIDController:
    """Full-frame-compatible controller backed by one native HID call per frame."""

    def __init__(
        self,
        zones: Iterable[DirectZone],
        *,
        native_dll: str | Path = DEFAULT_NATIVE_DLL,
        hidapi_dll: str | Path = DEFAULT_HIDAPI_DLL,
        vendor_id: int = 0x26CE,
        product_id: int = 0x01A2,
        response_timeout_ms: int = 100,
    ):
        self._zone_specs = tuple(zones)
        if not self._zone_specs:
            raise ValueError("direct HID transport requires at least one zone")
        self._layouts = self._make_layouts(self._zone_specs)
        self._library = ctypes.WinDLL(str(Path(native_dll).resolve()))
        self._bind()
        zone_count = len(self._zone_specs)
        zone_types = (c_ubyte * zone_count)(*(zone.zone_type for zone in self._zone_specs))
        counts = (c_ushort * zone_count)(*(zone.count for zone in self._zone_specs))
        maxima = (c_ushort * zone_count)(*(zone.maximum_count for zone in self._zone_specs))
        swaps = (c_ubyte * zone_count)(*(int(zone.rg_swap) for zone in self._zone_specs))
        error = (c_char * 1024)()
        self._handle = self._library.asrock_hid_open(
            str(Path(hidapi_dll).resolve()), vendor_id, product_id,
            zone_types, counts, maxima, swaps, zone_count,
            response_timeout_ms, error, len(error),
        )
        if not self._handle:
            raise OpenRGBShellError(self._error_text(error))
        self.report_count = int(self._library.asrock_hid_report_count(self._handle))
        self._packed_bytes = bytearray(3 * self.led_count)
        self._packed_view = (c_ubyte * len(self._packed_bytes)).from_buffer(self._packed_bytes)
        self._metric_sample_time = time.perf_counter()
        self._metric_sample_completed = 0
        self._recent_completion_hz = 0.0
        self.packet_mode = "burst"
        self.message_type = "stream"
        self.packet_report_count = self.report_count
        self.packet_window_size = self.report_count

    def _bind(self) -> None:
        library = self._library
        library.asrock_hid_open.argtypes = [
            ctypes.c_wchar_p, c_ushort, c_ushort,
            POINTER(c_ubyte), POINTER(c_ushort), POINTER(c_ushort), POINTER(c_ubyte),
            c_size_t, c_int, POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_open.restype = c_void_p
        library.asrock_hid_select_direct.argtypes = [c_void_p, POINTER(c_char), c_size_t]
        library.asrock_hid_select_direct.restype = c_int
        library.asrock_hid_read_external_zone_effect_message.argtypes = [
            c_void_p, POINTER(ASRockHIDExternalZoneEffect), POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_read_external_zone_effect_message.restype = c_int
        library.asrock_hid_write_external_zone_effect_message.argtypes = [
            c_void_p, POINTER(ASRockHIDExternalZoneEffect), POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_write_external_zone_effect_message.restype = c_int
        library.asrock_hid_send_uniform_color.argtypes = [
            c_void_p, POINTER(ASRockHIDUniformColor), POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_send_uniform_color.restype = c_int
        library.asrock_hid_send_frame.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_send_frame.restype = c_int
        library.asrock_hid_clock_ns.argtypes = []
        library.asrock_hid_clock_ns.restype = c_uint64
        library.asrock_hid_send_frame_at.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, c_uint64, c_int,
            POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_send_frame_at.restype = c_int
        library.asrock_hid_cancel_scheduled.argtypes = [c_void_p]
        library.asrock_hid_cancel_scheduled.restype = None
        library.asrock_hid_timing_metrics.argtypes = [
            c_void_p, POINTER(c_uint64), POINTER(c_int64), POINTER(c_int64),
            POINTER(c_uint64), POINTER(c_size_t), POINTER(c_uint64),
            POINTER(c_uint64), POINTER(c_int),
        ]
        library.asrock_hid_timing_metrics.restype = c_int
        library.asrock_hid_report_count.argtypes = [c_void_p]
        library.asrock_hid_report_count.restype = c_size_t
        library.asrock_hid_metrics.argtypes = [
            c_void_p,
            POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64),
            POINTER(c_int), POINTER(c_int), POINTER(c_double),
            POINTER(c_size_t), POINTER(c_size_t), POINTER(c_int),
        ]
        library.asrock_hid_metrics.restype = c_int
        library.asrock_hid_configure_packets.argtypes = [
            c_void_p, c_size_t, c_size_t, c_int, POINTER(c_char), c_size_t,
        ]
        library.asrock_hid_configure_packets.restype = c_int
        library.asrock_hid_close.argtypes = [c_void_p]
        library.asrock_hid_close.restype = None

    @staticmethod
    def _error_text(error) -> str:
        return bytes(error).split(b"\0", 1)[0].decode("utf-8", errors="replace") or "direct ASRock HID operation failed"

    @staticmethod
    def _make_layouts(zones: tuple[DirectZone, ...]) -> tuple[ZoneLayout, ...]:
        layouts = []
        start = 0
        for index, zone in enumerate(zones):
            indices = tuple(range(start, start + zone.count))
            layouts.append(ZoneLayout(index, zone.name, start, zone.count, tuple(range(zone.count)), indices))
            start += zone.count
        return tuple(layouts)

    @property
    def led_count(self) -> int:
        return sum(zone.count for zone in self._zone_specs)

    @property
    def zones(self) -> tuple[ZoneLayout, ...]:
        return self._layouts

    @property
    def address_domain(self) -> tuple[DomainAddress, ...]:
        return tuple(
            DomainAddress(
                domain_index=zone.start + position,
                frame_index=zone.start + position,
                system_address=position,
                region_index=zone.index,
                region_name=zone.name,
                region_position=position,
            )
            for zone in self._layouts
            for position in range(zone.count)
        )

    def zone(self, selector: int | str) -> ZoneLayout:
        if isinstance(selector, int):
            try:
                return self._layouts[selector]
            except IndexError as exc:
                raise OpenRGBShellError(f"Zone index {selector} is out of range") from exc
        for zone in self._layouts:
            if zone.name == selector:
                return zone
        raise OpenRGBShellError(f"Unknown direct HID zone {selector!r}")

    def blank_frame(self, background: RGB = (0, 0, 0)) -> list[RGB]:
        color = validate_rgb(background)
        return [color] * self.led_count

    def select_mode(self, mode: str) -> None:
        if mode.casefold() != "direct":
            raise OpenRGBShellError("direct ASRock HID transport supports only Direct mode")
        error = (c_char * 1024)()
        if not self._library.asrock_hid_select_direct(self._handle, error, len(error)):
            raise OpenRGBShellError(self._error_text(error))

    def read_external_zone_effect(self) -> NativeZoneEffect:
        """Read only physical zone 0x07 (Audio/ARGB3)."""
        effect = NativeZoneEffect()
        error = (c_char * 1024)()
        if not self._library.asrock_hid_read_external_zone_effect_message(
            self._handle, ctypes.byref(effect), error, len(error),
        ):
            raise OpenRGBShellError(self._error_text(error))
        return effect.validate()

    def write_external_zone_effect(self, effect: NativeZoneEffect) -> None:
        """Write zone 0x07 once without sending the explicit commit command."""
        if not isinstance(effect, NativeZoneEffect):
            raise TypeError("effect must be a NativeZoneEffect ABI message")
        effect.validate()
        error = (c_char * 1024)()
        if not self._library.asrock_hid_write_external_zone_effect_message(
            self._handle, ctypes.byref(effect), error, len(error),
        ):
            raise OpenRGBShellError(self._error_text(error))

    def send_uniform_color(self, message: NativeUniformColor) -> None:
        """Queue a one-report, whole-zone color through the native HID owner."""
        if not isinstance(message, NativeUniformColor):
            raise TypeError("message must be a NativeUniformColor ABI message")
        message.validate()
        error = (c_char * 1024)()
        if not self._library.asrock_hid_send_uniform_color(
            self._handle, ctypes.byref(message), error, len(error),
        ):
            raise OpenRGBShellError(self._error_text(error))

    def _pack_frame(self, frame: Iterable[RGB]) -> int:
        count = 0
        for index, source_color in enumerate(frame):
            if index >= self.led_count:
                raise OpenRGBShellError(
                    f"Frame has more than {self.led_count} colors"
                )
            color = validate_rgb(source_color)
            base = 3 * index
            self._packed_bytes[base] = color[0]
            self._packed_bytes[base + 1] = color[1]
            self._packed_bytes[base + 2] = color[2]
            count = index + 1
        if count != self.led_count:
            raise OpenRGBShellError(
                f"Frame has {count} colors; direct controller requires {self.led_count}"
            )
        return count

    def send_frame(self, frame: Iterable[RGB], *, fast: bool = True) -> None:
        del fast
        count = self._pack_frame(frame)
        error = (c_char * 1024)()
        if not self._library.asrock_hid_send_frame(
            self._handle, self._packed_view, count, error, len(error)
        ):
            raise OpenRGBShellError(self._error_text(error))

    def clock_ns(self) -> int:
        return int(self._library.asrock_hid_clock_ns())

    def send_frame_at(
        self, frame: Iterable[RGB], deadline_ns: int, *, event_kind: str = "frame",
    ) -> None:
        count = self._pack_frame(frame)
        event_kinds = {"frame": 0, "on": 1, "off": 2}
        try:
            native_event_kind = event_kinds[event_kind]
        except KeyError as exc:
            raise ValueError(f"unknown scheduled event kind {event_kind!r}") from exc
        error = (c_char * 1024)()
        if not self._library.asrock_hid_send_frame_at(
            self._handle, self._packed_view, count, int(deadline_ns),
            native_event_kind, error, len(error),
        ):
            raise OpenRGBShellError(self._error_text(error))

    def cancel_scheduled(self) -> None:
        self._library.asrock_hid_cancel_scheduled(self._handle)

    def timing_metrics(self) -> dict[str, int]:
        dispatched = c_uint64()
        last_error = c_int64()
        maximum_error = c_int64()
        transmit_duration = c_uint64()
        queued = c_size_t()
        on_dispatched = c_uint64()
        off_dispatched = c_uint64()
        last_event_kind = c_int()
        if not self._library.asrock_hid_timing_metrics(
            self._handle, ctypes.byref(dispatched), ctypes.byref(last_error),
            ctypes.byref(maximum_error), ctypes.byref(transmit_duration),
            ctypes.byref(queued), ctypes.byref(on_dispatched),
            ctypes.byref(off_dispatched), ctypes.byref(last_event_kind),
        ):
            raise OpenRGBShellError("could not read direct HID timing metrics")
        return {
            "scheduled_dispatched": dispatched.value,
            "last_schedule_error_ns": last_error.value,
            "maximum_schedule_error_ns": maximum_error.value,
            "last_transmit_duration_ns": transmit_duration.value,
            "scheduled_queued": queued.value,
            "scheduled_on_dispatched": on_dispatched.value,
            "scheduled_off_dispatched": off_dispatched.value,
            "last_scheduled_event_kind": last_event_kind.value,
        }

    def metrics(self) -> dict[str, int | float | bool]:
        submitted = c_uint64()
        completed = c_uint64()
        dropped = c_uint64()
        failed = c_uint64()
        busy = c_int()
        pending = c_int()
        average_hz = c_double()
        active_reports = c_size_t()
        acknowledgement_window = c_size_t()
        message_pattern = c_int()
        if not self._library.asrock_hid_metrics(
            self._handle,
            ctypes.byref(submitted), ctypes.byref(completed), ctypes.byref(dropped),
            ctypes.byref(failed), ctypes.byref(busy), ctypes.byref(pending),
            ctypes.byref(average_hz), ctypes.byref(active_reports),
            ctypes.byref(acknowledgement_window), ctypes.byref(message_pattern),
        ):
            raise OpenRGBShellError("could not read direct HID transport metrics")
        now = time.perf_counter()
        elapsed = now - self._metric_sample_time
        if elapsed >= 0.5:
            self._recent_completion_hz = (
                (completed.value - self._metric_sample_completed) / elapsed
            )
            self._metric_sample_time = now
            self._metric_sample_completed = completed.value
        result = {
            "submitted": submitted.value,
            "completed": completed.value,
            "dropped": dropped.value,
            "failed": failed.value,
            "busy": bool(busy.value),
            "pending": bool(pending.value),
            "completion_hz": self._recent_completion_hz,
            "average_completion_hz": average_hz.value,
            "report_count": self.report_count,
            "active_reports": active_reports.value,
            "acknowledgement_window": acknowledgement_window.value,
            "packet_mode": self.packet_mode,
            "message_type": self.message_type,
        }
        result.update(self.timing_metrics())
        return result

    def configure_packets(
        self,
        *,
        mode: str | None = None,
        message_type: str | None = None,
        report_count: int | None = None,
        window_size: int | None = None,
    ) -> None:
        mode = self.packet_mode if mode is None else mode
        if mode not in {"serial", "windowed", "burst"}:
            raise ValueError("packet mode must be serial, windowed, or burst")
        message_type = self.message_type if message_type is None else message_type
        message_patterns = {"stream": 0, "restart": 1, "continue": 2}
        if message_type not in message_patterns:
            raise ValueError("message type must be stream, restart, or continue")
        report_count = self.packet_report_count if report_count is None else int(report_count)
        report_count = min(self.report_count, max(1, report_count))
        requested_window = self.packet_window_size if window_size is None else int(window_size)
        preferred_window = min(report_count, max(2, requested_window))
        if mode == "serial":
            acknowledgement_window = 1
        elif mode == "burst":
            acknowledgement_window = report_count
        else:
            acknowledgement_window = preferred_window
        error = (c_char * 1024)()
        if not self._library.asrock_hid_configure_packets(
            self._handle, report_count, acknowledgement_window,
            message_patterns[message_type], error, len(error)
        ):
            raise OpenRGBShellError(self._error_text(error))
        self.packet_mode = mode
        self.message_type = message_type
        self.packet_report_count = report_count
        self.packet_window_size = preferred_window
        snapshot = self.metrics()
        self._metric_sample_time = time.perf_counter()
        self._metric_sample_completed = int(snapshot["completed"])
        self._recent_completion_hz = 0.0

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.asrock_hid_close(self._handle)
            self._handle = None


def connect_direct(config: dict) -> DirectASRockHIDController:
    transport = config["controller"]["transport"]
    zones = tuple(
        DirectZone(
            name=str(zone["name"]),
            zone_type=int(zone["zone_type"]),
            count=int(zone["count"]),
            maximum_count=int(zone["maximum_count"]),
            rg_swap=bool(zone["rg_swap"]),
        )
        for zone in transport["zones"]
    )
    controller = DirectASRockHIDController(
        zones,
        native_dll=transport.get("native_dll", DEFAULT_NATIVE_DLL),
        hidapi_dll=transport.get("hidapi_dll", DEFAULT_HIDAPI_DLL),
        vendor_id=int(transport.get("vendor_id", 0x26CE)),
        product_id=int(transport.get("product_id", 0x01A2)),
        response_timeout_ms=int(transport.get("response_timeout_ms", 100)),
    )
    controller.configure_packets(
        mode=str(transport.get("packet_mode", "burst")),
        message_type=str(transport.get("message_type", "stream")),
        report_count=int(transport.get("report_count", controller.report_count)),
        window_size=int(transport.get("window_size", 4)),
    )
    return controller


__all__ = [
    "DirectASRockHIDController", "DirectZone", "NativeZoneEffect",
    "NativeUniformColor", "connect_direct",
]
