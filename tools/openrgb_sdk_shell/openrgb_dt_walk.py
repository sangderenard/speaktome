"""SymPy -> AbstractTensor -> LLVM linear walk under Turing managed dt.

OpenRGB owns an arbitrary collection of device addresses and named regions.
This module projects that system layout onto a dense numerical domain, then
lets one simultaneous SymPy law compute the full RGB state for every domain
point.  The law is compiled through Turing's sanctioned native piece route and
is advanced only by Turing's instantiated dt system.

Importing this module does not connect to OpenRGB or write hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import sympy as sp

from openrgb_shell import DomainAddress, OpenRGBShellError, connect


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
TURING_ROOT = WORKSPACE / "turing"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))
if str(TURING_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(TURING_ROOT / "examples"))


@dataclass(frozen=True)
class AddressDomain:
    """Dense compiler coordinates plus preserved OpenRGB identities."""

    addresses: tuple[DomainAddress, ...]
    frame_count: int | None = None

    def __post_init__(self) -> None:
        if not self.addresses:
            raise ValueError("The OpenRGB address domain is empty")
        expected = tuple(range(len(self.addresses)))
        actual = tuple(address.domain_index for address in self.addresses)
        if actual != expected:
            raise ValueError(f"Domain indices must be dense 0..{len(expected) - 1}")
        frames = tuple(address.frame_index for address in self.addresses)
        if len(set(frames)) != len(frames):
            raise ValueError("Each domain address must map to one unique frame slot")
        frame_count = max(frames) + 1 if self.frame_count is None else self.frame_count
        if frame_count <= 0 or any(frame < 0 or frame >= frame_count for frame in frames):
            raise ValueError("Frame indices must fit inside the full controller frame")
        object.__setattr__(self, "frame_count", frame_count)

    @property
    def count(self) -> int:
        return len(self.addresses)

    def frame_from_channels(
        self,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        channels = tuple(np.asarray(value, dtype=np.float64) for value in (red, green, blue))
        if any(value.shape != (self.count,) for value in channels):
            raise ValueError(f"RGB columns must each have shape ({self.count},)")
        frame = [(0, 0, 0) for _ in range(self.frame_count)]
        for address, values in zip(self.addresses, zip(*channels)):
            frame[address.frame_index] = tuple(
                int(np.clip(np.rint(value), 0, 255)) for value in values
            )
        return frame


def address_domain_for(
    controller,
    zone_selector: int | str | None = None,
    *,
    position_count: int | None = None,
) -> AddressDomain:
    """Project a controller or one selected zone onto a dense law domain."""

    if position_count is not None:
        if zone_selector is None:
            raise ValueError("position_count requires a selected zone")
        if isinstance(position_count, bool) or position_count <= 0:
            raise ValueError("position_count must be a positive integer")
    addresses = controller.address_domain
    if zone_selector is not None:
        zone = controller.zone(zone_selector)
        zone_addresses = tuple(
            address
            for address in addresses
            if address.region_index == zone.index
            and (position_count is None or address.region_position < position_count)
        )
        if position_count is not None and len(zone_addresses) != position_count:
            raise ValueError(
                f"Zone {zone.name!r} has {len(zone_addresses)} of the requested "
                f"{position_count} positions"
            )
        addresses = tuple(
            DomainAddress(
                domain_index=domain_index,
                frame_index=address.frame_index,
                system_address=address.system_address,
                region_index=address.region_index,
                region_name=address.region_name,
                region_position=address.region_position,
            )
            for domain_index, address in enumerate(zone_addresses)
        )
    return AddressDomain(addresses, frame_count=controller.led_count)


@lru_cache(maxsize=1)
def compile_linear_walk_law():
    """Author the one-point white walk as simultaneous SymPy equations."""

    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    domain_index, cursor, address_count, speed, white_level, red, green, blue, dt = sp.symbols(
        "domain_index cursor address_count speed white_level red green blue dt"
    )
    advanced = cursor + speed * dt
    wrapped = sp.Piecewise(
        (advanced - address_count, advanced >= address_count),
        (advanced, True),
    )
    active = sp.Abs(domain_index - wrapped) < sp.Rational(1, 2)
    white = sp.Piecewise((white_level, active), (sp.Integer(0), True))
    # These zero-weight terms declare the mutable RGB columns as law inputs.
    # The dt system derives its state fields from piece arguments and owns the
    # corresponding ``*_next`` writes; no separate output buffer convention is
    # introduced here.
    red_value = sp.Add(white, sp.Mul(0, red, evaluate=False), evaluate=False)
    green_value = sp.Add(white, sp.Mul(0, green, evaluate=False), evaluate=False)
    blue_value = sp.Add(white, sp.Mul(0, blue, evaluate=False), evaluate=False)
    safe_speed = sp.Max(sp.Abs(speed), sp.Float(1.0e-12))
    return compile_sympy_equations(
        (
            sp.Eq(sp.Symbol("cursor_next"), wrapped, evaluate=False),
            sp.Eq(sp.Symbol("red_next"), red_value, evaluate=False),
            sp.Eq(sp.Symbol("green_next"), green_value, evaluate=False),
            sp.Eq(sp.Symbol("blue_next"), blue_value, evaluate=False),
            sp.Eq(sp.Symbol("dt_limit"), 1 / safe_speed, evaluate=False),
            sp.Eq(sp.Symbol("max_vel"), safe_speed, evaluate=False),
        ),
        name="openrgb_linear_walk",
    )


def compile_linear_walk_piece(count: int, *, directory: Path | None = None):
    """Compile the law with the controller-discovered domain as its batch."""

    from src.compiler.native_package import piece_from_law

    if count <= 0:
        raise ValueError("Address count must be positive")
    build = directory or (TURING_ROOT / "build" / "openrgb_dt_walk" / str(count))
    return piece_from_law(
        compile_linear_walk_law(),
        "openrgb_linear_walk",
        int(count),
        directory=build,
    )


def instantiate_managed_piece(piece, columns, frame_seconds: float, *, label: str):
    """Instantiate one repository dt graph around an already-compiled piece."""

    from src.common.dt_system.dt import SuperstepPlan
    from src.common.dt_system.dt_controller import STController, Targets
    from src.common.dt_system.dt_graph import ControllerNode, RoundNode
    from llvm_dt_system import instantiate_system, piece_leaf

    targets = Targets(
        cfl=1.0,
        div_max=1.0e9,
        mass_max=1.0e-3,
        energy_exchange_fraction=0.2,
    )
    graph = RoundNode(
        plan=SuperstepPlan(round_max=frame_seconds, dt_init=frame_seconds),
        controller=ControllerNode(
            ctrl=STController(dt_min=frame_seconds * 1.0e-6),
            targets=targets,
            dx=1.0,
        ),
        children=[piece_leaf(piece, label=label)],
        schedule="sequential",
        label=f"{label}-frame",
    )
    return instantiate_system(graph, columns)


def advance_managed_state(state, frame_seconds: float):
    """Advance exactly one display window through the repository dt system."""

    from llvm_dt_system import advance_round

    advanced, dt_next, telemetry = advance_round(state, window=frame_seconds)
    if not np.isclose(float(advanced), frame_seconds, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"dt system advanced {float(advanced)!r} of {frame_seconds!r} seconds"
        )
    return dt_next, telemetry


class ManagedLinearWalk:
    """One persistent Turing dt-system state for an OpenRGB address domain."""

    def __init__(
        self,
        domain: AddressDomain,
        *,
        frames_per_second: float,
        white_level: int = 255,
        start_position: int = 0,
        piece=None,
    ) -> None:
        if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be finite and positive")
        if isinstance(white_level, bool) or not isinstance(white_level, int):
            raise TypeError("white_level must be an integer")
        if white_level < 0 or white_level > 255:
            raise ValueError("white_level must be in the inclusive range 0..255")
        if isinstance(start_position, bool) or not isinstance(start_position, int):
            raise TypeError("start_position must be an integer")
        if start_position < 0 or start_position >= domain.count:
            raise ValueError(
                f"start_position must be in the inclusive range 0..{domain.count - 1}"
            )

        self.domain = domain
        self.frame_seconds = 1.0 / float(frames_per_second)
        self.piece = piece or compile_linear_walk_piece(domain.count)
        count = domain.count
        columns = {
            "domain_index": np.arange(count, dtype=np.float64),
            "cursor": np.full(count, float(start_position - 1), dtype=np.float64),
            "address_count": np.full(count, float(count), dtype=np.float64),
            "speed": np.full(count, float(frames_per_second), dtype=np.float64),
            "white_level": np.full(count, float(white_level), dtype=np.float64),
            "red": np.zeros(count, dtype=np.float64),
            "green": np.zeros(count, dtype=np.float64),
            "blue": np.zeros(count, dtype=np.float64),
        }
        self.state = instantiate_managed_piece(
            self.piece,
            columns,
            self.frame_seconds,
            label="openrgb-linear-walk",
        )

    def advance_frame(self) -> list[tuple[int, int, int]]:
        """Land one display-frame window and return its full RGB frame."""

        advance_managed_state(self.state, self.frame_seconds)
        return self.domain.frame_from_channels(
            self.state.red,
            self.state.green,
            self.state.blue,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6742)
    parser.add_argument("--allow-remote", action="store_true")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--device-index", type=int)
    selector.add_argument("--device-name")
    parser.add_argument(
        "--zone",
        help="Restrict the dense law domain to one exact zone name or numeric zone index",
    )
    parser.add_argument(
        "--position-count",
        type=int,
        help="Use the first N measured positions of the selected zone",
    )
    parser.add_argument("--mode", default="Direct")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--white-level", type=int, default=255)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    return parser


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
        zone_selector: int | str | None = None
        if args.zone is not None:
            try:
                zone_selector = int(args.zone)
            except ValueError:
                zone_selector = args.zone
        domain = address_domain_for(
            controller,
            zone_selector,
            position_count=args.position_count,
        )
        frames = domain.count if args.frames is None else args.frames
        if frames <= 0:
            raise ValueError("frames must be positive")
        walk = ManagedLinearWalk(
            domain,
            frames_per_second=args.fps,
            white_level=args.white_level,
            start_position=args.start_position,
        )
        if args.apply:
            controller.select_mode(args.mode)
        started = time.monotonic()
        for frame_number in range(frames):
            frame = walk.advance_frame()
            active = next(
                (
                    address
                    for address in domain.addresses
                    if frame[address.frame_index] != (0, 0, 0)
                ),
                None,
            )
            if args.apply:
                controller.send_frame(frame)
                deadline = started + (frame_number + 1) * walk.frame_seconds
                time.sleep(max(0.0, deadline - time.monotonic()))
            elif active is not None:
                print(
                    f"frame={frame_number} domain={active.domain_index} "
                    f"region={active.region_name!r} region_position={active.region_position} "
                    f"system_address={active.system_address} frame_index={active.frame_index}"
                )
        if not args.apply:
            print(f"Dry run: solved {frames} frame(s) through managed dt; no hardware update sent")
        return 0
    except (OpenRGBShellError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
