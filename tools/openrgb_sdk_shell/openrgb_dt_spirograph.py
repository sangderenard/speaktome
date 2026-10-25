"""Continuous radial light-field projection driven by a compiled spirograph.

Each point ``(x, y)`` generates an eight-float contribution row analytically.
The aligned radial light receives full perimeter contribution, its immediate
45-degree neighbors receive one half through ``cos(angle) ** 2``, and the
center blends to one dim uniform contribution across the ring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

import numpy as np
import sympy as sp

from openrgb_spectral_field import (
    RADIAL_ANGLES,
    RADIAL_DIRECTIONS,
    WHITE_SPECTRUM,
    radial_contributions,
    symbolic_radial_kernel,
    symbolic_spectral_display,
)
from openrgb_dt_walk import (
    AddressDomain,
    TURING_ROOT,
    address_domain_for,
    advance_managed_state,
    instantiate_managed_piece,
)
from openrgb_shell import OpenRGBShellError, connect


RADIAL_NAMES = (
    "bottom-left",
    "left",
    "top-left",
    "top",
    "top-right",
    "right",
    "bottom-right",
    "bottom",
)
def contribution_matrix(
    points: Sequence[tuple[float, float]],
    *,
    center_contribution: float = 0.25,
    falloff_power: float = 2.0,
) -> np.ndarray:
    """Evaluate the float generator at arbitrary sites, producing ``N x 8``."""

    return np.vstack([
        radial_contributions(
            x, y,
            center_contribution=center_contribution,
            falloff_power=falloff_power,
        )
        for x, y in points
    ])


def hypotrochoid_point(
    phase: float,
    *,
    fixed_radius: float = 5.0,
    rolling_radius: float = 3.0,
    pen_offset: float = 5.0,
    path_radius: float = 0.9,
) -> tuple[float, float]:
    """Return a normalized hypotrochoid point inside the unit radial field."""

    if rolling_radius == 0.0:
        raise ValueError("rolling_radius must be nonzero")
    normalizer = abs(fixed_radius - rolling_radius) + abs(pen_offset)
    if normalizer == 0.0:
        raise ValueError("hypotrochoid parameters have zero extent")
    ratio = (fixed_radius - rolling_radius) / rolling_radius
    scale = path_radius / normalizer
    return (
        scale * ((fixed_radius - rolling_radius) * math.cos(phase) + pen_offset * math.cos(ratio * phase)),
        scale * ((fixed_radius - rolling_radius) * math.sin(phase) - pen_offset * math.sin(ratio * phase)),
    )


def hypotrochoid_repeat_period(
    *,
    fixed_radius: float = 5.0,
    rolling_radius: float = 3.0,
    pen_offset: float = 5.0,
    path_radius: float = 0.9,
    repeat_epsilon: float = 1.0e-9,
    max_turns: int = 4096,
) -> float:
    """Find the first whole base turn that empirically closes the point law.

    Several phase offsets are compared so an isolated self-intersection is not
    mistaken for a repeat of the complete positioning signal.
    """

    if rolling_radius == 0.0:
        raise ValueError("rolling_radius must be nonzero")
    if repeat_epsilon <= 0.0 or not math.isfinite(repeat_epsilon):
        raise ValueError("repeat_epsilon must be finite and positive")
    probes = (0.0, 0.173, 0.619, 1.137)
    reference = tuple(
        hypotrochoid_point(
            probe,
            fixed_radius=fixed_radius,
            rolling_radius=rolling_radius,
            pen_offset=pen_offset,
            path_radius=path_radius,
        )
        for probe in probes
    )
    for turns in range(1, max_turns + 1):
        period = 2.0 * math.pi * turns
        errors = []
        for probe, expected in zip(probes, reference):
            actual = hypotrochoid_point(
                period + probe,
                fixed_radius=fixed_radius,
                rolling_radius=rolling_radius,
                pen_offset=pen_offset,
                path_radius=path_radius,
            )
            errors.append(math.hypot(actual[0] - expected[0], actual[1] - expected[1]))
        if max(errors) <= repeat_epsilon:
            return period
    raise ValueError(
        f"No hypotrochoid repeat found within {max_turns} turns at epsilon {repeat_epsilon}"
    )


def _adaptive_path_phases(
    *,
    color_bits: int = 8,
    center_contribution: float = 0.25,
    falloff_power: float = 2.0,
    fixed_radius: float = 5.0,
    rolling_radius: float = 3.0,
    pen_offset: float = 5.0,
    path_radius: float = 0.9,
    repeat_epsilon: float = 1.0e-9,
    max_sites: int | None = None,
    max_bake_seconds: float | None = None,
) -> tuple[np.ndarray, str | None]:
    """Sample until adjacent contribution rows differ by at most one code step."""

    if isinstance(color_bits, bool) or not isinstance(color_bits, int) or color_bits <= 0:
        raise ValueError("color_bits must be a positive integer")
    quantum = 1.0 / (2**color_bits - 1)
    if max_sites is not None and max_sites < 65:
        raise ValueError("max_sites must be at least 65")
    if max_bake_seconds is not None and max_bake_seconds <= 0.0:
        raise ValueError("max_bake_seconds must be positive")
    bake_started = time.perf_counter()
    cutoff_reason: str | None = None

    def row(phase: float) -> np.ndarray:
        point = hypotrochoid_point(
            phase,
            fixed_radius=fixed_radius,
            rolling_radius=rolling_radius,
            pen_offset=pen_offset,
            path_radius=path_radius,
        )
        return radial_contributions(
            *point,
            center_contribution=center_contribution,
            falloff_power=falloff_power,
        )

    phases = [0.0]

    def subdivide(a: float, b: float, va: np.ndarray, vb: np.ndarray, depth: int) -> None:
        nonlocal cutoff_reason
        if cutoff_reason is None and max_sites is not None and len(phases) >= max_sites:
            cutoff_reason = "site_budget"
        if (
            cutoff_reason is None
            and max_bake_seconds is not None
            and time.perf_counter() - bake_started >= max_bake_seconds
        ):
            cutoff_reason = "time_budget"
        if cutoff_reason is not None:
            phases.append(b)
            return
        midpoint = 0.5 * (a + b)
        vm = row(midpoint)
        whole_ok = float(np.max(np.abs(vb - va))) <= quantum
        left_ok = float(np.max(np.abs(vm - va))) <= quantum
        right_ok = float(np.max(np.abs(vb - vm))) <= quantum
        if depth >= 20 or (whole_ok and left_ok and right_ok):
            phases.append(b)
            return
        subdivide(a, midpoint, va, vm, depth + 1)
        subdivide(midpoint, b, vm, vb, depth + 1)

    repeat_period = hypotrochoid_repeat_period(
        fixed_radius=fixed_radius,
        rolling_radius=rolling_radius,
        pen_offset=pen_offset,
        path_radius=path_radius,
        repeat_epsilon=repeat_epsilon,
    )
    interval = repeat_period / 64.0
    for index in range(64):
        start = index * interval
        stop = (index + 1) * interval
        subdivide(start, stop, row(start), row(stop), 0)
    return np.asarray(phases, dtype=np.float64), cutoff_reason


def adaptive_path_phases(
    *,
    color_bits: int = 8,
    center_contribution: float = 0.25,
    falloff_power: float = 2.0,
    fixed_radius: float = 5.0,
    rolling_radius: float = 3.0,
    pen_offset: float = 5.0,
    path_radius: float = 0.9,
    repeat_epsilon: float = 1.0e-9,
    max_sites: int | None = None,
    max_bake_seconds: float | None = None,
) -> np.ndarray:
    phases, _cutoff_reason = _adaptive_path_phases(
        color_bits=color_bits,
        center_contribution=center_contribution,
        falloff_power=falloff_power,
        fixed_radius=fixed_radius,
        rolling_radius=rolling_radius,
        pen_offset=pen_offset,
        path_radius=path_radius,
        repeat_epsilon=repeat_epsilon,
        max_sites=max_sites,
        max_bake_seconds=max_bake_seconds,
    )
    return phases


@dataclass(frozen=True)
class ContributionLUT:
    phases: np.ndarray
    contributions: np.ndarray
    repeat_period: float
    color_epsilon: float
    achieved_epsilon: float
    repeat_error: float
    saturated: bool
    cutoff_reason: str | None


def bake_contribution_lut(
    *,
    color_bits: int = 8,
    center_contribution: float = 0.25,
    falloff_power: float = 2.0,
    fixed_radius: float = 5.0,
    rolling_radius: float = 3.0,
    pen_offset: float = 5.0,
    path_radius: float = 0.9,
    repeat_epsilon: float = 1.0e-9,
    max_sites: int | None = None,
    max_bake_seconds: float | None = None,
) -> ContributionLUT:
    """Bake one closed adaptive cycle of the analytic float generator."""

    phases, cutoff_reason = _adaptive_path_phases(
        color_bits=color_bits,
        center_contribution=center_contribution,
        falloff_power=falloff_power,
        fixed_radius=fixed_radius,
        rolling_radius=rolling_radius,
        pen_offset=pen_offset,
        path_radius=path_radius,
        repeat_epsilon=repeat_epsilon,
        max_sites=max_sites,
        max_bake_seconds=max_bake_seconds,
    )
    points = [
        hypotrochoid_point(
            float(phase),
            fixed_radius=fixed_radius,
            rolling_radius=rolling_radius,
            pen_offset=pen_offset,
            path_radius=path_radius,
        )
        for phase in phases
    ]
    matrix = contribution_matrix(
        points,
        center_contribution=center_contribution,
        falloff_power=falloff_power,
    )
    repeat_error = float(np.max(np.abs(matrix[-1] - matrix[0])))
    if repeat_error > repeat_epsilon:
        raise RuntimeError(
            f"Baked contribution row misses repeat epsilon: {repeat_error} > {repeat_epsilon}"
        )
    color_epsilon = 1.0 / (2**color_bits - 1)
    achieved_epsilon = float(np.max(np.abs(np.diff(matrix, axis=0))))
    saturated = achieved_epsilon <= color_epsilon + 1.0e-12
    return ContributionLUT(
        phases=phases,
        contributions=matrix,
        repeat_period=float(phases[-1]),
        color_epsilon=color_epsilon,
        achieved_epsilon=achieved_epsilon,
        repeat_error=repeat_error,
        saturated=saturated,
        cutoff_reason=None if saturated else (cutoff_reason or "depth_limit"),
    )


def symbolic_wrapped_phase(
    phase: sp.Expr,
    angular_speed: sp.Expr,
    dt: sp.Expr,
    repeat_period: sp.Expr,
) -> sp.Expr:
    """Advance phase and wrap at the empirically discovered path period."""

    advanced = phase + angular_speed * dt
    return sp.Piecewise(
        (advanced - repeat_period, advanced >= repeat_period),
        (advanced + repeat_period, advanced < 0),
        (advanced, True),
    )


@lru_cache(maxsize=1)
def compile_spirograph_law():
    """Author the continuous radial projection as simultaneous SymPy laws."""

    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    names = (
        "direction_x direction_y phase angular_speed center_contribution "
        "falloff_power fixed_radius rolling_radius pen_offset path_radius repeat_period "
        "white_level exposure x y red green blue dt"
    )
    (
        direction_x, direction_y, phase, angular_speed, center_level, falloff_power,
        fixed_radius, rolling_radius, pen_offset, path_radius, repeat_period,
        white_level, exposure, x, y, red, green, blue, dt,
    ) = sp.symbols(names)
    wrapped = symbolic_wrapped_phase(phase, angular_speed, dt, repeat_period)
    difference = fixed_radius - rolling_radius
    ratio = difference / rolling_radius
    extent = sp.Abs(difference) + sp.Abs(pen_offset)
    scale = path_radius / extent
    x_value = scale * (difference * sp.cos(wrapped) + pen_offset * sp.cos(ratio * wrapped))
    y_value = scale * (difference * sp.sin(wrapped) - pen_offset * sp.sin(ratio * wrapped))
    intensity = symbolic_radial_kernel(
        x_value, y_value, direction_x, direction_y, center_level, falloff_power,
    )
    spectrum_symbols = sp.symbols(" ".join(f"spectrum0_b{band}" for band in range(8)))
    rgb = symbolic_spectral_display(
        sp.Matrix((intensity,)),
        sp.Matrix(1, 8, spectrum_symbols),
        exposure=exposure,
        output_level=white_level,
    )
    safe_speed = sp.Max(sp.Abs(angular_speed), sp.Float(1.0e-12))
    return compile_sympy_equations(
        (
            sp.Eq(sp.Symbol("phase_next"), wrapped, evaluate=False),
            sp.Eq(sp.Symbol("x_next"), x_value, evaluate=False),
            sp.Eq(sp.Symbol("y_next"), y_value, evaluate=False),
            sp.Eq(sp.Symbol("red_next"), sp.Add(rgb[0], 0 * red, evaluate=False), evaluate=False),
            sp.Eq(sp.Symbol("green_next"), sp.Add(rgb[1], 0 * green, evaluate=False), evaluate=False),
            sp.Eq(sp.Symbol("blue_next"), sp.Add(rgb[2], 0 * blue, evaluate=False), evaluate=False),
            sp.Eq(sp.Symbol("dt_limit"), 1 / safe_speed, evaluate=False),
            sp.Eq(sp.Symbol("max_vel"), safe_speed, evaluate=False),
        ),
        name="openrgb_spectral_spirograph",
    )


def compile_spirograph_piece(*, directory: Path | None = None):
    """Compile the eight-output continuous light-field generator to LLVM."""

    from src.compiler.native_package import piece_from_law

    build = directory or (TURING_ROOT / "build" / "openrgb_spectral_spirograph" / "8")
    return piece_from_law(
        compile_spirograph_law(),
        "openrgb_spectral_spirograph",
        8,
        directory=build,
    )


class ManagedRadialSpirograph:
    """Persistent managed-dt state for the continuous eight-light field."""

    def __init__(
        self,
        domain: AddressDomain,
        *,
        frames_per_second: float,
        angular_speed: float = 1.0,
        center_contribution: float = 0.25,
        falloff_power: float = 2.0,
        white_level: int = 128,
        exposure: float = 3.0,
        fixed_radius: float = 5.0,
        rolling_radius: float = 3.0,
        pen_offset: float = 5.0,
        path_radius: float = 0.9,
        piece=None,
    ) -> None:
        if domain.count != 8:
            raise ValueError("The calibrated radial field requires exactly eight addresses")
        if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be finite and positive")
        if not np.isfinite(angular_speed) or angular_speed == 0.0:
            raise ValueError("angular_speed must be finite and nonzero")
        if not 0.0 <= center_contribution <= 1.0:
            raise ValueError("center_contribution must be in the inclusive range 0..1")
        if not np.isfinite(falloff_power) or falloff_power <= 0.0:
            raise ValueError("falloff_power must be finite and positive")
        if not 0 <= white_level <= 255:
            raise ValueError("white_level must be in the inclusive range 0..255")
        if not np.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure must be finite and positive")
        if rolling_radius == 0.0:
            raise ValueError("rolling_radius must be nonzero")
        if abs(fixed_radius - rolling_radius) + abs(pen_offset) == 0.0:
            raise ValueError("hypotrochoid parameters have zero extent")
        if not 0.0 <= path_radius <= 1.0:
            raise ValueError("path_radius must be in the inclusive range 0..1")

        self.domain = domain
        self.frame_seconds = 1.0 / float(frames_per_second)
        self.piece = piece or compile_spirograph_piece()
        count = domain.count
        repeat_period = hypotrochoid_repeat_period(
            fixed_radius=fixed_radius,
            rolling_radius=rolling_radius,
            pen_offset=pen_offset,
            path_radius=path_radius,
        )
        columns = {
            "direction_x": RADIAL_DIRECTIONS[:, 0].copy(),
            "direction_y": RADIAL_DIRECTIONS[:, 1].copy(),
            "phase": np.zeros(count, dtype=np.float64),
            "angular_speed": np.full(count, float(angular_speed), dtype=np.float64),
            "center_contribution": np.full(count, float(center_contribution), dtype=np.float64),
            "falloff_power": np.full(count, float(falloff_power), dtype=np.float64),
            "fixed_radius": np.full(count, float(fixed_radius), dtype=np.float64),
            "rolling_radius": np.full(count, float(rolling_radius), dtype=np.float64),
            "pen_offset": np.full(count, float(pen_offset), dtype=np.float64),
            "path_radius": np.full(count, float(path_radius), dtype=np.float64),
            "repeat_period": np.full(count, float(repeat_period), dtype=np.float64),
            "white_level": np.full(count, float(white_level), dtype=np.float64),
            "exposure": np.full(count, float(exposure), dtype=np.float64),
            "x": np.zeros(count, dtype=np.float64),
            "y": np.zeros(count, dtype=np.float64),
            "red": np.zeros(count, dtype=np.float64),
            "green": np.zeros(count, dtype=np.float64),
            "blue": np.zeros(count, dtype=np.float64),
        }
        for band, power in enumerate(WHITE_SPECTRUM):
            columns[f"spectrum0_b{band}"] = np.full(count, power, dtype=np.float64)
        self.state = instantiate_managed_piece(
            self.piece,
            columns,
            self.frame_seconds,
            label="openrgb-radial-spirograph",
        )

    def advance_frame(self) -> tuple[list[tuple[int, int, int]], tuple[float, float]]:
        advance_managed_state(self.state, self.frame_seconds)
        frame = self.domain.frame_from_channels(self.state.red, self.state.green, self.state.blue)
        return frame, (float(self.state.x[0]), float(self.state.y[0]))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6742)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--position-count", type=int, default=8)
    parser.add_argument("--mode", default="Direct")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--angular-speed", type=float, default=1.0)
    parser.add_argument("--center-contribution", type=float, default=0.25)
    parser.add_argument("--falloff-power", type=float, default=2.0)
    parser.add_argument("--white-level", type=int, default=128)
    parser.add_argument("--exposure", type=float, default=3.0)
    parser.add_argument("--fixed-radius", type=float, default=5.0)
    parser.add_argument("--rolling-radius", type=float, default=3.0)
    parser.add_argument("--pen-offset", type=float, default=5.0)
    parser.add_argument("--path-radius", type=float, default=0.9)
    parser.add_argument("--color-bits", type=int, default=8)
    parser.add_argument("--lut-sites-per-frame", type=int, default=8)
    parser.add_argument("--lut-max-sites", type=int)
    parser.add_argument("--lut-max-seconds", type=float, default=2.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.frames <= 0:
            raise ValueError("frames must be positive")
        controller = connect(host=args.host, port=args.port, device_name=args.device_name)
        try:
            zone_selector: int | str = int(args.zone)
        except ValueError:
            zone_selector = args.zone
        domain = address_domain_for(
            controller,
            zone_selector,
            position_count=args.position_count,
        )
        repeat_period = hypotrochoid_repeat_period(
            fixed_radius=args.fixed_radius,
            rolling_radius=args.rolling_radius,
            pen_offset=args.pen_offset,
            path_radius=args.path_radius,
        )
        temporal_sites = math.ceil(repeat_period * args.fps / abs(args.angular_speed))
        dynamic_site_budget = max(65, temporal_sites * args.lut_sites_per_frame)
        if args.lut_max_sites is not None:
            dynamic_site_budget = min(dynamic_site_budget, args.lut_max_sites)
        lut = bake_contribution_lut(
            color_bits=args.color_bits,
            center_contribution=args.center_contribution,
            falloff_power=args.falloff_power,
            fixed_radius=args.fixed_radius,
            rolling_radius=args.rolling_radius,
            pen_offset=args.pen_offset,
            path_radius=args.path_radius,
            max_sites=dynamic_site_budget,
            max_bake_seconds=args.lut_max_seconds,
        )
        effect = ManagedRadialSpirograph(
            domain,
            frames_per_second=args.fps,
            angular_speed=args.angular_speed,
            center_contribution=args.center_contribution,
            falloff_power=args.falloff_power,
            white_level=args.white_level,
            exposure=args.exposure,
            fixed_radius=args.fixed_radius,
            rolling_radius=args.rolling_radius,
            pen_offset=args.pen_offset,
            path_radius=args.path_radius,
        )
        if args.apply:
            controller.select_mode(args.mode)

        solve_times: list[float] = []
        send_times: list[float] = []
        deadline_lateness: list[float] = []
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        point = (0.0, 0.0)
        for frame_number in range(args.frames):
            solve_started = time.perf_counter()
            frame, point = effect.advance_frame()
            solve_times.append(time.perf_counter() - solve_started)
            if args.apply:
                send_started = time.perf_counter()
                controller.send_frame(frame)
                send_times.append(time.perf_counter() - send_started)
            deadline = wall_started + (frame_number + 1) * effect.frame_seconds
            completed = time.perf_counter()
            deadline_lateness.append(max(0.0, completed - deadline))
            if args.realtime or args.apply:
                time.sleep(max(0.0, deadline - completed))
        elapsed = time.perf_counter() - wall_started
        cpu_elapsed = time.process_time() - cpu_started
        achieved = args.frames / elapsed
        total_work = [solve + (send_times[i] if send_times else 0.0) for i, solve in enumerate(solve_times)]
        print(
            "benchmark "
            f"frames={args.frames} target_hz={args.fps:.3f} achieved_hz={achieved:.3f} "
            f"deadline_misses={sum(value > 0.0 for value in deadline_lateness)} "
            f"work_ms_p50={statistics.median(total_work) * 1000.0:.3f} "
            f"work_ms_p95={_percentile(total_work, 0.95) * 1000.0:.3f} "
            f"work_ms_max={max(total_work) * 1000.0:.3f} "
            f"solve_ms_p95={_percentile(solve_times, 0.95) * 1000.0:.3f} "
            f"send_ms_p95={_percentile(send_times, 0.95) * 1000.0:.3f} "
            f"process_cpu_percent={100.0 * cpu_elapsed / elapsed:.2f} "
            f"adaptive_sites_{args.color_bits}bit={len(lut.phases) - 1} "
            f"lut_site_budget={dynamic_site_budget} lut_saturated={str(lut.saturated).lower()} "
            f"lut_achieved_epsilon={lut.achieved_epsilon:.6g} "
            f"lut_cutoff={lut.cutoff_reason or 'none'} "
            f"repeat_period={lut.repeat_period:.6f} repeat_error={lut.repeat_error:.3e} "
            f"last_point=({point[0]:.6f},{point[1]:.6f})"
        )
        if not args.apply:
            print("Dry run: no hardware update sent")
        return 0
    except (OpenRGBShellError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
