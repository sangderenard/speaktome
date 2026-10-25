"""Compiled intrinsic-torus three-body CMY field for the eight-light ring."""

from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from openrgb_spectral_field import (
    CMY_SPECTRA,
    RADIAL_DIRECTIONS,
    symbolic_radial_kernel,
    symbolic_spectral_display,
)
from openrgb_dt_walk import AddressDomain, TURING_ROOT, address_domain_for, advance_managed_state
from openrgb_cpu_temperature import (
    CpuTemperatureReader,
    DEFAULT_ASSEMBLY,
    DEFAULT_SENSOR_IDENTIFIER,
)
from openrgb_shell import OpenRGBShellError, connect
from openrgb_torus_green import (
    build_torus_green_kernel,
    evaluate_green,
    symbolic_green_basis,
)


BODY_COUNT = 3
STATE_NAMES = tuple(
    f"{field}{index}"
    for index in range(BODY_COUNT)
    for field in ("u", "v", "du", "dv")
)
TORUS_ERROR_LIMITS = {
    "three_body_energy_relative_error": 1.0e-9,
    "three_body_momentum_relative_error": 1.0e-10,
}
PHYSICS_GEOMETRIES = ("torus", "sphere", "cartesian")


def _law_names(geometry: str) -> tuple[str, ...]:
    if geometry not in PHYSICS_GEOMETRIES:
        raise ValueError(f"unknown physics geometry {geometry!r}")
    return tuple(
        [f"openrgb_{geometry}_three_body_stage{index}" for index in range(4)]
        + [f"openrgb_{geometry}_three_body_endpoint"]
    )

def torus_face_projection(u, v, major_radius, minor_radius, scale):
    """Orthographically project a face-on embedded torus inside the LED ring."""

    radius = scale * (major_radius + minor_radius * sp.cos(v)) / (
        major_radius + minor_radius
    )
    return radius * sp.cos(u), radius * sp.sin(u)


def _torus_dynamics(_stage_time, state: Mapping[str, sp.Expr]):
    """Coupled intrinsic equations in torus coordinates."""

    gravity = sp.Symbol("gravity")
    major_radius = sp.Symbol("major_radius")
    minor_radius = sp.Symbol("minor_radius")
    kernel = build_torus_green_kernel()
    inverse_green = sp.ImmutableDenseMatrix(kernel.inverse_stiffness.tolist())
    masses = [sp.Symbol(f"mass{i}") for i in range(BODY_COUNT)]
    acceleration_u = []
    acceleration_v = []
    for index in range(BODY_COUNT):
        v = state[f"v{index}"]
        du = state[f"du{index}"]
        dv = state[f"dv{index}"]
        tube = major_radius + minor_radius * sp.cos(v)
        # Christoffel terms for diag((R+r cos(v))^2, r^2).
        acceleration_u.append(2 * minor_radius * sp.sin(v) * du * dv / tube)
        acceleration_v.append(-tube * sp.sin(v) * du**2 / minor_radius)

    for left in range(BODY_COUNT):
        for right in range(left + 1, BODY_COUNT):
            ul, vl = state[f"u{left}"], state[f"v{left}"]
            ur, vr = state[f"u{right}"], state[f"v{right}"]
            basis_l, basis_l_u, basis_l_v = symbolic_green_basis(ul, vl, kernel)
            basis_r, basis_r_u, basis_r_v = symbolic_green_basis(ur, vr, kernel)
            response_r = inverse_green * basis_r
            response_l = inverse_green * basis_l
            for body, other, v, green_gradient_u, green_gradient_v in (
                (left, right, vl, basis_l_u.dot(response_r), basis_l_v.dot(response_r)),
                (right, left, vr, basis_r_u.dot(response_l), basis_r_v.dot(response_l)),
            ):
                tube = major_radius + minor_radius * sp.cos(v)
                acceleration_u[body] += (
                    gravity * masses[other] * green_gradient_u / tube**2
                )
                acceleration_v[body] += (
                    gravity * masses[other] * green_gradient_v / minor_radius**2
                )

    rates: dict[str, sp.Expr] = {}
    for index in range(BODY_COUNT):
        rates[f"u{index}"] = state[f"du{index}"]
        rates[f"v{index}"] = state[f"dv{index}"]
        rates[f"du{index}"] = acceleration_u[index]
        rates[f"dv{index}"] = acceleration_v[index]
    return rates


def _sphere_green(left_u, left_v, right_u, right_v):
    """Zero-mean sphere Green kernel up to its irrelevant additive constant."""

    softening = sp.Symbol("softening")
    cosine_distance = (
        sp.sin(left_v) * sp.sin(right_v)
        + sp.cos(left_v) * sp.cos(right_v) * sp.cos(left_u - right_u)
    )
    return -sp.log(1 - cosine_distance + softening**2) / (4 * sp.pi)


def _cartesian_green(left_u, left_v, right_u, right_v):
    softening = sp.Symbol("softening")
    return 1 / sp.sqrt(
        (left_u - right_u) ** 2 + (left_v - right_v) ** 2 + softening**2
    )


def _non_torus_dynamics(geometry: str, state: Mapping[str, sp.Expr]):
    gravity = sp.Symbol("gravity")
    radius = sp.Symbol("surface_radius")
    masses = [sp.Symbol(f"mass{i}") for i in range(BODY_COUNT)]
    acceleration_u = [sp.Integer(0) for _ in range(BODY_COUNT)]
    acceleration_v = [sp.Integer(0) for _ in range(BODY_COUNT)]
    if geometry == "sphere":
        for index in range(BODY_COUNT):
            latitude = state[f"v{index}"]
            du = state[f"du{index}"]
            dv = state[f"dv{index}"]
            acceleration_u[index] = 2 * sp.tan(latitude) * du * dv
            acceleration_v[index] = -sp.sin(latitude) * sp.cos(latitude) * du**2

    green_function = _sphere_green if geometry == "sphere" else _cartesian_green
    left_u, left_v, right_u, right_v = sp.symbols(
        "green_left_u green_left_v green_right_u green_right_v"
    )
    green_template = green_function(left_u, left_v, right_u, right_v)
    gradient_templates = (
        (sp.diff(green_template, left_u), sp.diff(green_template, left_v)),
        (sp.diff(green_template, right_u), sp.diff(green_template, right_v)),
    )
    for left in range(BODY_COUNT):
        for right in range(left + 1, BODY_COUNT):
            ul, vl = state[f"u{left}"], state[f"v{left}"]
            ur, vr = state[f"u{right}"], state[f"v{right}"]
            substitution = {
                left_u: ul, left_v: vl, right_u: ur, right_v: vr,
            }
            for body, other, coordinate_v, gradients in (
                (left, right, vl, gradient_templates[0]),
                (right, left, vr, gradient_templates[1]),
            ):
                gradient_u = gradients[0].xreplace(substitution)
                gradient_v = gradients[1].xreplace(substitution)
                if geometry == "sphere":
                    inverse_u_metric = 1 / (radius**2 * sp.cos(coordinate_v) ** 2)
                    inverse_v_metric = 1 / radius**2
                else:
                    inverse_u_metric = inverse_v_metric = sp.Integer(1)
                acceleration_u[body] += (
                    gravity * masses[other] * inverse_u_metric * gradient_u
                )
                acceleration_v[body] += (
                    gravity * masses[other] * inverse_v_metric * gradient_v
                )

    rates: dict[str, sp.Expr] = {}
    for index in range(BODY_COUNT):
        rates[f"u{index}"] = state[f"du{index}"]
        rates[f"v{index}"] = state[f"dv{index}"]
        rates[f"du{index}"] = acceleration_u[index]
        rates[f"dv{index}"] = acceleration_v[index]
    return rates


def _integrated_state_equations(geometry: str):
    """Manifest the repository RK4 algorithm as causal symbolic pieces."""

    from src.common.dt_system.integrator.integrator import RK4Integrator

    dt = sp.Symbol("dt")
    initial = sp.Matrix([sp.Symbol(name) for name in STATE_NAMES])
    stages: list[tuple[sp.Equality, ...]] = []
    calls = 0

    def declare_derivative(stage_time, stage_state):
        nonlocal calls
        stage_mapping = dict(zip(STATE_NAMES, stage_state))
        rates = (
            _torus_dynamics(stage_time, stage_mapping)
            if geometry == "torus"
            else _non_torus_dynamics(geometry, stage_mapping)
        )
        derivative = tuple(
            sp.Symbol(f"rk_stage{calls}_{name}_rate") for name in STATE_NAMES
        )
        calls += 1
        stages.append(tuple(
            sp.Eq(sp.Symbol(f"{symbol.name}_next"), rates[name], evaluate=False)
            for name, symbol in zip(STATE_NAMES, derivative)
        ))
        return sp.Matrix(derivative)

    result = RK4Integrator().step(declare_derivative, sp.Integer(0), initial, dt)
    return tuple(stages), dict(zip(STATE_NAMES, result))


def _mechanical_energy(state: Mapping[str, sp.Expr], geometry: str):
    gravity = sp.Symbol("gravity")
    major_radius = sp.Symbol("major_radius")
    minor_radius = sp.Symbol("minor_radius")
    kernel = build_torus_green_kernel() if geometry == "torus" else None
    inverse_green = (
        sp.ImmutableDenseMatrix(kernel.inverse_stiffness.tolist())
        if kernel is not None else None
    )
    surface_radius = sp.Symbol("surface_radius")
    kinetic = sp.Integer(0)
    momentum = sp.Integer(0)
    for index in range(BODY_COUNT):
        mass = sp.Symbol(f"mass{index}")
        v = state[f"v{index}"]
        du = state[f"du{index}"]
        dv = state[f"dv{index}"]
        if geometry == "torus":
            tube = major_radius + minor_radius * sp.cos(v)
            speed_squared = tube**2 * du**2 + minor_radius**2 * dv**2
            momentum += mass * tube**2 * du
        elif geometry == "sphere":
            speed_squared = surface_radius**2 * (sp.cos(v) ** 2 * du**2 + dv**2)
            momentum += mass * surface_radius**2 * sp.cos(v) ** 2 * du
        else:
            speed_squared = du**2 + dv**2
            momentum += mass * du
        kinetic += mass * speed_squared / 2
    potential = sp.Integer(0)
    for left in range(BODY_COUNT):
        for right in range(left + 1, BODY_COUNT):
            if geometry == "torus":
                basis_l, _du_l, _dv_l = symbolic_green_basis(
                    state[f"u{left}"], state[f"v{left}"], kernel
                )
                basis_r, _du_r, _dv_r = symbolic_green_basis(
                    state[f"u{right}"], state[f"v{right}"], kernel
                )
                green = basis_l.dot(inverse_green * basis_r)
            elif geometry == "sphere":
                green = _sphere_green(
                    state[f"u{left}"], state[f"v{left}"],
                    state[f"u{right}"], state[f"v{right}"],
                )
            else:
                green = _cartesian_green(
                    state[f"u{left}"], state[f"v{left}"],
                    state[f"u{right}"], state[f"v{right}"],
                )
            potential -= gravity * sp.Symbol(f"mass{left}") * sp.Symbol(
                f"mass{right}"
            ) * green
    return kinetic + potential, momentum


@lru_cache(maxsize=3)
def compile_three_body_laws(geometry: str = "torus"):
    """Author RK4 stages, conservation channels, and spectral projection."""

    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    names = _law_names(geometry)
    stages, next_state = _integrated_state_equations(geometry)
    laws = [
        compile_sympy_equations(equations, name=names[index])
        for index, equations in enumerate(stages)
    ]

    direction_x, direction_y = sp.symbols("direction_x direction_y")
    center, falloff_power, brightness, exposure = sp.symbols(
        "center_contribution falloff_power brightness exposure"
    )
    major_radius, minor_radius, projection_scale = sp.symbols(
        "major_radius minor_radius projection_scale"
    )
    spectra = sp.Matrix(
        BODY_COUNT, 8,
        lambda source, band: sp.Symbol(f"spectrum{source}_b{band}"),
    )
    weights = []
    speeds = []
    for index in range(BODY_COUNT):
        if geometry == "torus":
            x, y = torus_face_projection(
                next_state[f"u{index}"], next_state[f"v{index}"],
                major_radius, minor_radius, projection_scale,
            )
        elif geometry == "sphere":
            x = projection_scale * sp.cos(next_state[f"v{index}"]) * sp.cos(next_state[f"u{index}"])
            y = projection_scale * sp.sin(next_state[f"v{index}"])
        else:
            x = projection_scale * next_state[f"u{index}"]
            y = projection_scale * next_state[f"v{index}"]
        weights.append(symbolic_radial_kernel(
            x, y, direction_x, direction_y, center, falloff_power,
        ))
        if geometry == "torus":
            tube = major_radius + minor_radius * sp.cos(next_state[f"v{index}"])
            speed_squared = (
                tube**2 * next_state[f"du{index}"]**2
                + minor_radius**2 * next_state[f"dv{index}"]**2
            )
        elif geometry == "sphere":
            surface_radius = sp.Symbol("surface_radius")
            speed_squared = surface_radius**2 * (
                sp.cos(next_state[f"v{index}"])**2 * next_state[f"du{index}"]**2
                + next_state[f"dv{index}"]**2
            )
        else:
            speed_squared = next_state[f"du{index}"]**2 + next_state[f"dv{index}"]**2
        speeds.append(sp.sqrt(speed_squared))
    rgb = symbolic_spectral_display(
        sp.Matrix(weights), spectra, exposure=exposure, output_level=brightness,
    )
    old_state = {name: sp.Symbol(name) for name in STATE_NAMES}
    energy0, momentum0 = _mechanical_energy(old_state, geometry)
    energy1, momentum1 = _mechanical_energy(next_state, geometry)
    energy_error = sp.Abs(energy1 - energy0) / sp.Max(sp.Abs(energy0), sp.Integer(1))
    momentum_error = sp.Abs(momentum1 - momentum0) / sp.Max(
        sp.Abs(momentum0), sp.Integer(1)
    )
    max_speed = sp.Max(*speeds, sp.Float(1.0e-12))
    equations: list[sp.Equality] = [
        sp.Eq(sp.Symbol(f"{name}_next"), next_state[name], evaluate=False)
        for name in STATE_NAMES
    ]
    equations.extend((
        sp.Eq(
            sp.Symbol("system_temperature_c_next"),
            sp.Symbol("system_temperature_c"),
            evaluate=False,
        ),
        sp.Eq(sp.Symbol("red_next"), rgb[0], evaluate=False),
        sp.Eq(sp.Symbol("green_next"), rgb[1], evaluate=False),
        sp.Eq(sp.Symbol("blue_next"), rgb[2], evaluate=False),
        sp.Eq(sp.Symbol("three_body_energy_relative_error"), energy_error, evaluate=False),
        sp.Eq(sp.Symbol("three_body_momentum_relative_error"), momentum_error, evaluate=False),
        sp.Eq(
            sp.Symbol("dt_limit"),
            sp.Min(sp.Float(0.02), sp.Symbol("geometry_length_scale") / max_speed),
            evaluate=False,
        ),
        sp.Eq(sp.Symbol("max_vel"), max_speed, evaluate=False),
    ))
    laws.append(compile_sympy_equations(
        tuple(equations), name=names[-1]
    ))
    return tuple(laws)


def compile_three_body_pieces(*, geometry: str = "torus", directory: Path | None = None):
    from src.compiler.native_package import piece_from_law

    names = _law_names(geometry)
    build = directory or (TURING_ROOT / "build" / f"openrgb_{geometry}_three_body" / "8")
    pieces = []
    for law, name in zip(compile_three_body_laws(geometry), names):
        pieces.append(piece_from_law(law, name, 8, directory=build / name))
    return tuple(pieces)


class _EagerEquationPiece:
    """CSE'd SymPy truth behind the ordinary dt-system piece API."""

    def __init__(self, compilation, entry: str, batch: int) -> None:
        metadata = compilation.function.metadata
        self.entry = entry
        self.batch = int(batch)
        self.argument_names = tuple(metadata["argument_names"])
        self.output_names = tuple(metadata["output_names"])
        symbols = tuple(sp.Symbol(name) for name in self.argument_names)
        self._stage = sp.lambdify(
            symbols,
            [equation.rhs for equation in compilation.equations],
            modules="math",
            cse=True,
        )

    def instantiate(self, columns, outputs=None) -> None:
        # The eager stage has no pointer table to bind.  PieceState still owns
        # and rolls back every column handed through this same piece contract.
        return None

    def __call__(self, *columns):
        lanes = [
            self._stage(*(
                float(np.asarray(column, dtype=np.float64).reshape(-1)[lane])
                for column in columns
            ))
            for lane in range(self.batch)
        ]
        return tuple(
            np.asarray(values, dtype=np.float64)
            for values in zip(*lanes)
        )


@lru_cache(maxsize=3)
def eager_three_body_pieces(geometry: str = "torus"):
    """Run the exact compiler materialization before lowering it to LLVM."""

    return tuple(
        _EagerEquationPiece(law, name, 8)
        for law, name in zip(compile_three_body_laws(geometry), _law_names(geometry))
    )


class ManagedThreeBodyField:
    """Persistent managed-dt three-body state on a selectable intrinsic geometry."""

    def __init__(
        self,
        domain: AddressDomain,
        *,
        frames_per_second: float = 60.0,
        center_contribution: float = 0.125,
        falloff_power: float = 2.0,
        brightness: int = 96,
        exposure: float = 3.0,
        gravity: float = 0.35,
        major_radius: float = 1.5,
        minor_radius: float = 0.5,
        projection_scale: float = 0.82,
        system_temperature_c: float = 0.0,
        thermal_coupling: float = 1.0,
        thermal_seed: int | None = None,
        error_limits: Mapping[str, float] | None = None,
        geometry: str = "torus",
        surface_radius: float = 1.0,
        softening: float = 0.08,
        pieces=None,
    ) -> None:
        if domain.count != 8:
            raise ValueError("The calibrated radial field requires exactly eight addresses")
        if frames_per_second <= 0.0 or not math.isfinite(frames_per_second):
            raise ValueError("frames_per_second must be finite and positive")
        if not 0.0 <= center_contribution <= 1.0:
            raise ValueError("center_contribution must be in the inclusive range 0..1")
        if not math.isfinite(falloff_power) or falloff_power <= 0.0:
            raise ValueError("falloff_power must be finite and positive")
        if not 0 <= brightness <= 255:
            raise ValueError("brightness must be in the inclusive range 0..255")
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure must be finite and positive")
        if not (major_radius > minor_radius > 0.0):
            raise ValueError("torus radii require major_radius > minor_radius > 0")
        kernel = build_torus_green_kernel()
        if not (
            math.isclose(major_radius, kernel.major_radius)
            and math.isclose(minor_radius, kernel.minor_radius)
        ):
            raise ValueError(
                "the current torus Green operator is built for "
                f"R={kernel.major_radius}, r={kernel.minor_radius}"
            )
        if not 0.0 < projection_scale < 1.0:
            raise ValueError("projection_scale must place the torus strictly inside the ring")
        if not math.isfinite(system_temperature_c) or system_temperature_c < 0.0:
            raise ValueError("system_temperature_c must be finite and nonnegative")
        if not math.isfinite(thermal_coupling) or thermal_coupling < 0.0:
            raise ValueError("thermal_coupling must be finite and nonnegative")
        if geometry not in PHYSICS_GEOMETRIES:
            raise ValueError(f"unknown physics geometry {geometry!r}")
        if not math.isfinite(surface_radius) or surface_radius <= 0.0:
            raise ValueError("surface_radius must be finite and positive")
        if not math.isfinite(softening) or softening <= 0.0:
            raise ValueError("softening must be finite and positive")

        from llvm_dt_system import instantiate_system, piece_leaf
        from src.common.dt_system.dt import SuperstepPlan
        from src.common.dt_system.dt_controller import STController, Targets
        from src.common.dt_system.dt_graph import ControllerNode, RoundNode
        from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields

        limits = dict(TORUS_ERROR_LIMITS)
        if error_limits is not None:
            unknown = set(error_limits) - limits.keys()
            if unknown:
                raise ValueError(f"undeclared torus error channels: {sorted(unknown)}")
            limits.update({name: float(value) for name, value in error_limits.items()})
        channel_names = (*DT_CHANNEL_NAMES, *limits)
        self.domain = domain
        self.geometry = geometry
        self.frame_seconds = 1.0 / float(frames_per_second)
        self.pieces = tuple(pieces or compile_three_body_pieces(geometry=geometry))
        count = domain.count
        initial = {
            "torus": (
                (1.2490457723982544, -0.45),
                (-2.677945044588987, 1.75),
                (-0.7853981633974483, -2.15),
            ),
            "sphere": ((1.15, -0.42), (-2.35, 0.68), (-0.55, -0.92)),
            "cartesian": ((0.6, -0.35), (-0.6, -0.35), (0.0, 0.7)),
        }[geometry]
        masses = (3.0, 4.0, 5.0)
        columns = {
            "direction_x": RADIAL_DIRECTIONS[:, 0].copy(),
            "direction_y": RADIAL_DIRECTIONS[:, 1].copy(),
            "gravity": np.full(count, gravity),
            "major_radius": np.full(count, major_radius),
            "minor_radius": np.full(count, minor_radius),
            "projection_scale": np.full(count, projection_scale),
            "surface_radius": np.full(count, surface_radius),
            "softening": np.full(count, softening),
            "geometry_length_scale": np.full(
                count,
                minor_radius if geometry == "torus" else surface_radius,
            ),
            "system_temperature_c": np.full(count, system_temperature_c),
            "center_contribution": np.full(count, center_contribution),
            "falloff_power": np.full(count, falloff_power),
            "brightness": np.full(count, float(brightness)),
            "exposure": np.full(count, exposure),
            "red": np.zeros(count), "green": np.zeros(count), "blue": np.zeros(count),
        }
        for index in range(BODY_COUNT):
            columns[f"u{index}"] = np.full(count, initial[index][0])
            columns[f"v{index}"] = np.full(count, initial[index][1])
            columns[f"du{index}"] = np.zeros(count)
            columns[f"dv{index}"] = np.zeros(count)
            columns[f"mass{index}"] = np.full(count, masses[index])
            for band in range(8):
                columns[f"spectrum{index}_b{band}"] = np.full(
                    count, CMY_SPECTRA[index, band]
                )
        # RK4Integrator's callback manifests derivative stages as ordinary
        # graph columns.  Register exactly those declared scratch outputs so
        # PieceState includes them in its normal save/restore transaction.
        for piece in self.pieces:
            for output in piece.output_names:
                if (
                    output.startswith("rk_stage")
                    and output.endswith("_next")
                    and output[:-5] not in columns
                ):
                    columns[output[:-5]] = np.zeros(count)
        targets = Targets(
            cfl=0.5, div_max=1.0e9, mass_max=1.0e-3,
            energy_exchange_fraction=0.2,
            **channel_fields(limits, names=channel_names, limits=True),
        )
        graph = RoundNode(
            plan=SuperstepPlan(
                round_max=self.frame_seconds,
                dt_init=self.frame_seconds,
                allow_increase_mid_round=True,
                rollback=True,
            ),
            controller=ControllerNode(
                ctrl=STController(dt_min=None),
                targets=targets,
                dx=minor_radius if geometry == "torus" else surface_radius,
            ),
            children=[piece_leaf(piece, label=piece.entry) for piece in self.pieces],
            schedule="sequential",
            label=f"openrgb-{geometry}-three-body",
        )
        self.state = instantiate_system(graph, columns, channel_names=channel_names)
        self.thermal_coupling = float(thermal_coupling)
        self._thermal_rng = np.random.default_rng(thermal_seed)
        self.last_thermal_velocity_scale = 1.0
        self.last_thermal_energy_delta = 0.0
        self.last_thermal_impulse_rms = 0.0

    def set_system_temperature(self, celsius: float) -> None:
        value = float(celsius)
        if not math.isfinite(value) or not 0.0 <= value < 150.0:
            raise ValueError(f"invalid system temperature in Celsius: {value}")
        self.state.system_temperature_c[:] = value

    def set_thermal_coupling(self, coupling: float) -> None:
        value = float(coupling)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid thermal coupling: {value}")
        self.thermal_coupling = value

    def _kinetic_energy(self) -> float:
        major = float(self.state.major_radius[0])
        minor = float(self.state.minor_radius[0])
        result = 0.0
        for index in range(BODY_COUNT):
            mass = float(getattr(self.state, f"mass{index}")[0])
            if mass <= 0.0:
                continue
            body_v = float(getattr(self.state, f"v{index}")[0])
            body_du = float(getattr(self.state, f"du{index}")[0])
            body_dv = float(getattr(self.state, f"dv{index}")[0])
            if self.geometry == "torus":
                tube = major + minor * math.cos(body_v)
                speed_squared = tube**2 * body_du**2 + minor**2 * body_dv**2
            elif self.geometry == "sphere":
                radius = float(self.state.surface_radius[0])
                speed_squared = radius**2 * (
                    math.cos(body_v) ** 2 * body_du**2 + body_dv**2
                )
            else:
                speed_squared = body_du**2 + body_dv**2
            result += mass * speed_squared / 2.0
        return result

    def apply_thermal_agitation(self) -> None:
        """Apply one exact Ornstein--Uhlenbeck heat-bath impulse in the tangent plane."""

        temperature = float(self.state.system_temperature_c[0])
        coupling = self.thermal_coupling
        if temperature <= 0.0 or coupling <= 0.0:
            self.last_thermal_velocity_scale = 1.0
            self.last_thermal_energy_delta = 0.0
            self.last_thermal_impulse_rms = 0.0
            return
        energy_before = self._kinetic_energy()
        decay = math.exp(-coupling * self.frame_seconds)
        variance_fraction = max(0.0, 1.0 - decay * decay)
        dimensionless_temperature = temperature / 100.0
        major = float(self.state.major_radius[0])
        minor = float(self.state.minor_radius[0])
        impulse_squared = 0.0
        impulse_count = 0
        for index in range(BODY_COUNT):
            mass = float(getattr(self.state, f"mass{index}")[0])
            if mass <= 0.0:
                continue
            v_column = getattr(self.state, f"v{index}")
            du_column = getattr(self.state, f"du{index}")
            dv_column = getattr(self.state, f"dv{index}")
            if self.geometry == "torus":
                scale_u = major + minor * math.cos(float(v_column[0]))
                scale_v = minor
            elif self.geometry == "sphere":
                radius = float(self.state.surface_radius[0])
                scale_u = radius * max(1.0e-6, abs(math.cos(float(v_column[0]))))
                scale_v = radius
            else:
                scale_u = scale_v = 1.0
            tangent_u = scale_u * float(du_column[0])
            tangent_v = scale_v * float(dv_column[0])
            sigma = math.sqrt(dimensionless_temperature * variance_fraction / mass)
            next_u = decay * tangent_u + sigma * float(self._thermal_rng.standard_normal())
            next_v = decay * tangent_v + sigma * float(self._thermal_rng.standard_normal())
            impulse_squared += (next_u - tangent_u) ** 2 + (next_v - tangent_v) ** 2
            impulse_count += 2
            du_column[:] = next_u / scale_u
            dv_column[:] = next_v / scale_v
        self.last_thermal_velocity_scale = decay
        self.last_thermal_energy_delta = self._kinetic_energy() - energy_before
        self.last_thermal_impulse_rms = math.sqrt(
            impulse_squared / max(1, impulse_count)
        )

    def advance_frame(self) -> list[tuple[int, int, int]]:
        self.apply_thermal_agitation()
        advance_managed_state(self.state, self.frame_seconds)
        self._normalize_coordinate_chart()
        return self.domain.frame_from_channels(self.state.red, self.state.green, self.state.blue)

    def _normalize_coordinate_chart(self) -> None:
        """Keep periodic coordinates bounded without changing the represented point."""

        if self.geometry == "cartesian":
            return
        for index in range(BODY_COUNT):
            u = getattr(self.state, f"u{index}")
            v = getattr(self.state, f"v{index}")
            u[:] = np.mod(u + math.pi, 2.0 * math.pi) - math.pi
            if self.geometry == "torus":
                v[:] = np.mod(v + math.pi, 2.0 * math.pi) - math.pi
                continue
            dv = getattr(self.state, f"dv{index}")
            latitude = np.mod(v + math.pi, 2.0 * math.pi) - math.pi
            north = latitude > 0.5 * math.pi
            south = latitude < -0.5 * math.pi
            crossed = north | south
            latitude[north] = math.pi - latitude[north]
            latitude[south] = -math.pi - latitude[south]
            u[crossed] = np.mod(u[crossed] + math.pi + math.pi, 2.0 * math.pi) - math.pi
            dv[crossed] *= -1.0
            v[:] = latitude


def _grb_wire_frame(frame: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Compensate the measured Header 3 physical red/green swap."""

    return [(green, red, blue) for red, green, blue in frame]


def _thermal_velocity_scale(previous_celsius: float, current_celsius: float) -> float:
    """Velocity ratio whose square gives the absolute-temperature ratio."""

    previous_kelvin = float(previous_celsius) + 273.15
    current_kelvin = float(current_celsius) + 273.15
    if not (
        math.isfinite(previous_kelvin)
        and math.isfinite(current_kelvin)
        and previous_kelvin > 0.0
        and current_kelvin > 0.0
    ):
        raise ValueError("thermal coupling requires positive finite Kelvin values")
    return math.sqrt(current_kelvin / previous_kelvin)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)]


def _body_coordinates(field: ManagedThreeBodyField) -> list[tuple[float, float, float, float]]:
    scale = float(field.state.projection_scale[0])
    if field.geometry == "torus":
        major = float(field.state.major_radius[0])
        minor = float(field.state.minor_radius[0])
    result = []
    for index in range(BODY_COUNT):
        u = float(getattr(field.state, f"u{index}")[0])
        v = float(getattr(field.state, f"v{index}")[0])
        if field.geometry == "torus":
            radius = scale * (major + minor * math.cos(v)) / (major + minor)
            x, y = radius * math.cos(u), radius * math.sin(u)
        elif field.geometry == "sphere":
            x = scale * math.cos(v) * math.cos(u)
            y = scale * math.sin(v)
        else:
            x, y = scale * u, scale * v
        result.append((u, v, x, y))
    return result


def _logical_domain_frame(
    domain: AddressDomain,
    full_frame: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Read the selected domain's colors from their full-controller slots."""

    return [full_frame[address.frame_index] for address in domain.addresses]


def _draw_live_window(
    pygame,
    screen,
    font,
    value_font,
    field: ManagedThreeBodyField,
    frame: Sequence[tuple[int, int, int]],
    trails,
    *,
    frame_number: int,
    sent: bool,
    attempts: int,
    rejected: int,
    energy_error: float,
    momentum_error: float,
    latest_rejection: str | None,
) -> None:
    width, height = screen.get_size()
    background = (12, 14, 18)
    screen.fill(background)
    center = (min(width // 2, 330), height // 2)
    ring = min(245, height // 2 - 35, center[0] - 25)
    major = float(field.state.major_radius[0])
    minor = float(field.state.minor_radius[0])
    scale = float(field.state.projection_scale[0])
    outer = max(1, round(ring * scale))
    inner = max(1, round(ring * scale * (major - minor) / (major + minor)))
    pygame.draw.circle(screen, (38, 42, 50), center, outer)
    pygame.draw.circle(screen, background, center, inner)
    pygame.draw.circle(screen, (94, 100, 112), center, outer, 2)
    pygame.draw.circle(screen, (94, 100, 112), center, inner, 2)

    for light_index, (direction, color) in enumerate(zip(RADIAL_DIRECTIONS, frame)):
        point = (
            round(center[0] + ring * float(direction[0])),
            round(center[1] - ring * float(direction[1])),
        )
        pygame.draw.circle(screen, tuple(int(x) for x in color), point, 14)
        pygame.draw.circle(screen, (150, 155, 165), point, 15, 1)
        value_point = (
            round(center[0] + ring * 1.13 * float(direction[0])),
            round(center[1] - ring * 1.13 * float(direction[1])),
        )
        value = value_font.render(
            f"{light_index} {int(color[0])}/{int(color[1])}/{int(color[2])}",
            True,
            (205, 210, 220),
        )
        screen.blit(value, value.get_rect(center=value_point))

    body_colors = ((0, 255, 255), (255, 0, 255), (255, 235, 0))
    coordinates = _body_coordinates(field)
    for index, ((u, v, x, y), color) in enumerate(zip(coordinates, body_colors)):
        point = (round(center[0] + ring * x), round(center[1] - ring * y))
        trails[index].append(point)
        if len(trails[index]) > 1:
            pygame.draw.lines(screen, tuple(channel // 3 for channel in color), False, trails[index], 2)
        pygame.draw.circle(screen, color, point, 9)
        label = font.render(str(index), True, (230, 233, 240))
        screen.blit(label, (point[0] + 11, point[1] - 9))

    lines = [
        f"frame {frame_number}   sent {sent}",
        f"system / processor temperature {float(field.state.system_temperature_c[0]):.1f} C",
        f"thermal tangent retention {field.last_thermal_velocity_scale:.9f}",
        f"thermal impulse rms {field.last_thermal_impulse_rms:.6g}",
        f"thermal kinetic delta {field.last_thermal_energy_delta:+.6g}",
        f"attempts {attempts}   rejected {rejected}",
        f"dt_next {float(field.state.dt_next):.12g}",
        f"energy error max {energy_error:.6g}",
        f"momentum error max {momentum_error:.6g}",
        "",
    ]
    lines.extend(
        f"{index}  u={u:+.9f}  v={v:+.9f}"
        for index, (u, v, _x, _y) in enumerate(coordinates)
    )
    if latest_rejection:
        lines.extend(("", "latest rejection:", latest_rejection))
    lines.extend(("", "Space: pause", "Esc / close: stop"))
    x_text = max(center[0] + ring + 32, 680)
    for row, line in enumerate(lines):
        screen.blit(font.render(line, True, (220, 224, 232)), (x_text, 35 + row * 25))
    pygame.display.flip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--position-count", type=int, default=8)
    parser.add_argument("--mode", default="Direct")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--brightness", type=int, default=96)
    parser.add_argument("--center-contribution", type=float, default=0.125)
    parser.add_argument("--falloff-power", type=float, default=2.0)
    parser.add_argument("--exposure", type=float, default=3.0)
    parser.add_argument("--backend", choices=("eager", "llvm"), default="eager")
    parser.add_argument(
        "--cpu-temperature",
        action="store_true",
        help="make the managed system-temperature column follow the processor sensor",
    )
    parser.add_argument(
        "--temperature-assembly",
        default=str(DEFAULT_ASSEMBLY),
        help="path to LibreHardwareMonitorLib.dll",
    )
    parser.add_argument(
        "--temperature-sensor",
        default=DEFAULT_SENSOR_IDENTIFIER,
        help="LibreHardwareMonitor processor sensor identifier",
    )
    parser.add_argument("--window", action="store_true")
    parser.add_argument(
        "--print-every", type=int, default=0,
        help="print live coordinates every N completed frames; 0 disables",
    )
    parser.add_argument(
        "--trace-rejections", action="store_true",
        help="print every rejected dt attempt with state and failing metrics",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    temperature_reader = None
    try:
        if args.frames < 0:
            raise ValueError("frames must be nonnegative (0 runs until the window closes)")
        if args.frames == 0 and not args.window:
            raise ValueError("frames=0 requires --window so the run has a stop control")
        if args.print_every < 0:
            raise ValueError("print-every must be nonnegative")
        initial_temperature = 0.0
        if args.cpu_temperature:
            temperature_reader = CpuTemperatureReader(
                args.temperature_assembly,
                args.temperature_sensor,
            )
            initial_temperature = temperature_reader.read_celsius()
        controller = connect(device_name=args.device_name)
        try:
            zone_selector: int | str = int(args.zone)
        except ValueError:
            zone_selector = args.zone
        domain = address_domain_for(controller, zone_selector, position_count=args.position_count)
        pieces = (
            eager_three_body_pieces()
            if args.backend == "eager"
            else compile_three_body_pieces()
        )
        field = ManagedThreeBodyField(
            domain,
            frames_per_second=args.fps,
            center_contribution=args.center_contribution,
            falloff_power=args.falloff_power,
            brightness=args.brightness,
            exposure=args.exposure,
            system_temperature_c=initial_temperature,
            pieces=pieces,
        )
        attempt_count = 0
        accepted_count = 0
        energy_error_max = 0.0
        momentum_error_max = 0.0
        dt_attempt_min = math.inf
        dt_attempt_max = 0.0
        latest_rejection: str | None = None
        rejection_streak = 0
        native_advance = field.state.program["advance_pieces"]

        def observe_advance(actual_state, step_dt):
            nonlocal attempt_count, accepted_count
            nonlocal energy_error_max, momentum_error_max
            nonlocal dt_attempt_min, dt_attempt_max
            nonlocal latest_rejection, rejection_streak
            ok, metrics = native_advance(actual_state, step_dt)
            attempt_count += 1
            energy_error = float(metrics.error_channels[-2])
            momentum_error = float(metrics.error_channels[-1])
            energy_limit = float(field.state.targets.error_limits[-2])
            momentum_limit = float(field.state.targets.error_limits[-1])
            reasons = []
            if not bool(ok):
                reasons.append("advance")
            if bool(metrics.hard_failure):
                reasons.append("hard_failure")
            if not math.isfinite(energy_error):
                reasons.append("energy_nonfinite")
            elif energy_error > energy_limit:
                reasons.append("energy")
            if not math.isfinite(momentum_error):
                reasons.append("momentum_nonfinite")
            elif momentum_error > momentum_limit:
                reasons.append("momentum")
            candidate = []
            for index in range(BODY_COUNT):
                values = tuple(
                    float(getattr(actual_state, f"{name}{index}")[0])
                    for name in ("u", "v", "du", "dv")
                )
                candidate.append(values)
            if not all(math.isfinite(value) for body in candidate for value in body):
                reasons.append("state_nonfinite")
            rejected = bool(reasons)
            accepted_count += int(not rejected)
            energy_error_max = max(energy_error_max, energy_error)
            momentum_error_max = max(momentum_error_max, momentum_error)
            dt_attempt_min = min(dt_attempt_min, float(step_dt))
            dt_attempt_max = max(dt_attempt_max, float(step_dt))
            if rejected:
                rejection_streak += 1
                major = float(actual_state.major_radius[0])
                minor = float(actual_state.minor_radius[0])
                speeds = [
                    math.sqrt(
                        (major + minor * math.cos(v)) ** 2 * du**2
                        + minor**2 * dv**2
                    )
                    for _u, v, du, dv in candidate
                ]
                green_pairs = tuple(
                    evaluate_green(
                        candidate[left][0], candidate[left][1],
                        candidate[right][0], candidate[right][1],
                    )
                    for left, right in ((0, 1), (0, 2), (1, 2))
                )
                latest_rejection = (
                    f"dt={float(step_dt):.6g} {'+'.join(reasons)} "
                    f"E={energy_error:.3g}/{energy_limit:.3g} "
                    f"P={momentum_error:.3g}/{momentum_limit:.3g}"
                )
                if args.trace_rejections:
                    print(
                        f"dt-reject frame={frame_number + 1} attempt={attempt_count} "
                        f"streak={rejection_streak} {latest_rejection} "
                        f"dt_limit={float(metrics.dt_limit):.9g} "
                        f"max_vel={float(metrics.max_vel):.9g}",
                        flush=True,
                    )
                    for index, ((u, v, du, dv), speed) in enumerate(zip(candidate, speeds)):
                        print(
                            f"  body{index} u={u:+.12g} v={v:+.12g} "
                            f"du={du:+.12g} dv={dv:+.12g} speed={speed:.12g}",
                            flush=True,
                        )
                    print(
                        f"  green_pairs g01={green_pairs[0]:.12g} "
                        f"g02={green_pairs[1]:.12g} g12={green_pairs[2]:.12g}",
                        flush=True,
                    )
            elif rejection_streak:
                if args.trace_rejections:
                    print(
                        f"dt-accepted-after-reject frame={frame_number + 1} "
                        f"attempt={attempt_count} rejected_before={rejection_streak} "
                        f"dt={float(step_dt):.12g} E={energy_error:.6g} "
                        f"P={momentum_error:.6g}",
                        flush=True,
                    )
                rejection_streak = 0
            return ok, metrics

        field.state.program["advance_pieces"] = observe_advance
        if args.apply:
            controller.select_mode(args.mode)
        pygame = screen = font = value_font = None
        trails = [deque(maxlen=600) for _ in range(BODY_COUNT)]
        paused = False
        if args.window:
            import pygame as pygame_module

            pygame = pygame_module
            pygame.init()
            screen = pygame.display.set_mode((1024, 650), pygame.RESIZABLE)
            pygame.display.set_caption("Intrinsic torus three-body LED field")
            font = pygame.font.Font(None, 25)
            value_font = pygame.font.Font(None, 18)
        work_times: list[float] = []
        send_times: list[float] = []
        misses = 0
        started = time.perf_counter()
        cpu_started = time.process_time()
        frame_number = 0
        running = True
        while running and (args.frames == 0 or frame_number < args.frames):
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = not paused
                if not running:
                    break
                if paused:
                    pygame.time.wait(10)
                    continue
            work_started = time.perf_counter()
            if temperature_reader is not None:
                field.set_system_temperature(temperature_reader.read_celsius())
            frame = field.advance_frame()
            if args.apply:
                send_started = time.perf_counter()
                controller.send_frame(_grb_wire_frame(frame))
                send_times.append(time.perf_counter() - send_started)
            work_times.append(time.perf_counter() - work_started)
            frame_number += 1
            if args.print_every and frame_number % args.print_every == 0:
                coordinates = _body_coordinates(field)
                print(
                    f"frame={frame_number} "
                    + " ".join(
                        f"body{index}.u={u:+.9f} body{index}.v={v:+.9f}"
                        for index, (u, v, _x, _y) in enumerate(coordinates)
                    )
                    + f" dt_next={float(field.state.dt_next):.12g} "
                    + f"attempts={attempt_count} rejected={attempt_count-accepted_count}",
                    flush=True,
                )
            if pygame is not None:
                _draw_live_window(
                    pygame, screen, font, value_font, field,
                    _logical_domain_frame(field.domain, frame), trails,
                    frame_number=frame_number, sent=bool(args.apply),
                    attempts=attempt_count, rejected=attempt_count-accepted_count,
                    energy_error=energy_error_max,
                    momentum_error=momentum_error_max,
                    latest_rejection=latest_rejection,
                )
            deadline = started + frame_number * field.frame_seconds
            now = time.perf_counter()
            misses += now > deadline
            time.sleep(max(0.0, deadline - now))
        if pygame is not None:
            pygame.quit()
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        print(
            "benchmark "
            f"frames={frame_number} target_hz={args.fps:.3f} achieved_hz={frame_number / elapsed:.3f} "
            f"deadline_misses={misses} work_ms_p50={statistics.median(work_times)*1000:.3f} "
            f"work_ms_p95={_percentile(work_times,0.95)*1000:.3f} "
            f"work_ms_max={max(work_times)*1000:.3f} "
            f"send_ms_p95={(_percentile(send_times,0.95)*1000 if send_times else 0.0):.3f} "
            f"process_cpu_percent={100.0*cpu_elapsed/elapsed:.2f} "
            f"dt_attempts={attempt_count} dt_accepted={accepted_count} "
            f"dt_rejected={attempt_count-accepted_count} "
            f"dt_attempt_min={dt_attempt_min:.9g} dt_attempt_max={dt_attempt_max:.9g} "
            f"energy_error_max={energy_error_max:.9g} "
            f"momentum_error_max={momentum_error_max:.9g}"
        )
        if not args.apply:
            print("Dry run: no hardware update sent")
        return 0
    except (OpenRGBShellError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if temperature_reader is not None:
            temperature_reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
