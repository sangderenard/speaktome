"""Intrinsic zero-mean Green operator for a standard torus.

The geometry is exact SymPy ``ParametricDomain`` geometry.  A finite real
Fourier space excludes the constant mode, so the compact-surface Poisson
compatibility condition is built in.  AbstractTensor assembles and solves the
Laplace--Beltrami weak operator once; the resulting constants return to SymPy
as a smooth, exactly periodic particle interaction law.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
TURING_ROOT = HERE.parents[2] / "turing"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))


@lru_cache(maxsize=1)
def symbolic_torus_domain():
    from src.compiler.bitops import ParametricDomain

    u, v = sp.symbols("u v", real=True)
    major, minor = sp.symbols("major_radius minor_radius", positive=True)
    tube = major + minor * sp.cos(v)
    domain = ParametricDomain(
        (u, v),
        ((0, 2 * sp.pi), (0, 2 * sp.pi)),
        (
            tube * sp.cos(u),
            tube * sp.sin(u),
            minor * sp.sin(v),
        ),
        periodic=(u, v),
        orientation=1,
    )
    return domain, (u, v, major, minor)


def symbolic_torus_laplace_beltrami(field):
    """Apply the repository's exact parametric Laplace--Beltrami operator."""

    domain, _symbols = symbolic_torus_domain()
    return sp.simplify(domain.laplacian(field))


def _canonical_wavevectors(order: int) -> tuple[tuple[int, int], ...]:
    if order < 1:
        raise ValueError("Fourier order must be positive")
    # One representative from each {k, -k}; cos and sin supply both real modes.
    return tuple(
        (m, n)
        for m in range(order + 1)
        for n in range(-order, order + 1)
        if m > 0 or n > 0
    )


@dataclass(frozen=True)
class TorusGreenKernel:
    wavevectors: tuple[tuple[int, int], ...]
    inverse_stiffness: np.ndarray
    stiffness: np.ndarray
    major_radius: float
    minor_radius: float
    quadrature: int

    @property
    def basis_count(self) -> int:
        return 2 * len(self.wavevectors)


@lru_cache(maxsize=8)
def build_torus_green_kernel(
    major_radius: float = 1.5,
    minor_radius: float = 0.5,
    order: int = 2,
    quadrature: int = 48,
) -> TorusGreenKernel:
    """Galerkin inverse of ``-Delta_g`` in a zero-mean Fourier space.

    Assembly and the dense solve use AbstractTensor.  Uniform periodic
    quadrature is spectrally exact in ``u`` for this basis and convergent in
    ``v`` for the torus metric coefficients.  No coordinate wrapping occurs:
    every basis function is periodic by construction.
    """

    if not major_radius > minor_radius > 0.0:
        raise ValueError("torus radii require major_radius > minor_radius > 0")
    if quadrature < 2 * order + 3:
        raise ValueError("quadrature is too small for the requested Fourier order")

    from src.common.tensors import AbstractTensor

    wavevectors = _canonical_wavevectors(int(order))
    grid = np.arange(quadrature, dtype=np.float64) * (2.0 * math.pi / quadrature)
    u_grid, v_grid = np.meshgrid(grid, grid, indexing="ij")
    u = AbstractTensor.tensor(u_grid.reshape(-1).tolist(), dtype="float64")
    v = AbstractTensor.tensor(v_grid.reshape(-1).tolist(), dtype="float64")
    m = AbstractTensor.tensor([value[0] for value in wavevectors], dtype="float64")
    n = AbstractTensor.tensor([value[1] for value in wavevectors], dtype="float64")
    phase = u.unsqueeze(1) * m.unsqueeze(0) + v.unsqueeze(1) * n.unsqueeze(0)
    sine, cosine = phase.sin(), phase.cos()
    # Basis ordering: all cos(k.q), followed by all sin(k.q).
    du = AbstractTensor.cat((-sine * m.unsqueeze(0), cosine * m.unsqueeze(0)), dim=1)
    dv = AbstractTensor.cat((-sine * n.unsqueeze(0), cosine * n.unsqueeze(0)), dim=1)
    tube = major_radius + minor_radius * v.cos()
    area_weight = (2.0 * math.pi / quadrature) ** 2
    weight_u = area_weight * minor_radius / tube
    weight_v = area_weight * tube / minor_radius
    stiffness = du.T() @ (du * weight_u.unsqueeze(1))
    stiffness = stiffness + dv.T() @ (dv * weight_v.unsqueeze(1))
    identity = AbstractTensor.eye(stiffness.shape[0], dtype="float64")
    inverse = AbstractTensor.linalg.solve(stiffness, identity)
    stiffness_np = np.asarray(stiffness.tolist(), dtype=np.float64)
    inverse_np = np.asarray(inverse.tolist(), dtype=np.float64)
    # Quadrature roundoff must not turn analytic block zeros into thousands of
    # meaningless SymPy terms.
    cutoff = 256.0 * np.finfo(np.float64).eps * np.max(np.abs(inverse_np))
    inverse_np[np.abs(inverse_np) < cutoff] = 0.0
    return TorusGreenKernel(
        wavevectors=wavevectors,
        inverse_stiffness=inverse_np,
        stiffness=stiffness_np,
        major_radius=float(major_radius),
        minor_radius=float(minor_radius),
        quadrature=int(quadrature),
    )


def symbolic_green_basis(u, v, kernel: TorusGreenKernel):
    phases = [m * u + n * v for m, n in kernel.wavevectors]
    basis = sp.Matrix([sp.cos(phase) for phase in phases] + [sp.sin(phase) for phase in phases])
    du = sp.Matrix(
        [-m * sp.sin(phase) for phase, (m, _n) in zip(phases, kernel.wavevectors)]
        + [m * sp.cos(phase) for phase, (m, _n) in zip(phases, kernel.wavevectors)]
    )
    dv = sp.Matrix(
        [-n * sp.sin(phase) for phase, (_m, n) in zip(phases, kernel.wavevectors)]
        + [n * sp.cos(phase) for phase, (_m, n) in zip(phases, kernel.wavevectors)]
    )
    return basis, du, dv


def evaluate_green(u: float, v: float, source_u: float, source_v: float,
                   kernel: TorusGreenKernel | None = None) -> float:
    """Evaluate the periodic Galerkin Green kernel without coordinate wraps."""

    kernel = kernel or build_torus_green_kernel()
    modes = np.asarray(kernel.wavevectors, dtype=np.float64)

    def basis(q_u, q_v):
        phase = modes[:, 0] * q_u + modes[:, 1] * q_v
        return np.concatenate((np.cos(phase), np.sin(phase)))

    return float(
        basis(u, v) @ kernel.inverse_stiffness @ basis(source_u, source_v)
    )


def evaluate_green_gradient(
    u: float,
    v: float,
    source_u: float,
    source_v: float,
    kernel: TorusGreenKernel | None = None,
) -> tuple[float, float]:
    """Evaluate the target-coordinate gradient of the periodic Green kernel."""

    kernel = kernel or build_torus_green_kernel()
    modes = np.asarray(kernel.wavevectors, dtype=np.float64)
    target_phase = modes[:, 0] * u + modes[:, 1] * v
    source_phase = modes[:, 0] * source_u + modes[:, 1] * source_v
    source_basis = np.concatenate((np.cos(source_phase), np.sin(source_phase)))
    response = kernel.inverse_stiffness @ source_basis
    du = np.concatenate(
        (-modes[:, 0] * np.sin(target_phase), modes[:, 0] * np.cos(target_phase))
    )
    dv = np.concatenate(
        (-modes[:, 1] * np.sin(target_phase), modes[:, 1] * np.cos(target_phase))
    )
    return float(du @ response), float(dv @ response)


__all__ = [
    "TorusGreenKernel",
    "build_torus_green_kernel",
    "evaluate_green",
    "evaluate_green_gradient",
    "symbolic_green_basis",
    "symbolic_torus_domain",
    "symbolic_torus_laplace_beltrami",
]
