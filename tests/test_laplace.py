"""Tests for the Laplace builder."""
# --- END HEADER ---

import torch
from laplace.laplace.builder import GridDomain, RectangularTransform, BuildLaplace3D


def create_simple_domain(nx: int, ny: int, nz: int) -> GridDomain:
    u = torch.arange(nx).view(nx, 1, 1).expand(nx, ny, nz)
    v = torch.arange(ny).view(1, ny, 1).expand(nx, ny, nz)
    w = torch.arange(nz).view(1, 1, nz).expand(nx, ny, nz)
    return GridDomain(u, v, w, RectangularTransform())


def test_laplace_dirichlet() -> None:
    domain = create_simple_domain(2, 2, 2)
    builder = BuildLaplace3D(domain)
    laplace, _ = builder.build_general_laplace()
    assert laplace.shape == (8, 8)
    assert laplace.nnz == 32


def test_laplace_periodic() -> None:
    domain = create_simple_domain(2, 2, 2)
    builder = BuildLaplace3D(
        domain,
        boundary_conditions=(
            'periodic', 'periodic', 'periodic', 'periodic', 'periodic', 'periodic'
        )
    )
    laplace, _ = builder.build_general_laplace()
    assert laplace.shape == (8, 8)
    assert laplace.nnz == 56
