"""Build Laplace matrices with optional neural metric tensors."""
# --- END HEADER ---

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from scipy.sparse import coo_matrix
from typing import Callable, Tuple


class GridDomain:
    """Simple 3D grid domain with transformation support."""

    def __init__(self, U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, transform: 'Transform') -> None:
        self.U = U
        self.V = V
        self.W = W
        self.transform = transform
        self.vertices = self.transform.transform(U, V, W)


class Transform:
    """Base coordinate transform."""

    def transform(self, U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RectangularTransform(Transform):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def transform(self, U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.stack([U, V, W], dim=-1).to(self.device)


class PeriodicLinspace:
    """Generate periodic density modulations for grid creation."""

    def __init__(self, min_density: float = 0.5, max_density: float = 1.5, num_oscillations: int = 1) -> None:
        self.min_density = min_density
        self.max_density = max_density
        self.num_oscillations = num_oscillations

    def _oscillate(self, func: Callable[[torch.Tensor], torch.Tensor], normalized_i: torch.Tensor) -> torch.Tensor:
        phase = 2 * torch.pi * self.num_oscillations * normalized_i - torch.pi / 2
        return self.min_density + (self.max_density - self.min_density) * 0.5 * (1 + func(phase))

    def sin(self, normalized_i: torch.Tensor) -> torch.Tensor:
        return self._oscillate(torch.sin, normalized_i)

    def cos(self, normalized_i: torch.Tensor) -> torch.Tensor:
        return self._oscillate(torch.cos, normalized_i)

    def tan(self, normalized_i: torch.Tensor) -> torch.Tensor:
        density = self._oscillate(torch.tan, normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def get_density(self, normalized_i: torch.Tensor, oscillation_type: str) -> torch.Tensor:
        if not hasattr(self, oscillation_type):
            raise ValueError(f"Unknown oscillation_type: {oscillation_type}")
        return getattr(self, oscillation_type)(normalized_i)


# ########## STUB: neural_metric_tensor ##########
# PURPOSE: Provide learned metric tensors per grid point.
# EXPECTED BEHAVIOR: When implemented, returns a 3x3 tensor field defining the
#   metric at each grid vertex.
# INPUTS: coordinates (N,3) tensor
# OUTPUTS: metric tensor of shape (N,3,3)
# KEY ASSUMPTIONS/DEPENDENCIES: depends on a trained neural network model
# TODO:
#   - Load pretrained weights from ``neural_metric.pt`` if available.
#   - Implement full training pipeline for the network.
# NOTES: Fallbacks to identity metrics when no model is present.
# ###########################################################################


class _NeuralMetricNet(nn.Module):
    """Minimal feedforward network predicting symmetric 3x3 metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:  # (N,3)
        out = self.layers(pts)
        diag = F.softplus(out[:, :3]) + 1.0
        off = out[:, 3:]
        g = torch.zeros(pts.shape[0], 3, 3, device=pts.device)
        g[:, 0, 0] = diag[:, 0]
        g[:, 1, 1] = diag[:, 1]
        g[:, 2, 2] = diag[:, 2]
        g[:, 0, 1] = g[:, 1, 0] = off[:, 0]
        g[:, 0, 2] = g[:, 2, 0] = off[:, 1]
        g[:, 1, 2] = g[:, 2, 1] = off[:, 2]
        return g


_METRIC_MODEL: _NeuralMetricNet | None = None


def neural_metric_tensor(coords: torch.Tensor) -> torch.Tensor:
    """Return a metric tensor for each coordinate."""
    global _METRIC_MODEL
    if _METRIC_MODEL is None:
        model_path = Path(__file__).with_name("neural_metric.pt")
        if model_path.exists():
            _METRIC_MODEL = _NeuralMetricNet()
            state = torch.load(model_path, map_location="cpu")
            _METRIC_MODEL.load_state_dict(state)
            _METRIC_MODEL.eval()

    if _METRIC_MODEL is None:
        size = coords.shape[0]
        eye = torch.eye(3, device=coords.device).expand(size, 3, 3)
        return eye

    with torch.no_grad():
        return _METRIC_MODEL(coords)


class BuildLaplace3D:
    """Construct sparse Laplace matrices for a 3D grid with boundary handling."""

    def __init__(self, grid_domain: GridDomain,
                 metric_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 boundary_conditions: Tuple[str, str, str, str, str, str] | None = None) -> None:
        self.grid_domain = grid_domain
        self.metric_func = metric_func or neural_metric_tensor
        self.boundary_conditions = boundary_conditions or (
            'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet'
        )

    def _idx(self, i: int, j: int, k: int, ny: int, nz: int) -> int:
        return i * ny * nz + j * nz + k

    def _bc_action(self, bc: str, i: int, lim: int) -> tuple[bool, int | None]:
        if 0 <= i < lim:
            return True, i
        if bc == 'periodic':
            return True, i % lim
        if bc == 'neumann':
            return False, None
        return False, None

    def build_general_laplace(self) -> Tuple[coo_matrix, torch.Tensor]:
        U, V, W = self.grid_domain.U, self.grid_domain.V, self.grid_domain.W
        nx, ny, nz = U.shape[0], V.shape[1], W.shape[2]
        coords = self.grid_domain.vertices.reshape(-1, 3)
        metric = self.metric_func(coords)

        data = []
        row = []
        col = []

        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    idx = self._idx(x, y, z, ny, nz)
                    diag = 0
                    # neighbours in +/- x
                    # x neighbors
                    for dx, bc in [(-1, self.boundary_conditions[0]), (1, self.boundary_conditions[1])]:
                        nx_i = x + dx
                        valid, mapped = self._bc_action(bc, nx_i, nx)
                        if valid:
                            row.append(idx)
                            col.append(self._idx(mapped, y, z, ny, nz))
                            data.append(-1.0)
                            diag += 1
                        elif bc == 'neumann':
                            diag += 1
                    # y neighbors
                    for dy, bc in [(-1, self.boundary_conditions[2]), (1, self.boundary_conditions[3])]:
                        ny_i = y + dy
                        valid, mapped = self._bc_action(bc, ny_i, ny)
                        if valid:
                            row.append(idx)
                            col.append(self._idx(x, mapped, z, ny, nz))
                            data.append(-1.0)
                            diag += 1
                        elif bc == 'neumann':
                            diag += 1
                    # z neighbors
                    for dz, bc in [(-1, self.boundary_conditions[4]), (1, self.boundary_conditions[5])]:
                        nz_i = z + dz
                        valid, mapped = self._bc_action(bc, nz_i, nz)
                        if valid:
                            row.append(idx)
                            col.append(self._idx(x, y, mapped, ny, nz))
                            data.append(-1.0)
                            diag += 1
                        elif bc == 'neumann':
                            diag += 1
                    row.append(idx)
                    col.append(idx)
                    data.append(float(diag))
        n = nx * ny * nz
        laplace = coo_matrix((data, (row, col)), shape=(n, n))
        return laplace, metric
