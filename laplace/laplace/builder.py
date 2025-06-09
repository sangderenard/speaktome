"""Build Laplace matrices with optional neural metric tensors."""
# --- END HEADER ---

from __future__ import annotations

import torch
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


# ########## STUB: neural_metric_tensor ##########
# PURPOSE: Provide learned metric tensors per grid point.
# EXPECTED BEHAVIOR: When implemented, returns a 3x3 tensor field defining the
#   metric at each grid vertex.
# INPUTS: coordinates (N,3) tensor
# OUTPUTS: metric tensor of shape (N,3,3)
# KEY ASSUMPTIONS/DEPENDENCIES: depends on a trained neural network model
# TODO:
#   - Implement neural network forward pass
# NOTES: Placeholder uses identity metric everywhere.
# ###########################################################################
def neural_metric_tensor(coords: torch.Tensor) -> torch.Tensor:
    size = coords.shape[0]
    eye = torch.eye(3, device=coords.device).expand(size, 3, 3)
    return eye


class BuildLaplace3D:
    """Constructs sparse Laplace matrices for a 3D grid."""

    def __init__(self, grid_domain: GridDomain, metric_func: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        self.grid_domain = grid_domain
        self.metric_func = metric_func or neural_metric_tensor

    def build_general_laplace(self) -> Tuple[coo_matrix, torch.Tensor]:
        coords = self.grid_domain.vertices.reshape(-1, 3)
        metric = self.metric_func(coords)
        n = coords.shape[0]
        data = []
        row = []
        col = []
        for i in range(n):
            # simple 6-neighbour stencil with unit spacing
            row.append(i)
            col.append(i)
            data.append(6.0)
            if i + 1 < n:
                row.append(i)
                col.append(i + 1)
                data.append(-1.0)
        laplace = coo_matrix((data, (row, col)), shape=(n, n))
        return laplace, metric
