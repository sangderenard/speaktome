"""Utility helpers for array operations with optional torch support."""

import numpy as np
# --- END HEADER ---

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - runtime path
    torch = None
    TORCH_AVAILABLE = False


def topk(values, k, dim=-1, largest=True):
    """Return top-k values and indices using torch when available."""
    if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
        return torch.topk(values, k=k, dim=dim, largest=largest)

    # NumPy fallback. Assumes ``values`` is a NumPy array.
    axis = dim
    if not largest:
        indices = np.argpartition(values, k, axis=axis)[..., :k]
    else:
        indices = np.argpartition(values, -k, axis=axis)[..., -k:]
    gathered = np.take_along_axis(values, indices, axis=axis)
    order = np.argsort(gathered, axis=axis)
    if largest:
        order = order[..., ::-1]
    sorted_indices = np.take_along_axis(indices, order, axis=axis)
    sorted_values = np.take_along_axis(values, sorted_indices, axis=axis)
    return sorted_values, sorted_indices
