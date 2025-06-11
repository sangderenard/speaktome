"""Grand Printing Press core machinery."""
# --- END HEADER ---

from __future__ import annotations

from typing import Any
from tensors.abstraction import AbstractTensor

from .ruler import Ruler


class GrandPrintingPress:
    """Compose tensor glyphs and apply complex post-processing."""

    def __init__(
        self,
        tensor_ops: AbstractTensor,
        page_size: tuple[int, int] = (512, 512),
        dpi: int = 300,
    ) -> None:
        self.tensor_ops = tensor_ops
        self.page_size = page_size
        self.ruler = Ruler(dpi)
        height, width = page_size
        self.canvas = tensor_ops.zeros(
            (height, width), dtype=tensor_ops.float_dtype, device=None
        )
        # Initialize glyph library and kernel pipeline
        self.glyph_library: dict[str, Any] = {}
        self.kernels: list[callable] = []

    def add_kernel(self, func: callable) -> None:
        """Register a post-processing kernel."""
        self.kernels.append(func)

    def print_glyph(
        self,
        glyph: Any,
        position: tuple[float, float],
        unit: str = "mm",
    ) -> Any:
        """Apply a glyph tensor at the given position."""
        # Basic placement logic with optional kernel hooks
        y_idx, x_idx = self.ruler.coordinates_to_tensor(position[0], position[1], unit)
        g_height, g_width = self.tensor_ops.shape(glyph)
        c_height, c_width = self.tensor_ops.shape(self.canvas)

        end_y = min(y_idx + g_height, c_height)
        end_x = min(x_idx + g_width, c_width)
        if end_y <= y_idx or end_x <= x_idx:
            return self.canvas

        sub_glyph = glyph[: end_y - y_idx, : end_x - x_idx]
        for row in range(self.tensor_ops.shape(sub_glyph)[0]):
            indices_dim0 = [y_idx + row] * (end_x - x_idx)
            indices_dim1 = list(range(x_idx, end_x))
            values = self.tensor_ops.tolist(sub_glyph[row])
            self.tensor_ops.assign_at_indices(
                self.canvas, indices_dim0, indices_dim1, values
            )

        for kernel in self.kernels:
            self.canvas = kernel(self.canvas)
        return self.canvas

    def finalize_page(self) -> Any:
        """Return the completed tensor page."""
        output = self.canvas
        for kernel in self.kernels:
            output = kernel(output)
        return self.tensor_ops.clamp(output, 0.0, 1.0)
