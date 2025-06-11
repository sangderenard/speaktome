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
        # Glyph libraries keyed by (font_path, font_size)
        self.glyph_libraries: dict[tuple[str | None, int], dict[str, Any]] = {}
        # Post-processing kernels
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

    def load_font(
        self,
        font_path: str | None,
        font_size: int,
        characters: str = "".join(chr(i) for i in range(32, 127)),
    ) -> None:
        """Load a font into a glyph library using Pillow if available."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pillow is required to load fonts") from exc

        if font_path is None:
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, font_size)

        library: dict[str, Any] = {}
        for ch in characters:
            bbox = font.getbbox(ch)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            img = Image.new("L", (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), ch, fill=255, font=font)
            arr = [
                list(img.getdata()[i * width : (i + 1) * width])
                for i in range(height)
            ]
            glyph = self.tensor_ops.tensor_from_list(
                arr, dtype=self.tensor_ops.float_dtype, device=None
            )
            library[ch] = glyph

        self.glyph_libraries[(font_path, font_size)] = library

    def print_text(
        self,
        text: str,
        position: tuple[float, float],
        font_path: str | None,
        font_size: int,
        unit: str = "mm",
    ) -> Any:
        """Render text using a previously loaded font."""
        key = (font_path, font_size)
        if key not in self.glyph_libraries:
            self.load_font(font_path, font_size)

        x, y = position
        library = self.glyph_libraries[key]
        for ch in text:
            if ch == "\n":
                sample = next(iter(library.values()))
                g_height = self.tensor_ops.shape(sample)[0]
                # move down by glyph height in tensor units
                y -= self.ruler.tensor_to_coordinates(0, g_height, unit)[1]
                x = position[0]
                continue
            glyph = library.get(ch)
            if glyph is None:
                continue
            self.print_glyph(glyph, (x, y), unit=unit)
            g_width = self.tensor_ops.shape(glyph)[1]
            x += self.ruler.tensor_to_coordinates(g_width, 0, unit)[0]
        return self.canvas

