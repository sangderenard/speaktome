"""Coordinate translation utilities for the Grand Printing Press."""
# --- END HEADER ---

class Ruler:
    """Convert between physical units and tensor indices."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def coordinates_to_tensor(self, x: float, y: float, unit: str = "mm") -> tuple[int, int]:
        """Translate coordinates in ``unit`` to tensor indices."""
        if unit == "mm":
            x_in = x / 25.4
            y_in = y / 25.4
        elif unit == "inch":
            x_in = x
            y_in = y
        elif unit == "count":
            return int(x), int(y)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        return int(x_in * self.dpi), int(y_in * self.dpi)

    def tensor_to_coordinates(self, x_idx: int, y_idx: int, unit: str = "mm") -> tuple[float, float]:
        """Translate tensor indices back to coordinates in ``unit``."""
        if unit == "mm":
            x_val = x_idx / self.dpi * 25.4
            y_val = y_idx / self.dpi * 25.4
        elif unit == "inch":
            x_val = x_idx / self.dpi
            y_val = y_idx / self.dpi
        elif unit == "count":
            return float(x_idx), float(y_idx)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        return x_val, y_val
