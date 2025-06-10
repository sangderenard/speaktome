#!/usr/bin/env python3
"""Pygame window for displaying pixel subunit grids."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import numpy as np
    import pygame
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class SubunitWindow:
    """Display a grid of pixel data using Pygame."""

    def __init__(self, grid_shape: tuple[int, int], subunit_size: int = 10) -> None:
        """Create a window sized for ``grid_shape``."""
        pygame.init()
        self.subunit_size = max(1, subunit_size)
        width = grid_shape[1] * self.subunit_size
        height = grid_shape[0] * self.subunit_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Subunit Grid")

    def update_subunits(
        self,
        changes: list[tuple[int, int, np.ndarray]],
    ) -> None:
        """Update the window with ``changes`` in-place."""
        for y, x, sub in changes:
            surf = pygame.surfarray.make_surface(sub.swapaxes(0, 1))
            surf = pygame.transform.scale(
                surf,
                (sub.shape[1] * self.subunit_size, sub.shape[0] * self.subunit_size),
            )
            self.screen.blit(surf, (x * self.subunit_size, y * self.subunit_size))
        pygame.display.update()

    def close(self) -> None:
        pygame.quit()
