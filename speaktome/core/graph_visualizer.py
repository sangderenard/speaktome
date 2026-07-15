#!/usr/bin/env python3
"""Live spring/repulsor view of a FluxGraph, rendered on its own thread.

Runs pygame + raw OpenGL (lines for edges, points for nodes) in a background
thread so it never blocks the graph's tick loop. Every frame it reads
``graph.published_snapshot`` -- a complete, ready-to-read dict the graph
republishes wholesale at the end of every seed()/spawn_first_children()/
tick() call, never a lock to race for (see FluxGraph.published_snapshot's
own docstring for why a lock-based version of this was unreliable).
ForceLayout (graph_layout.py) owns the actual spring/repulsor math and has
no pygame/OpenGL dependency, so that part is unit-testable without a
display; this module is just the render shell around it and can only
really be verified by watching the window.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from .graph_layout import ForceLayout
from .noodle_explorer import Direction

try:
    import pygame
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT, GL_LINES, GL_POINTS, GL_PROJECTION, GL_MODELVIEW,
        glBegin, glEnd, glVertex2f, glColor3f, glPointSize, glClear, glClearColor,
        glViewport, glMatrixMode, glLoadIdentity, glOrtho,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pygame = None
# --- END HEADER ---


class FluxGraphVisualizer:
    """Force-directed live view of a FluxGraph's nodes and edges, on its own thread."""

    def __init__(
        self,
        graph,
        width: int = 900,
        height: int = 700,
        fps: int = 30,
        layout: Optional[ForceLayout] = None,
    ):
        if pygame is None:
            raise RuntimeError("pygame and PyOpenGL are required for FluxGraphVisualizer")
        self.graph = graph
        self.width = width
        self.height = height
        self.fps = fps
        self.layout = layout or ForceLayout(width=width, height=height)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _snapshot(self) -> Optional[Dict[int, Tuple[Optional[int], bool, Optional[Direction], float]]]:
        """Read the graph's currently-published snapshot.

        A plain attribute read, no lock: graph.published_snapshot is only
        ever reassigned wholesale (never mutated in place) at the end of a
        seed()/spawn_first_children()/tick() call, and a single reference
        read/write is already atomic under CPython's GIL -- this either
        sees the previous complete snapshot or the new complete one, never
        something in between. Returns None only before the graph has been
        seeded at all.
        """
        return self.graph.published_snapshot

    def _layout_snapshot(
        self, snapshot: Dict[int, Tuple[Optional[int], bool, Optional[Direction], float]]
    ) -> Dict[int, Tuple[Optional[int], bool, float]]:
        return {
            nid: (parent_id, burned, elevation)
            for nid, (parent_id, burned, _direction, elevation) in snapshot.items()
        }

    def _draw(self, snapshot: Dict[int, Tuple[Optional[int], bool, Optional[Direction], float]]) -> None:
        positions = self.layout.positions
        glClear(GL_COLOR_BUFFER_BIT)

        glColor3f(0.4, 0.4, 0.4)
        glBegin(GL_LINES)
        for nid, (parent_id, burned, _direction, _elevation) in snapshot.items():
            if burned or parent_id is None:
                continue
            if nid not in positions or parent_id not in positions:
                continue
            x1, y1 = positions[nid]
            x2, y2 = positions[parent_id]
            glVertex2f(x1, y1)
            glVertex2f(x2, y2)
        glEnd()

        glPointSize(4.0)
        glBegin(GL_POINTS)
        for nid, (_parent_id, burned, direction, _elevation) in snapshot.items():
            if burned or nid not in positions:
                continue
            if direction is None:
                glColor3f(1.0, 1.0, 1.0)
            elif direction is Direction.FORWARD:
                glColor3f(0.3, 0.8, 1.0)
            else:
                glColor3f(1.0, 0.6, 0.2)
            x, y = positions[nid]
            glVertex2f(x, y)
        glEnd()

        pygame.display.flip()

    def _run(self) -> None:
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("FluxGraph -- live spring layout")
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glClearColor(0.05, 0.05, 0.08, 1.0)

        clock = pygame.time.Clock()
        last_snapshot: Dict[int, Tuple[Optional[int], bool, Optional[Direction], float]] = {}
        try:
            while not self._stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop_event.set()
                fresh = self._snapshot()
                if fresh is not None:
                    last_snapshot = fresh
                layout_snapshot = self._layout_snapshot(last_snapshot)
                self.layout.sync(layout_snapshot)
                self.layout.step(layout_snapshot)
                self._draw(last_snapshot)
                clock.tick(self.fps)
        finally:
            pygame.quit()
