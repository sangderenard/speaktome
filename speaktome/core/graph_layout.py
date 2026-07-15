#!/usr/bin/env python3
"""Pure spring/repulsor force-directed layout, with no rendering dependency.

Kept separate from graph_visualizer.py so the physics (does a new node get
placed near its parent? does a burned node get dropped? does repulsion push
apart overlapping nodes? does elevation pin the vertical axis correctly?)
can be tested without pygame/OpenGL or a display.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple
# --- END HEADER ---


class ForceLayout:
    """2D spring-electrical layout over an externally-owned node/edge snapshot.

    ``snapshot`` is a plain ``{node_id: (parent_id, burned, elevation)}``
    dict, refreshed by the caller every frame/step -- this class holds no
    reference to a graph object, only the position/velocity state it's
    accumulated for whichever node ids it has seen. ``elevation`` is
    FluxNode.height (signed token-distance from the anchor: positive
    forward, negative backward) -- pure topology, not a simulated quantity.

    Only the horizontal axis is a real simulation (repulsion pushes
    overlapping nodes apart, springs pull children toward their parent at a
    resting length, a weak center pull keeps the whole thing from drifting
    off-screen). The vertical axis is not simulated at all: every node's Y
    is set directly from its elevation every step, so "forward floats up,
    backward sinks down" is a fixed fact about the tree, not a force that
    has to be tuned to balance against anything else.
    """

    def __init__(
        self,
        width: float = 900.0,
        height: float = 700.0,
        repulsion: float = 4000.0,
        spring_length: float = 60.0,
        spring_strength: float = 0.02,
        damping: float = 0.85,
        center_pull: float = 0.001,
        dt: float = 0.02,
        vertical_scale: float = 40.0,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.repulsion = repulsion
        self.spring_length = spring_length
        self.spring_strength = spring_strength
        self.damping = damping
        self.center_pull = center_pull
        self.dt = dt
        self.vertical_scale = vertical_scale
        self._rng = random.Random(seed)

        self.positions: Dict[int, Tuple[float, float]] = {}
        self.velocities: Dict[int, float] = {}  # horizontal velocity only

    def _pixel_y(self, elevation: float) -> float:
        return self.height / 2 - elevation * self.vertical_scale

    def sync(self, snapshot: Dict[int, Tuple[Optional[int], bool, float]]) -> None:
        """Add newly-seen live nodes, drop burned/vanished ones."""
        live_ids = {nid for nid, (_, burned, _elevation) in snapshot.items() if not burned}

        for nid in live_ids:
            if nid in self.positions:
                continue
            parent_id, _burned, elevation = snapshot[nid]
            if parent_id is not None and parent_id in self.positions:
                px, _py = self.positions[parent_id]
            else:
                px = self.width / 2
            jitter = (self._rng.random() - 0.5) * 10
            self.positions[nid] = (px + jitter, self._pixel_y(elevation))
            self.velocities[nid] = 0.0

        for nid in list(self.positions):
            if nid not in live_ids:
                self.positions.pop(nid, None)
                self.velocities.pop(nid, None)

    def step(self, snapshot: Dict[int, Tuple[Optional[int], bool, float]]) -> None:
        """One physics tick: horizontal repulsion + springs; vertical is pinned, not simulated."""
        live = [nid for nid, (_, burned, _e) in snapshot.items() if not burned and nid in self.positions]
        forces: Dict[int, float] = {nid: 0.0 for nid in live}

        for i, a in enumerate(live):
            ax, ay = self.positions[a]
            for b in live[i + 1:]:
                bx, by = self.positions[b]
                dx, dy = ax - bx, ay - by
                dist_sq = dx * dx + dy * dy + 0.01
                dist = math.sqrt(dist_sq)
                force = self.repulsion / dist_sq
                fx = force * dx / dist
                forces[a] += fx
                forces[b] -= fx

        for nid in live:
            parent_id, _burned, _e = snapshot[nid]
            if parent_id is None or parent_id not in self.positions:
                continue
            ax, ay = self.positions[nid]
            bx, by = self.positions[parent_id]
            dx, dy = bx - ax, by - ay
            dist = math.sqrt(dx * dx + dy * dy) + 0.01
            stretch = dist - self.spring_length
            fx = self.spring_strength * stretch * dx / dist
            forces[nid] += fx
            if parent_id in forces:
                forces[parent_id] -= fx

        cx = self.width / 2
        for nid in live:
            x, _y = self.positions[nid]
            forces[nid] += (cx - x) * self.center_pull

        for nid in live:
            vx = self.velocities.get(nid, 0.0)
            fx = forces.get(nid, 0.0)
            vx = (vx + fx * self.dt) * self.damping
            self.velocities[nid] = vx
            x, _y = self.positions[nid]
            elevation = snapshot[nid][2]
            self.positions[nid] = (x + vx * self.dt, self._pixel_y(elevation))
