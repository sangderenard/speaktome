"""Tests for speaktome.core.graph_layout."""

import math

from speaktome.core.graph_layout import ForceLayout


def test_sync_adds_new_nodes_near_parent():
    layout = ForceLayout(width=200, height=200, seed=42)
    snap = {0: (None, False, 0.0), 1: (0, False, 1.0)}
    layout.sync(snap)
    assert 0 in layout.positions and 1 in layout.positions


def test_sync_drops_burned_nodes():
    layout = ForceLayout(width=200, height=200, seed=42)
    snap = {0: (None, False, 0.0), 1: (0, False, 1.0)}
    layout.sync(snap)
    burned_snap = {0: (None, False, 0.0), 1: (0, True, 1.0)}
    layout.sync(burned_snap)
    assert 1 not in layout.positions
    assert 0 in layout.positions


def test_repulsion_separates_nearly_overlapping_nodes():
    layout = ForceLayout(width=200, height=200, seed=42)
    snap = {0: (None, False, 0.0), 1: (0, False, 0.0)}
    layout.sync(snap)
    layout.positions[0] = (100.0, 100.0)
    layout.positions[1] = (100.5, 100.0)
    d0 = math.dist(layout.positions[0], layout.positions[1])
    for _ in range(30):
        layout.step(snap)
    d1 = math.dist(layout.positions[0], layout.positions[1])
    assert d1 > d0


def test_y_is_pinned_to_elevation_not_simulated():
    layout = ForceLayout(width=200, height=200, vertical_scale=10.0, seed=42)
    snap = {0: (None, False, 0.0), 1: (0, False, 2.0), 2: (0, False, -3.0)}
    layout.sync(snap)

    expected_y0 = layout.height / 2
    expected_y1 = layout.height / 2 - 20.0
    expected_y2 = layout.height / 2 + 30.0
    assert math.isclose(layout.positions[0][1], expected_y0)
    assert math.isclose(layout.positions[1][1], expected_y1)
    assert math.isclose(layout.positions[2][1], expected_y2)

    # forward (positive elevation) draws above the anchor (smaller y);
    # backward (negative elevation) draws below (larger y) -- screen-space
    # y increases downward.
    assert layout.positions[1][1] < layout.positions[0][1] < layout.positions[2][1]


def test_y_stays_pinned_across_physics_steps():
    layout = ForceLayout(width=200, height=200, vertical_scale=10.0, seed=42)
    snap = {0: (None, False, 0.0), 1: (0, False, 2.0), 2: (0, False, -3.0)}
    layout.sync(snap)
    for _ in range(30):
        layout.step(snap)
    assert math.isclose(layout.positions[1][1], layout.height / 2 - 20.0)
    assert math.isclose(layout.positions[2][1], layout.height / 2 + 30.0)


def test_x_moves_freely_under_repulsion_while_y_stays_fixed():
    layout = ForceLayout(width=200, height=200, vertical_scale=10.0, seed=42)
    # two nodes at the same elevation, nudged close together on x
    snap = {0: (None, False, 0.0), 1: (0, False, 0.0), 2: (0, False, 0.0)}
    layout.sync(snap)
    layout.positions[1] = (99.5, layout._pixel_y(0.0))
    layout.positions[2] = (100.5, layout._pixel_y(0.0))
    for _ in range(20):
        layout.step(snap)
    assert abs(layout.positions[1][0] - layout.positions[2][0]) > 1.0
    assert math.isclose(layout.positions[1][1], layout._pixel_y(0.0))
    assert math.isclose(layout.positions[2][1], layout._pixel_y(0.0))
