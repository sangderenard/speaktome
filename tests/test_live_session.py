"""Tests for speaktome.flux_radar_server.LiveSession -- pure orchestration logic.

Uses a fake bundle (no real FluxGraph/model) so these run fast and without
GPU/network -- the thing worth testing carefully here is the rolling
window ("burn the oldest tick"), clean stop, and error surfacing, not
FluxGraph's own tick semantics (covered elsewhere).
"""

import tempfile
import threading
import time
from pathlib import Path

from speaktome import flux_radar_server as server_mod
from speaktome.flux_radar_server import LiveSession


def _isolate_state_file(monkeypatch):
    """Every test that actually ticks a LiveSession triggers _autosave() (see
    LiveSession._run_loop) -- point it at a throwaway file so these tests
    never write to the real speaktome/flux_radar_state.json.
    """
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")


class FakeGraph:
    def __init__(self):
        self.ticks = 0
        self.spawned = False

    def spawn_first_children(self):
        self.spawned = True

    def tick(self):
        self.ticks += 1
        if hasattr(self, "status_callback"):
            self.status_callback({
                "tick": self.ticks, "phase": "complete", "detail": "fake",
                "current": 1, "total": 1, "live_nodes": 1, "elapsed_seconds": 0.01,
            })

    def set_status_callback(self, callback):
        self.status_callback = callback


class FakeBundle:
    def __init__(self, fail_after=None):
        self._run_lock = threading.Lock()
        self.fail_after = fail_after

    def build_graph(self, params):
        return FakeGraph(), params.get("seed", "seed"), {"budget": 3}

    def snapshot_graph(self, graph):
        if self.fail_after is not None and graph.ticks >= self.fail_after:
            raise RuntimeError("simulated tick failure")
        return {"nodes": [], "best_path": "", "best_score": 0.0, "ticks_seen": graph.ticks}


def test_start_records_tick_zero_immediately(monkeypatch):
    _isolate_state_file(monkeypatch)
    session = LiveSession(FakeBundle(), {"window": 5, "interval_ms": 100000})
    session.start()
    try:
        state = session.snapshot_state()
        assert state["tick_index"] == 0
        assert len(state["history"]) == 1
        assert state["history"][0]["tick"] == 0
        assert state["running"] is True
    finally:
        session.stop()


def test_live_state_includes_latest_mid_tick_status(monkeypatch):
    _isolate_state_file(monkeypatch)
    session = LiveSession(FakeBundle(), {"window": 5, "interval_ms": 200})
    session.start()
    try:
        deadline = time.time() + 5
        while not session.snapshot_state()["mid_tick"] and time.time() < deadline:
            time.sleep(0.05)
        status = session.snapshot_state()["mid_tick"]
        assert status["phase"] == "complete"
        assert status["tick"] >= 1
    finally:
        session.stop()


def test_rolling_window_burns_the_oldest_tick(monkeypatch):
    _isolate_state_file(monkeypatch)
    session = LiveSession(FakeBundle(), {"window": 3, "interval_ms": 200})
    session.start()
    try:
        deadline = time.time() + 5
        while session.tick_index < 5 and time.time() < deadline:
            time.sleep(0.05)
        state = session.snapshot_state()
        assert len(state["history"]) <= 3  # never exceeds the window
        ticks = [h["tick"] for h in state["history"]]
        assert ticks == sorted(ticks)  # oldest dropped, not newest
        assert ticks[-1] == state["tick_index"]
    finally:
        session.stop()


def test_stop_actually_halts_the_background_thread(monkeypatch):
    _isolate_state_file(monkeypatch)
    session = LiveSession(FakeBundle(), {"window": 5, "interval_ms": 150})
    session.start()
    time.sleep(0.3)
    session.stop()
    ticks_at_stop = session.tick_index
    time.sleep(0.5)
    assert session.tick_index == ticks_at_stop  # no further ticks after stop
    assert session.snapshot_state()["running"] is False


def test_interval_is_floored_to_a_safe_minimum():
    # Construction-time clamping only -- never started, so nothing to stop.
    session = LiveSession(FakeBundle(), {"interval_ms": 1})  # way below the floor
    assert session.interval >= LiveSession.MIN_INTERVAL_S


def test_window_is_capped_to_a_sane_maximum():
    session = LiveSession(FakeBundle(), {"window": 999999})
    assert session.window <= LiveSession.MAX_WINDOW


def test_a_failing_tick_surfaces_as_error_and_stops_the_loop(monkeypatch):
    _isolate_state_file(monkeypatch)
    session = LiveSession(FakeBundle(fail_after=1), {"window": 5, "interval_ms": 100})
    session.start()
    try:
        deadline = time.time() + 5
        while session.snapshot_state()["running"] and time.time() < deadline:
            time.sleep(0.05)
        state = session.snapshot_state()
        assert state["running"] is False
        assert "simulated tick failure" in state["error"]
    finally:
        session.stop()
