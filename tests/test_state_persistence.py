"""Tests for speaktome.flux_radar_server's autosave/resume persistence.

Uses the same lightweight dummy-model pattern as test_flux_graph.py (a
BigramDummyModel + FakeTokenizer) wired into a real ModelBundle instance
(bypassing __init__, which needs real nltk/wordfreq dictionary data) --
this exercises the real FluxGraph.restore_state / LiveSession.to_dict /
LiveSession.resume round trip with real (if tiny) graph growth, no GPU
or network access.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import torch

from tensors.torch_backend import PyTorchTensorOperations
from speaktome.core.model_abstraction import AbstractModelWrapper
from speaktome.core.choice_policy import TopKPolicy
from speaktome.core.implicit_backpath import ImplicitBackpathScorer
from speaktome.core.noodle_explorer import Direction
from speaktome.core.flux_graph import (
    FluxGraph,
    FluxGraphConfig,
    FluxNode,
    MaterialFactory,
)
from speaktome import flux_radar_server as server_mod

VOCAB = 5


class BigramDummyModel(AbstractModelWrapper):
    def __init__(self, table):
        self.table = table

    def forward(self, input_ids, attention_mask, **kwargs):
        table_t = torch.tensor(self.table, dtype=torch.float32)
        return {"logits": table_t[input_ids]}

    def get_device(self):
        return "cpu"


def _cycle_table(vocab=VOCAB, peak=10.0):
    table = []
    for i in range(vocab):
        row = [0.0] * vocab
        row[(i + 1) % vocab] = peak
        table.append(row)
    return table


class _DummyTokenizer:
    vocab_size = VOCAB
    eos_token_id = 0

    def encode(self, text):
        return [0]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def _dummy_bundle():
    """A real ModelBundle wired to a tiny dummy model -- can actually build/tick/restore graphs."""
    bundle = server_mod.ModelBundle.__new__(server_mod.ModelBundle)
    bundle.engine_name = "gpt2"
    bundle.dictionary_size = 20000
    bundle.min_word_len = 2
    bundle.max_word_len = None
    bundle.dictionary_enabled = True
    bundle.tokenizer = _DummyTokenizer()
    model = BigramDummyModel(_cycle_table())
    bundle.wrapper = model
    bundle.candidate_filter = None
    bundle.word_trie = None
    bundle.backward_word_trie = None
    bundle.ops = PyTorchTensorOperations(track_time=False)
    bundle.device = "cpu"
    bundle._run_lock = threading.Lock()
    return bundle


def test_node_to_dict_and_from_dict_roundtrip():
    node = FluxNode(
        id=7, tokens=[1, 2], direction=Direction.BACKWARD, parent_id=3, depth=2,
        local_evidence=-0.5, children_ids=[9, 10], pressure=1.25, low_pressure_ticks=2,
        created_tick=4, burned=False, expanded=True, cumulative_evidence=-1.0,
        rollup_mean=-0.5, subtree_auxin=0.1, auxin_level=0.2,
        hull_permeability=0.4, pore_permeabilities={"oxygen": 0.25},
        factories=[
            MaterialFactory(
                name="test synthesis",
                inputs={"a": 2.0},
                outputs={"auxin": 1.0},
                medium="csf",
                throughput=0.5,
            )
        ],
        factory_auxin=0.75,
    )
    restored = server_mod._node_from_dict(server_mod._node_to_dict(node))
    assert restored == node


def test_node_to_dict_and_from_dict_roundtrip_anchor_node():
    anchor = FluxNode(id=0, tokens=[], direction=None, parent_id=None, depth=0, local_evidence=0.0, pressure=1.05)
    restored = server_mod._node_from_dict(server_mod._node_to_dict(anchor))
    assert restored == anchor


def test_live_session_to_dict_and_resume_produces_an_equivalent_tickable_graph():
    bundle = _dummy_bundle()
    params = {"seed": "hi", "budget": 2, "branch": 2, "window": 10, "interval_ms": 100000}
    session = server_mod.LiveSession(bundle, params)
    session.graph.tick()
    session.graph.tick()
    saved = session.to_dict()

    resumed = server_mod.LiveSession.resume(bundle, saved)

    assert resumed.graph.anchor_id == session.graph.anchor_id
    assert resumed.graph.anchor_tokens == session.graph.anchor_tokens
    assert resumed.graph.tick_count == session.graph.tick_count
    assert set(resumed.graph.nodes.keys()) == set(session.graph.nodes.keys())
    for nid, node in session.graph.nodes.items():
        assert resumed.graph.nodes[nid] == node
    assert resumed.tick_index == session.tick_index
    assert list(resumed.history) == list(session.history)

    # The resumed graph is a real, live graph -- ticking it further doesn't raise.
    resumed.graph.tick()
    assert resumed.graph.tick_count == session.graph.tick_count + 1


def test_write_and_read_state_file_roundtrip(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")

    payload = {"last_params": {"seed": "x"}, "live_session": None}
    server_mod._write_state_file(payload)
    loaded = server_mod._read_state_file()
    assert loaded == payload


def test_read_state_file_missing_returns_empty_dict(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "does_not_exist.json")
    assert server_mod._read_state_file() == {}


def test_read_state_file_corrupt_returns_empty_dict_not_a_crash(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    state_file = tmp_dir / "state.json"
    state_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(server_mod, "STATE_FILE", state_file)
    assert server_mod._read_state_file() == {}


def test_autosave_writes_last_params_and_live_session(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")
    monkeypatch.setattr(server_mod, "_last_params", {"seed": "hello"})

    bundle = _dummy_bundle()
    session = server_mod.LiveSession(bundle, {"seed": "hi", "window": 5, "interval_ms": 100000})
    monkeypatch.setattr(server_mod, "_live_session", session)

    server_mod._autosave()

    on_disk = json.loads(server_mod.STATE_FILE.read_text(encoding="utf-8"))
    assert on_disk["last_params"] == {"seed": "hello"}
    assert on_disk["live_session"]["seed_text"] == "hi"
    assert on_disk["live_session"]["graph"]["anchor_id"] == session.graph.anchor_id


def test_clear_persisted_live_session_keeps_last_params_drops_live_session(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")
    server_mod._write_state_file({"last_params": {"seed": "keep-me"}, "live_session": {"tick_index": 5}})

    server_mod._clear_persisted_live_session()

    on_disk = json.loads(server_mod.STATE_FILE.read_text(encoding="utf-8"))
    assert on_disk["last_params"] == {"seed": "keep-me"}
    assert "live_session" not in on_disk


def test_resume_saved_state_restores_last_params_with_no_live_session(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")
    monkeypatch.setattr(server_mod, "_last_params", None)
    monkeypatch.setattr(server_mod, "_live_session", None)
    server_mod._write_state_file({"last_params": {"seed": "restored"}})

    server_mod._resume_saved_state()

    assert server_mod._last_params == {"seed": "restored"}
    assert server_mod._live_session is None


def test_resume_saved_state_restarts_ticking_from_saved_live_session(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")
    monkeypatch.setattr(server_mod, "_last_params", None)
    monkeypatch.setattr(server_mod, "_live_session", None)

    bundle = _dummy_bundle()
    monkeypatch.setattr(server_mod, "get_bundle", lambda params=None: bundle)

    original = server_mod.LiveSession(bundle, {"seed": "hi", "budget": 2, "branch": 2, "window": 5, "interval_ms": 100})
    original.graph.tick()
    saved_state = {"last_params": {"seed": "hi"}, "live_session": original.to_dict()}
    server_mod._write_state_file(saved_state)

    try:
        server_mod._resume_saved_state()
        assert server_mod._live_session is not None
        assert server_mod._live_session.tick_index == original.tick_index
        deadline = time.time() + 3
        while server_mod._live_session.tick_index == original.tick_index and time.time() < deadline:
            time.sleep(0.02)
        assert server_mod._live_session.tick_index > original.tick_index  # actually resumed ticking
    finally:
        if server_mod._live_session is not None:
            server_mod._live_session.stop()


def test_resume_saved_state_with_bad_live_session_does_not_raise(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server_mod, "STATE_FILE", tmp_dir / "state.json")
    monkeypatch.setattr(server_mod, "_last_params", None)
    monkeypatch.setattr(server_mod, "_live_session", None)
    server_mod._write_state_file({"last_params": None, "live_session": {"totally": "malformed"}})

    server_mod._resume_saved_state()  # must not raise

    assert server_mod._live_session is None
