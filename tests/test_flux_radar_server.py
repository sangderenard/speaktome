"""Tests for speaktome.flux_radar_server -- HTTP routing/safety only, no real model load."""

import json
import threading
import urllib.request
import urllib.error

from http.server import ThreadingHTTPServer

from speaktome import flux_radar_server as server_mod
from speaktome.core.poetic_attractor import PoeticAttractor


def _start_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


class _FakeTokenizer:
    eos_token_id = 0
    vocab_size = 5

    def encode(self, text):
        return [1, 2]

    def decode(self, ids):
        return "x" * len(ids)


def _fake_bundle():
    """A ModelBundle with a real build_graph() but no real model/tokenizer loaded.

    build_graph() never calls the model (FluxGraph/ImplicitBackpathScorer
    both just store references at construction time; the model is only
    ever touched lazily during tick()), so this exercises the real
    param-resolution logic (ngram/poetic/engine/dictionary/decay/...)
    without any GPU work.
    """
    bundle = server_mod.ModelBundle.__new__(server_mod.ModelBundle)
    bundle.engine_name = "gpt2"
    bundle.dictionary_size = 20000
    bundle.min_word_len = 2
    bundle.max_word_len = None
    bundle.dictionary_enabled = True
    bundle.tokenizer = _FakeTokenizer()
    bundle.wrapper = object()
    bundle.candidate_filter = None
    bundle.word_trie = None
    bundle.backward_word_trie = None
    bundle.ops = object()
    bundle.device = "cpu"
    return bundle


def test_build_graph_ngram_size_zero_means_disabled():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"no_repeat_ngram_size": 0})
    assert graph.config.no_repeat_ngram_size is None
    assert resolved["no_repeat_ngram_size"] == 0


def test_build_graph_ngram_size_defaults_to_three():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({})
    assert graph.config.no_repeat_ngram_size == 3
    assert resolved["no_repeat_ngram_size"] == 3


def test_build_graph_enables_global_csf_scarcity_and_rhizome_defaults():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({})

    assert graph.config.growth_target_ion_concentration == 0.1
    assert graph.config.csf_link_rate == 0.05
    assert graph.config.lymph_return_rate == 0.02
    assert graph.config.rhizome_csf_pump_rate == 0.1
    assert graph.config.rhizome_soil_exudation_rate == 0.01
    assert graph.config.growth_commitment_threshold == 3.0
    assert graph.config.habitat_ring_ion_amount == 4.0
    assert graph.config.branch_maturity_gain == 0.1
    assert graph.config.physiology_learning_enabled is True
    assert graph.config.physiology_learning_rate == 0.05
    assert graph.config.physiology_resource_cost == 0.1
    assert graph.config.physiology_initial_opening == 0.8
    assert graph.config.physiology_traversal_temperature == 0.5
    assert graph.config.physiology_track_model_gradients is False
    assert resolved["rhizome_csf_pump_rate"] == 0.1
    assert resolved["growth_commitment_threshold"] == 3.0
    assert resolved["habitat_ring_ion_amount"] == 4.0
    assert resolved["physiology_learning_enabled"] is True

def test_build_graph_poetic_booster_off_by_default():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({})
    assert graph.config.poetic_attractor is None
    assert resolved["poetic_enabled"] is False


def test_build_graph_poetic_booster_enabled_builds_a_real_attractor():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"poetic_enabled": True, "poetic_scale": 2.5})
    assert isinstance(graph.config.poetic_attractor, PoeticAttractor)
    assert graph.config.poetic_scale == 2.5
    assert resolved["poetic_enabled"] is True
    assert resolved["poetic_scale"] == 2.5


def test_build_graph_reports_its_own_engine_name():
    bundle = _fake_bundle()
    _, _, resolved = bundle.build_graph({})
    assert resolved["engine"] == "gpt2"


def test_build_graph_reports_its_own_dictionary_params():
    bundle = _fake_bundle()
    bundle.dictionary_size = 5000
    bundle.min_word_len = 3
    bundle.max_word_len = 8
    bundle.dictionary_enabled = False
    _, _, resolved = bundle.build_graph({})
    assert resolved["dictionary_size"] == 5000
    assert resolved["min_word_len"] == 3
    assert resolved["max_word_len"] == 8
    assert resolved["dictionary_enabled"] is False


def test_build_graph_decay_and_population_target_zero_means_off():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"decay_rate": 0.3, "population_target": 0})
    assert graph.config.decay_rate == 0.3
    assert graph.config.population_target is None
    assert resolved["decay_rate"] == 0.3
    assert resolved["population_target"] == 0


def test_build_graph_population_target_passes_through_when_set():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"population_target": 40})
    assert graph.config.population_target == 40
    assert resolved["population_target"] == 40


def test_build_graph_shared_token_pressure_defaults_on():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({})
    assert graph.config.shared_token_pressure_enabled is True
    assert resolved["shared_token_pressure_enabled"] is True


def test_build_graph_shared_token_pressure_can_be_disabled():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"shared_token_pressure_enabled": False})
    assert graph.config.shared_token_pressure_enabled is False
    assert resolved["shared_token_pressure_enabled"] is False


def test_build_graph_max_expand_elements_zero_means_unbounded():
    bundle = _fake_bundle()
    graph, _, resolved = bundle.build_graph({"max_expand_elements": 0})
    assert graph.config.max_expand_elements is None
    assert resolved["max_expand_elements"] == 0


def test_build_graph_physics_fields_pass_through():
    bundle = _fake_bundle()
    params = {
        "found_bonus": 0.1, "damping": 0.7, "starvation_floor": 0.2, "burn_after_ticks": 5,
        "max_context_tokens": 32, "auxin_suppression": 0.5, "auxin_decay": 0.4,
        "head_pressure_coefficient": 0.02, "poetic_shortlist_k": 15,
    }
    graph, _, resolved = bundle.build_graph(params)
    for key, value in params.items():
        assert getattr(graph.config, key) == value
        assert resolved[key] == value


def test_get_root_serves_index_html():
    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "FluxGraph radar" in body
    finally:
        httpd.shutdown()


def test_radar_serves_bundled_threejs_webgl2_renderer():
    bundle_path = server_mod.STATIC_DIR / "webgl" / "webgl_renderer.js"
    source_path = server_mod.STATIC_DIR / "src" / "webgl_renderer.ts"
    index_source = (server_mod.STATIC_DIR / "index.html").read_text(
        encoding="utf-8"
    )
    assert bundle_path.is_file()
    assert source_path.is_file()
    assert 'id="fgWebGL"' in index_source
    assert 'from "/static/webgl/webgl_renderer.js"' in index_source
    assert "currentTubeState = snap.tube_state || []" in index_source
    source = source_path.read_text(encoding="utf-8")
    assert "new THREE.WebGLRenderer" in source
    assert "new THREE.InstancedMesh" in source
    assert "rebuildLumens" in source
    assert "dominantFluid" in source

    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/static/webgl/webgl_renderer.js"
        ) as resp:
            assert resp.status == 200
            assert resp.headers.get_content_type() == "application/javascript"
            assert b"FluxWebGLRenderer" in resp.read()
    finally:
        httpd.shutdown()


def test_radar_uses_change_aware_hydraulic_straightening():
    """The browser layout must not kick a settled graph on every live poll."""
    index_path = server_mod.STATIC_DIR / "index.html"
    source = index_path.read_text(encoding="utf-8")

    assert "pipeStraighteningForce" in source
    assert "Math.abs(flow * pressureDrop)" in source
    assert "snapshotChanged" in source
    assert "if (snapshotChanged)" in source
    assert "simulation.alpha(0.9).restart()" not in source
    assert "pipeRestLength" in source
    assert "pressureLengthFactor" in source
    assert "component_flows" in source
    assert "fg-water-flow" in source
    assert "fg-ion-flow" in source
    assert "regionBindingForce" in source
    assert "ringWedgeCollisionForce" in source
    assert "enforceNetworkGeometry" in source
    assert ".force('collide', ringWedgeCollisionForce" in source
    assert "node.x = CX + radius * radialX" in source
    assert "addRayLoad(direction + ':' + boundaryIndex" in source
    assert "addRayLoad" in source
    assert "rayContainmentForce" not in source


def test_ring_depth_is_absolute_signed_level_for_main_and_cousin_seeds():
    from types import SimpleNamespace

    assert server_mod._ring_index_from_level(SimpleNamespace(level=0)) == 0
    assert server_mod._ring_index_from_level(SimpleNamespace(level=-1)) == 1
    assert server_mod._ring_index_from_level(SimpleNamespace(level=1)) == 1
    assert server_mod._ring_index_from_level(SimpleNamespace(level=-2)) == 2
    assert server_mod._ring_index_from_level(SimpleNamespace(level=2)) == 2


def test_radar_uses_one_authoritative_hop_radius_and_recovery_force():
    source = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "const initialRadius = targetRadius(n)" in source
    assert "sn.targetR = targetRadius(n)" in source
    assert "ringData = currentRingRadii.map" in source
    assert "ringData = displayRingRadii.map" not in source
    assert "depthRingRecoveryForce" in source
    assert "node.x += shiftX; node.y += shiftY" in source
    assert "selectAll('line.fg-edge')" in source
    assert ".attr('class', 'fg-edge')" in source

def test_radar_has_snapshot_driven_heart_and_storage_hud():
    source = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="fgHeartHud"' in source
    assert 'id="fgHeartHudBody"' in source
    assert "currentReservoirs = snap.reservoirs || {}" in source
    assert "function renderHeartHud()" in source
    assert "chamberHudValue" in source
    for field in (
        "storage_volume", "ion_amount", "solvent", "fullness",
        "concentration", "band", "owner_id", "ion_name",
    ):
        assert field in source
    assert "ventricles" in source
    assert "renderHeartHud();" in source
    assert "const tokenString = pumpNode" in source
    assert "heartNameNode.title = tokenString" in source
    assert "currentRhizome = snap.rhizome || {}" in source
    assert "currentHabitat = snap.current_habitat || {}" in source
    assert "currentPhysiology = snap.physiology || {}" in source
    assert "learned physiology" in source
    assert "global fluid" in source
    assert "CSF / lymph" in source
    assert "const ionName = String(store.ion_name || name)" in source
    assert "`${ionName} ${hudNumber(store.ion_amount)}`" in source

def test_snapshot_preserves_each_reservoirs_full_ion_name():
    bundle = _fake_bundle()
    graph, _, _ = bundle.build_graph({"seed": "x"})
    graph.rhizome["waste:salt"] = 0.4

    snapshot = bundle.snapshot_graph(graph)

    stores = snapshot["reservoirs"]["main"]
    assert stores["main:forward"]["ion_name"] == "main:forward"
    assert stores["main:backward"]["ion_name"] == "main:backward"
    assert snapshot["rhizome"] == {"waste:salt": 0.4}
    assert snapshot["rhizome_owner_id"] == graph.anchor_id
    assert snapshot["current_habitat_id"] == graph.anchor_id
    assert snapshot["current_habitat"]["main:forward"] == 4.0
    assert snapshot["current_habitat_phrase"] == bundle.tokenizer.decode(
        graph.habitat_signatures[graph.anchor_id]
    )
    assert snapshot["movement_count"] == 0
    assert snapshot["physiology"]["enabled"] is True
    assert snapshot["physiology"]["parameter_count"] == 60
    assert snapshot["physiology"]["archetype_parameter_count"] == 60
    assert all(
        key.startswith("archetype:")
        for key in graph.physiology_parameters
    )

def test_radar_exposes_separate_sprout_and_air_root_shapes():
    from pathlib import Path

    source = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    server_source = Path(server_mod.__file__).read_text(encoding="utf-8")
    assert 'value="stddev"' in source

    for field_id, param in (
        ("fSproutBranch", "sprout_branch_factor"),
        ("fSproutDepth", "sprout_hot_loop_depth"),
        ("fAirRootBranch", "air_root_branch_factor"),
        ("fAirRootDepth", "air_root_hot_loop_depth"),
        ("fGrowthIonTarget", "growth_target_ion_concentration"),
        ("fCsfLinkRate", "csf_link_rate"),
        ("fLymphReturnRate", "lymph_return_rate"),
        ("fRhizomePumpRate", "rhizome_csf_pump_rate"),
        ("fRhizomeExudeRate", "rhizome_soil_exudation_rate"),
        ("fPhysiologyEnabled", "physiology_learning_enabled"),
        ("fPhysiologyRate", "physiology_learning_rate"),
        ("fPhysiologyCost", "physiology_resource_cost"),
        ("fPhysiologyInitial", "physiology_initial_opening"),
        ("fPhysiologyTemperature", "physiology_traversal_temperature"),
        ("fModelGradients", "physiology_track_model_gradients"),
        ("fCommitmentGain", "growth_commitment_gain"),
        ("fCommitmentRetention", "growth_commitment_retention"),
        ("fCommitmentThreshold", "growth_commitment_threshold"),
        ("fHabitatIonAmount", "habitat_ring_ion_amount"),
        ("fRingUptake", "ring_uptake_rate"),
        ("fMaturityGain", "branch_maturity_gain"),
        ("fMaturityBonus", "branch_maturity_conductance_bonus"),
    ):
        assert f'id="{field_id}"' in source
        assert f"'{field_id}', '{param}'" in source
        assert param in server_source

def test_radar_caches_snapshot_data_outside_the_animation_frame():
    from pathlib import Path

    source = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "function rebuildSnapshotRenderCaches(nodesArr)" in source
    assert "rebuildSnapshotRenderCaches(nodesArr);" in source
    assert "currentPressureData" in source
    assert "currentWaterData" in source
    assert "currentIonData" in source
    assert "currentFluidData" in source
    assert "currentLabelData" in source
    assert "link.influence = currentEdgeInfluence.get" in source
    assert "const materialByLink = currentLinks.map" not in source
    assert "const ranked = nodesArr.filter" in source
    server_source = Path(server_mod.__file__).read_text(encoding="utf-8")
    assert "network_roots = graph.orthogonal_network_roots(orthogonal)" in server_source

def test_radar_refresh_uses_compact_live_state_and_visible_rings():
    index_path = server_mod.STATIC_DIR / "index.html"
    source = index_path.read_text(encoding="utf-8")

    assert "/api/live/state?latest=1" in source
    assert "/api/live/state?after_tick=" in source
    assert "liveLastSnapshotTick" in source
    assert ".attr('class', 'fg-ring')" in source
    assert ".attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 1.15)" in source


def test_live_session_snapshot_state_filters_history_without_changing_storage():
    from collections import deque

    session = server_mod.LiveSession.__new__(server_mod.LiveSession)
    session.seed_text = "seed"
    session.params = {"window": 3}
    session.tick_index = 3
    session.error = None
    session.mid_tick_status = {}
    session.history = deque([{"tick": 1}, {"tick": 2}, {"tick": 3}], maxlen=3)
    session._state_lock = threading.RLock()
    session._stop_event = threading.Event()
    session._thread = type("AliveThread", (), {"is_alive": lambda self: True})()

    assert [s["tick"] for s in session.snapshot_state()["history"]] == [1, 2, 3]
    assert [s["tick"] for s in session.snapshot_state(latest_only=True)["history"]] == [3]
    assert [s["tick"] for s in session.snapshot_state(after_tick=1)["history"]] == [2, 3]
    assert session.snapshot_state(after_tick=3)["history"] == []
    assert [s["tick"] for s in session.history] == [1, 2, 3]


def test_api_live_state_passes_refresh_and_incremental_filters(monkeypatch):
    calls = []

    class FakeSession:
        def snapshot_state(self, after_tick=None, latest_only=False):
            calls.append((after_tick, latest_only))
            return {
                "seed_text": "seed", "params": {"window": 3}, "tick_index": 3,
                "running": True, "error": None, "mid_tick": {}, "history": [],
            }

    monkeypatch.setattr(server_mod, "_live_session", FakeSession())
    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/live/state?latest=1") as resp:
            assert resp.status == 200
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/live/state?after_tick=17") as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown()

    assert calls == [(None, True), (17, False)]


def test_send_json_ignores_browser_refresh_disconnect():
    class DisconnectedWriter:
        def write(self, data):
            raise ConnectionAbortedError("browser navigated away")

    handler = server_mod.Handler.__new__(server_mod.Handler)
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    handler.wfile = DisconnectedWriter()
    handler.close_connection = False

    handler._send_json(200, {"history": []})

    assert handler.close_connection is True
def test_get_unknown_path_is_404():
    httpd, port = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()


def test_static_path_traversal_is_rejected():
    httpd, port = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/static/../flux_radar_server.py")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)
    finally:
        httpd.shutdown()


def test_api_engines_lists_the_registry_and_default():
    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/engines") as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert "gpt2" in payload["engines"]
            assert "qwen3-1.7b-base" in payload["engines"]
            assert payload["default"] in payload["engines"]
    finally:
        httpd.shutdown()


def test_api_run_surfaces_errors_as_json_not_a_crash(monkeypatch):
    class ExplodingBundle:
        def run(self, params):
            raise ValueError("simulated failure")

    monkeypatch.setattr(server_mod, "get_bundle", lambda params=None: ExplodingBundle())

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"seed": "x", "ticks": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 500
            payload = json.loads(e.read().decode("utf-8"))
            assert "simulated failure" in payload["error"]
    finally:
        httpd.shutdown()


def test_api_run_passes_the_full_request_params_to_get_bundle(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")

    seen = {}

    class FakeBundle:
        def run(self, params):
            return {"run_id": "x", "history": [{"tick": 0, "nodes": []}], "params": dict(params)}

    def fake_get_bundle(params=None):
        seen["params"] = params
        return FakeBundle()

    monkeypatch.setattr(server_mod, "get_bundle", fake_get_bundle)

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({
                "seed": "x", "ticks": 1, "engine": "qwen3-1.7b-base",
                "dictionary_size": 5000, "min_word_len": 4,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        assert seen["params"]["engine"] == "qwen3-1.7b-base"
        assert seen["params"]["dictionary_size"] == 5000
        assert seen["params"]["min_word_len"] == 4
    finally:
        httpd.shutdown()


def test_api_run_returns_bundle_result_as_json(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")

    class FakeBundle:
        def run(self, params):
            return {"run_id": params.get("seed"), "history": [{"tick": 0, "nodes": []}], "params": {}}

    monkeypatch.setattr(server_mod, "get_bundle", lambda params=None: FakeBundle())

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"seed": "hello", "ticks": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["run_id"] == "hello"
            assert payload["history"][0]["tick"] == 0
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# get_bundle()'s two-tier cache: engine handles vs. dictionary-param bundles
# ---------------------------------------------------------------------------

class _FakeEngineHandle:
    def __init__(self, name):
        self.name = name
        self.preloaded = False

    def preload(self):
        self.preloaded = True

    @property
    def tokenizer(self):
        return _FakeTokenizer()

    @property
    def model(self):
        class _FakeParam:
            def parameters(self):
                import torch
                return iter([torch.zeros(1)])
        return _FakeParam()


def _patch_engine_and_bundle_caches(monkeypatch):
    """Reset both module-level caches and stub out the expensive real work.

    load_engine is stubbed to hand back a cheap fake handle (counting
    calls, so tests can assert it's reused); ModelBundle itself is left
    real (it's cheap once tokenizer/model are fakes) so these tests
    exercise get_bundle()'s actual caching/key logic, not a mock of it.
    """
    monkeypatch.setattr(server_mod, "_engine_handles", {})
    monkeypatch.setattr(server_mod, "_engine_locks", {})
    monkeypatch.setattr(server_mod, "_bundles", {})
    load_calls = []

    def fake_load_engine(name):
        load_calls.append(name)
        return _FakeEngineHandle(name)

    monkeypatch.setattr(server_mod, "load_engine", fake_load_engine)
    monkeypatch.setattr(server_mod, "DictionaryTokenFilter", type(
        "FakeDictFilter", (), {
            "from_curated_wordlist": staticmethod(lambda *a, **k: type(
                "M", (), {"mask_as_list": lambda self, n: [True] * n}
            )())
        }
    ))
    monkeypatch.setattr(server_mod, "WordTrie", type(
        "FakeWordTrie", (), {"from_curated_wordlist": staticmethod(lambda *a, **k: object())}
    ))
    return load_calls


def test_get_bundle_reuses_the_engine_handle_across_dictionary_combinations(monkeypatch):
    load_calls = _patch_engine_and_bundle_caches(monkeypatch)

    b1 = server_mod.get_bundle({"engine": "gpt2", "dictionary_size": 1000})
    b2 = server_mod.get_bundle({"engine": "gpt2", "dictionary_size": 2000})

    assert load_calls == ["gpt2"]  # loaded once, not twice
    assert b1 is not b2  # different ModelBundle (different dictionary_size)
    assert b1._run_lock is b2._run_lock  # but sharing the same underlying engine's lock


def test_get_bundle_caches_the_exact_same_combination(monkeypatch):
    _patch_engine_and_bundle_caches(monkeypatch)

    b1 = server_mod.get_bundle({"engine": "gpt2", "dictionary_size": 1000})
    b2 = server_mod.get_bundle({"engine": "gpt2", "dictionary_size": 1000})

    assert b1 is b2


def test_get_bundle_dictionary_enabled_false_drops_the_dictionary_filter(monkeypatch):
    _patch_engine_and_bundle_caches(monkeypatch)

    bundle = server_mod.get_bundle({"engine": "gpt2", "dictionary_enabled": False})

    assert bundle.word_trie is None
    assert bundle.backward_word_trie is None
    # candidate_filter falls back to WritingTokenFilter alone, not the
    # (mocked-away) DictionaryTokenFilter/CombinedTokenFilter stack.
    from speaktome.core.writing_token_filter import WritingTokenFilter
    assert isinstance(bundle.candidate_filter, WritingTokenFilter)


def test_get_bundle_unknown_engine_raises_without_loading_anything(monkeypatch):
    load_calls = _patch_engine_and_bundle_caches(monkeypatch)

    try:
        server_mod.get_bundle({"engine": "not-a-real-engine"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not-a-real-engine" in str(e)
    assert load_calls == []


# ---------------------------------------------------------------------------
# _reset_everything() / POST /api/reset
# ---------------------------------------------------------------------------

def test_reset_everything_clears_engine_and_bundle_caches(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")
    monkeypatch.setattr(server_mod, "_live_session", None)
    _patch_engine_and_bundle_caches(monkeypatch)
    server_mod.get_bundle({"engine": "gpt2"})
    assert server_mod._engine_handles
    assert server_mod._bundles

    server_mod._reset_everything()

    assert server_mod._engine_handles == {}
    assert server_mod._engine_locks == {}
    assert server_mod._bundles == {}


def test_reset_everything_stops_the_live_session(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")
    monkeypatch.setattr(server_mod, "_bundles", {})
    monkeypatch.setattr(server_mod, "_engine_handles", {})
    monkeypatch.setattr(server_mod, "_engine_locks", {})

    class FakeSession:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    session = FakeSession()
    monkeypatch.setattr(server_mod, "_live_session", session)

    server_mod._reset_everything()

    assert session.stopped is True
    assert server_mod._live_session is None


def test_reset_everything_survives_a_missing_torch(monkeypatch):
    # torch is always importable in this repo's real environment, but
    # _reset_everything's torch.cuda cleanup is explicitly best-effort --
    # simulate it being unavailable and confirm reset still completes.
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")
    monkeypatch.setattr(server_mod, "_bundles", {})
    monkeypatch.setattr(server_mod, "_engine_handles", {})
    monkeypatch.setattr(server_mod, "_engine_locks", {})
    monkeypatch.setattr(server_mod, "_live_session", None)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    server_mod._reset_everything()  # must not raise


def test_reset_everything_clears_run_progress(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")
    monkeypatch.setattr(server_mod, "_bundles", {})
    monkeypatch.setattr(server_mod, "_engine_handles", {})
    monkeypatch.setattr(server_mod, "_engine_locks", {})
    monkeypatch.setattr(server_mod, "_live_session", None)
    server_mod._publish_run_progress(active=True, run_id="x", tick=3, total_ticks=6)

    server_mod._reset_everything()

    assert server_mod._run_progress == {"active": False}


# ---------------------------------------------------------------------------
# GET /api/run/progress
# ---------------------------------------------------------------------------

def test_api_run_progress_defaults_to_inactive_with_no_run_in_flight(monkeypatch):
    monkeypatch.setattr(server_mod, "_run_progress", {"active": False})
    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run/progress") as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload == {"active": False}
    finally:
        httpd.shutdown()


def test_api_run_progress_reflects_published_state(monkeypatch):
    monkeypatch.setattr(server_mod, "_run_progress", {"active": False})
    server_mod._publish_run_progress(
        active=True, run_id="my-run", seed_text="the ocean", total_ticks=10,
        tick=4, latest={"tick": 4, "nodes": [], "best_path": "x", "best_score": 0.5},
        mid_tick={
            "tick": 5, "phase": "model_inference", "detail": "scoring rows",
            "current": 2, "total": 4, "live_nodes": 19, "elapsed_seconds": 1.25,
        },
        error=None,
    )

    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run/progress") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["active"] is True
        assert payload["tick"] == 4
        assert payload["total_ticks"] == 10
        assert payload["latest"]["best_score"] == 0.5
        assert payload["mid_tick"]["phase"] == "model_inference"
        assert payload["mid_tick"]["current"] == 2
    finally:
        httpd.shutdown()


def test_publish_run_progress_merges_fields_without_clobbering_others():
    server_mod._run_progress.clear()
    server_mod._publish_run_progress(active=True, run_id="x", total_ticks=5, tick=0)
    server_mod._publish_run_progress(tick=1)  # a later tick's update, e.g.
    assert server_mod._run_progress["run_id"] == "x"  # untouched by the second call
    assert server_mod._run_progress["tick"] == 1


def test_api_reset_returns_200_and_clears_caches(monkeypatch):
    import tempfile
    from pathlib import Path
    monkeypatch.setattr(server_mod, "STATE_FILE", Path(tempfile.mkdtemp()) / "state.json")
    monkeypatch.setattr(server_mod, "_live_session", None)
    _patch_engine_and_bundle_caches(monkeypatch)
    server_mod.get_bundle({"engine": "gpt2"})

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/reset", method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["reset"] is True
        assert server_mod._bundles == {}
        assert server_mod._engine_handles == {}
    finally:
        httpd.shutdown()
