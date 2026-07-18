#!/usr/bin/env python3
"""Standalone local web server for live FluxGraph visualization.

Runs a real FluxGraph session on demand, driven by parameters submitted
from the browser (seed text, ticks, budget, branch, sampling alpha/beta,
engine choice, poetic booster, no-repeat-ngram size), and serves a
radar-style live view of it: concentric rings by generation depth,
forward and backward as two mirrored hemispheres, node size by pressure,
brightness by score, spring-clustered children. Root displacement (see
FluxGraph._reroot) can change which node is anchor at any time -- the
graph itself never loses or detaches anything; nodes that no longer read
as a clean two-hemisphere layout are just tagged "orthogonal" (see
FluxGraph.orthogonal_node_ids) so the frontend can choose not to force
them into the radar. The frontend lives in speaktome/flux_radar/index.html.

Deliberately stdlib-only (http.server, no Flask/FastAPI) so this needs no
new dependency -- see AGENTS_DO_NOT_PIP_MANUALLY.md.

The backing language model is picked per request via an "engine" param
(GET /api/engines lists what's registered -- see core.model_engine),
not hardcoded to GPT-2. One ModelBundle per engine is lazily loaded and
cached forever the first time that engine is used, so switching engines
across runs costs a one-time load, not a reload every request.

Besides the one-shot POST /api/run, there's a continuous mode (see
LiveSession): POST /api/live/start ticks a single FluxGraph forever in a
background thread until POST /api/live/stop, with GET /api/live/state
polled by the frontend for the latest snapshot. Only ever one graph
doing real work at a time -- starting a new live session stops any
existing one first -- and its tick history is a fixed-size rolling
window (oldest tick dropped as each new one lands), so an indefinitely
long session has bounded memory, not unbounded growth.

State survives a restart: the last-used params (GET /api/last_params)
and, if a live session is running, its full graph state are written to
flux_radar_state.json after every request and every single tick -- not
just on a graceful shutdown -- so killing the process mid-session loses
at most the in-flight tick, never previously-completed progress. On
startup a saved live session resumes ticking from exactly where it left
off (see LiveSession.resume); an explicit POST /api/live/stop clears it
instead, since that's an intentional end, not a kill.

Recovering from a stuck engine: POST /api/reset drops every cached
engine/model/bundle and stops any live session -- a real, useful "start
over" for the common case, but it can't rescue a request already wedged
inside a synchronous model call (nothing in Python can signal or cancel
that thread). For an engine that's genuinely hung and not responding at
all, run flux_radar_restart.py instead -- it force-kills whatever's on
this server's port and starts a fresh process. Safe at any time given
the autosave above.

Usage:
    python -m speaktome.flux_radar_server [--port 8877] [--engine gpt2] [--preload]
Then open http://127.0.0.1:8877/ in a browser. The first /api/run call
loads the engine and the dictionary/trie infrastructure (slow, one time,
and slower for a larger vocab than GPT-2's); --preload does that at
startup instead of on first request.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from tensors.torch_backend import PyTorchTensorOperations
from .core.model_engine import ENGINES, DEFAULT_ENGINE, load_engine
from .core.model_abstraction import PyTorchModelWrapper
from .core.writing_token_filter import WritingTokenFilter
from .core.token_filters import DictionaryTokenFilter, CombinedTokenFilter
from .core.implicit_backpath import ImplicitBackpathScorer
from .core.choice_policy import AlphaBetaPolicy
from .core.flux_graph import FluxGraph, FluxGraphConfig, FluxNode
from .core.poetic_attractor import PoeticAttractor
from .core.word_trie import WordTrie
from .core.noodle_explorer import Direction
# --- END HEADER ---

STATE_FILE = Path(__file__).parent / "flux_radar_state.json"


def _node_to_dict(node: FluxNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "tokens": node.tokens,
        "direction": node.direction.name if node.direction is not None else None,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "local_evidence": node.local_evidence,
        "children_ids": node.children_ids,
        "pressure": node.pressure,
        "low_pressure_ticks": node.low_pressure_ticks,
        "created_tick": node.created_tick,
        "burned": node.burned,
        "expanded": node.expanded,
        "cumulative_evidence": node.cumulative_evidence,
        "rollup_mean": node.rollup_mean,
        "subtree_auxin": node.subtree_auxin,
        "auxin_level": node.auxin_level,
    }


def _node_from_dict(d: Dict[str, Any]) -> FluxNode:
    direction = Direction[d["direction"]] if d["direction"] is not None else None
    return FluxNode(
        id=d["id"], tokens=list(d["tokens"]), direction=direction, parent_id=d["parent_id"],
        depth=d["depth"], local_evidence=d["local_evidence"], children_ids=list(d["children_ids"]),
        pressure=d["pressure"], low_pressure_ticks=d["low_pressure_ticks"], created_tick=d["created_tick"],
        burned=d["burned"], expanded=d["expanded"], cumulative_evidence=d["cumulative_evidence"],
        rollup_mean=d["rollup_mean"], subtree_auxin=d["subtree_auxin"], auxin_level=d["auxin_level"],
    )

STATIC_DIR = Path(__file__).parent / "flux_radar"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
}


class ModelBundle:
    """The dictionary/trie filtering infrastructure for one engine + one set of
    vocabulary params, built on top of an already-loaded engine handle.

    Building the curated dictionary and both WordTrie instances is a
    real, multi-second cost that scales with dictionary_size (see
    demo_flux_graph.py) -- kept deliberately separate from the (far more
    expensive: can mean pulling gigabytes from HuggingFace) model/
    tokenizer load, so that live-editing dictionary_size/min_word_len/
    max_word_len/dictionary_enabled from the UI only ever pays for a
    dictionary rebuild, never a model reload. See the module-level
    get_bundle() for the cache keyed on the full (engine, dictionary
    params) combination, built on top of _get_engine()'s separate
    per-engine-name cache.
    """

    def __init__(
        self,
        engine_handle: Any,
        run_lock: threading.Lock,
        engine_name: str,
        dictionary_size: int = 20000,
        min_word_len: int = 2,
        max_word_len: Optional[int] = None,
        dictionary_enabled: bool = True,
    ):
        self.engine_name = engine_name
        self.dictionary_size = dictionary_size
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.dictionary_enabled = dictionary_enabled

        self.tokenizer = engine_handle.tokenizer
        self.model = engine_handle.model
        self.device = next(self.model.parameters()).device
        self.wrapper = PyTorchModelWrapper(self.model)

        writing_filter = WritingTokenFilter(self.tokenizer)
        if dictionary_enabled:
            dictionary_filter = DictionaryTokenFilter.from_curated_wordlist(
                self.tokenizer, n=dictionary_size, min_word_len=min_word_len, max_word_len=max_word_len,
            )
            self.candidate_filter = CombinedTokenFilter([writing_filter, dictionary_filter])
            self.word_trie = WordTrie.from_curated_wordlist(
                n=dictionary_size, min_word_len=min_word_len, max_word_len=max_word_len,
            )
            self.backward_word_trie = WordTrie.from_curated_wordlist(
                n=dictionary_size, min_word_len=min_word_len, max_word_len=max_word_len, reverse=True,
            )
        else:
            # Matches pre-trie-gating behavior exactly: candidates are
            # only screened for "looks like writing" (WritingTokenFilter),
            # and word growth is unconstrained (single BPE token per edge,
            # no dictionary validation at all).
            self.candidate_filter = writing_filter
            self.word_trie = None
            self.backward_word_trie = None
        self.candidate_filter.mask_as_list(self.tokenizer.vocab_size)

        self.ops = PyTorchTensorOperations(track_time=False)
        # Shared with every other ModelBundle for this same engine name
        # (see _get_engine) -- different dictionary-param combinations of
        # the same engine still share one underlying model, and a real
        # FluxGraph run does GPU work, so they still need to serialize
        # against each other, not just against themselves.
        self._run_lock = run_lock

    def _build_empty_graph(self, params: Dict[str, Any]) -> Tuple[FluxGraph, Dict[str, Any]]:
        """Everything build_graph() does except seeding -- shared by fresh runs and state-restore.

        A restored (previously-persisted) session needs this same
        config/choice_policy/graph construction but must NOT call
        graph.seed() (that creates a brand-new anchor with no history);
        it installs previously-saved node state instead (see
        FluxGraph.restore_state, used by LiveSession.resume). Every
        tunable FluxGraphConfig field that's safe to expose (i.e. not
        internal plumbing like verbose or backward_left_context) is
        settable here from request params, each with the same default
        FluxGraphConfig itself uses when the param is missing.
        """
        backpath = ImplicitBackpathScorer(self.wrapper, self.tokenizer, writing_filter=self.candidate_filter)

        def _zero_means_none(key: str, default: int) -> Optional[int]:
            raw = params.get(key, default)
            value = int(raw) if raw else None
            return None if value is not None and value <= 0 else value

        no_repeat_ngram_size = _zero_means_none("no_repeat_ngram_size", 3)
        population_target = _zero_means_none("population_target", 0)
        max_expand_elements = _zero_means_none("max_expand_elements", 400_000_000)
        forward_branch_factor = _zero_means_none("forward_branch_factor", 0)
        backward_branch_factor = _zero_means_none("backward_branch_factor", 0)
        forward_hot_loop_depth = _zero_means_none("forward_hot_loop_depth", 0)
        backward_hot_loop_depth = _zero_means_none("backward_hot_loop_depth", 0)
        forward_budget_per_tick = _zero_means_none("forward_budget_per_tick", 0)
        backward_budget_per_tick = _zero_means_none("backward_budget_per_tick", 0)

        poetic_scale = float(params.get("poetic_scale", 1.0))
        poetic_attractor = PoeticAttractor() if params.get("poetic_enabled", False) else None

        config = FluxGraphConfig(
            found_bonus=float(params.get("found_bonus", 0.05)),
            damping=float(params.get("damping", 0.5)),
            starvation_floor=float(params.get("starvation_floor", 0.08)),
            overpressure_ceiling=float(params.get("overpressure_ceiling", 0.0)),
            burn_after_ticks=int(params.get("burn_after_ticks", 3)),
            anchor_can_decay=bool(params.get("anchor_can_decay", False)),
            compute_budget_per_tick=int(params.get("budget", 3)),
            branch_factor=int(params.get("branch", 3)),
            hot_loop_depth=int(params.get("hot_loop_depth", 1)),
            verbose=False,
            max_context_tokens=int(params.get("max_context_tokens", 64)),
            max_relaxation_iterations=int(params.get("max_relaxation_iterations", 25)),
            relaxation_tolerance=float(params.get("relaxation_tolerance", 1e-4)),
            rollup_weight=float(params.get("rollup_weight", 0.3)),
            exploration_constant=float(params.get("exploration_constant", 0.05)),
            balance_weight=float(params.get("balance_weight", 0.0)),
            backward_left_context=[self.tokenizer.eos_token_id],
            no_repeat_ngram_size=no_repeat_ngram_size,
            expand_batch_chunk_size=int(params.get("expand_batch_chunk_size", 2048)),
            max_expand_elements=max_expand_elements,
            word_trie=self.word_trie,
            backward_word_trie=self.backward_word_trie,
            max_subword_steps=int(params.get("max_subword_steps", 8)),
            poetic_attractor=poetic_attractor,
            poetic_scale=poetic_scale,
            poetic_shortlist_k=int(params.get("poetic_shortlist_k", 20)),
            auxin_suppression=float(params.get("auxin_suppression", 0.0)),
            auxin_decay=float(params.get("auxin_decay", 0.6)),
            head_pressure_coefficient=float(params.get("head_pressure_coefficient", 0.0)),
            decay_rate=float(params.get("decay_rate", 0.0)),
            population_target=population_target,
            shared_token_pressure_enabled=bool(params.get("shared_token_pressure_enabled", True)),
            return_conductance_scale=float(params.get("return_conductance_scale", 1.0)),
            forward_branch_factor=forward_branch_factor,
            backward_branch_factor=backward_branch_factor,
            forward_hot_loop_depth=forward_hot_loop_depth,
            backward_hot_loop_depth=backward_hot_loop_depth,
            forward_budget_per_tick=forward_budget_per_tick,
            backward_budget_per_tick=backward_budget_per_tick,
            forward_selection_mode=str(params.get("forward_selection_mode", "topk")),
            backward_selection_mode=str(params.get("backward_selection_mode", "topk")),
            forward_top_p=float(params.get("forward_top_p", 0.9)),
            backward_top_p=float(params.get("backward_top_p", 0.9)),
            top_p_shortlist_ceiling=int(params.get("top_p_shortlist_ceiling", 40)),
        )
        choice_policy = AlphaBetaPolicy(
            alpha=float(params.get("alpha", 0.8)),
            beta=float(params.get("beta", 1.0)),
            seed=params.get("rng_seed"),
        )
        graph = FluxGraph(self.wrapper, backpath, choice_policy, self.ops, config=config, device=self.device)

        resolved_params = {
            "engine": self.engine_name,
            "dictionary_size": self.dictionary_size,
            "min_word_len": self.min_word_len,
            "max_word_len": self.max_word_len or 0,
            "dictionary_enabled": self.dictionary_enabled,
            "budget": config.compute_budget_per_tick,
            "branch": config.branch_factor,
            "hot_loop_depth": config.hot_loop_depth,
            "alpha": choice_policy.alpha,
            "beta": choice_policy.beta,
            "anchor_can_decay": config.anchor_can_decay,
            "no_repeat_ngram_size": config.no_repeat_ngram_size or 0,
            "poetic_enabled": config.poetic_attractor is not None,
            "poetic_scale": config.poetic_scale,
            "poetic_shortlist_k": config.poetic_shortlist_k,
            "found_bonus": config.found_bonus,
            "damping": config.damping,
            "starvation_floor": config.starvation_floor,
            "overpressure_ceiling": config.overpressure_ceiling,
            "burn_after_ticks": config.burn_after_ticks,
            "max_context_tokens": config.max_context_tokens,
            "max_relaxation_iterations": config.max_relaxation_iterations,
            "relaxation_tolerance": config.relaxation_tolerance,
            "rollup_weight": config.rollup_weight,
            "exploration_constant": config.exploration_constant,
            "balance_weight": config.balance_weight,
            "expand_batch_chunk_size": config.expand_batch_chunk_size,
            "max_expand_elements": config.max_expand_elements or 0,
            "max_subword_steps": config.max_subword_steps,
            "auxin_suppression": config.auxin_suppression,
            "auxin_decay": config.auxin_decay,
            "head_pressure_coefficient": config.head_pressure_coefficient,
            "decay_rate": config.decay_rate,
            "population_target": config.population_target or 0,
            "shared_token_pressure_enabled": config.shared_token_pressure_enabled,
            "return_conductance_scale": config.return_conductance_scale,
            "forward_branch_factor": config.forward_branch_factor or 0,
            "backward_branch_factor": config.backward_branch_factor or 0,
            "forward_hot_loop_depth": config.forward_hot_loop_depth or 0,
            "backward_hot_loop_depth": config.backward_hot_loop_depth or 0,
            "forward_budget_per_tick": config.forward_budget_per_tick or 0,
            "backward_budget_per_tick": config.backward_budget_per_tick or 0,
            "forward_selection_mode": config.forward_selection_mode,
            "backward_selection_mode": config.backward_selection_mode,
            "forward_top_p": config.forward_top_p,
            "backward_top_p": config.backward_top_p,
            "top_p_shortlist_ceiling": config.top_p_shortlist_ceiling,
        }
        return graph, resolved_params

    def build_graph(self, params: Dict[str, Any]) -> Tuple[FluxGraph, str, Dict[str, Any]]:
        """Construct a fresh, seeded FluxGraph from request params.

        Shared by both a one-shot run() and a continuously-ticking
        LiveSession -- the two only differ in how many times and how
        often they call graph.tick(), not in how the graph itself gets
        built.
        """
        graph, resolved_params = self._build_empty_graph(params)
        seed_text = str(params.get("seed") or "the ocean")
        seed_ids = self.tokenizer.encode(seed_text)
        graph.seed(seed_ids)
        return graph, seed_text, resolved_params

    def snapshot_graph(self, graph: FluxGraph) -> Dict[str, Any]:
        """One tick's worth of display data for ``graph`` in its current state.

        Every node is tagged "orthogonal": true/false (see
        FluxGraph.orthogonal_node_ids) purely as a display hint -- the
        frontend can choose not to force those into the two-hemisphere
        layout, but they're still live data, still eligible to become
        anchor themselves later. Orthogonal nodes additionally carry
        "network_root": the id of the shallowest node in their own
        disconnected-feeling branch (see FluxGraph.orthogonal_network_roots)
        -- the frontend groups by this to give each branch its own pair of
        pie wedges instead of lumping every orthogonal node together.

        "display_radius" is what the frontend actually positions nodes
        with. For an ordinary (non-orthogonal) node it's just depth --
        the anchor is the active seed, the crown where the live graph is
        actually growing, so anchor-relative depth is exactly the right
        distance to show. A node in an orthogonal "cousin" network isn't
        part of the seed's own bowtie though -- it's part of its own
        separate one, centered on its own network_root (the exact point
        where that lineage's direction first broke from its parent's, and
        always a strict ancestor of every other member of its network,
        since orthogonal status only ever propagates downward). Using
        plain depth for those would measure distance from the *current*
        anchor instead, dragging every cousin network's ring radius
        around on every re-root even though nothing in that cousin
        network moved.

        Also runs the graph auditor and folds its result into "edges" --
        every real parent/child connection currently alive gets its own
        {"from", "to", "total", "count", "avg"} entry (see FluxGraph.
        audit_edge_influence): "avg" is that edge's mean causal-path
        quality across every traversal crossing it, "total" is the
        summed quality across all of them -- the discrete integral of
        influence flowing through that one connection. This runs on
        every snapshot (every tick, one-shot run or live), not behind
        the graph_auditor_enabled tick-time flag, so edge appearance is
        always current for whichever tick is actually being displayed,
        never something a caller has to separately ask for.
        """
        orthogonal = graph.orthogonal_node_ids()
        network_roots = graph.orthogonal_network_roots()
        nodes = []
        for nid, n in graph.nodes.items():
            # A node's own tokens are empty only while it's the current
            # anchor -- anchor_tokens is whatever it represents (the
            # original seed, or a former anchor's own word span once
            # demoted).
            text = self.tokenizer.decode(n.tokens) if n.tokens else self.tokenizer.decode(graph.anchor_tokens)
            direction = None if n.direction is None else (
                "forward" if n.direction is Direction.FORWARD else "backward"
            )
            root_id = network_roots.get(nid)
            display_radius = n.depth if root_id is None else n.depth - graph.nodes[root_id].depth
            nodes.append({
                "id": nid,
                "parent_id": n.parent_id,
                "direction": direction,
                "height": n.height,
                "depth": n.depth,
                "display_radius": display_radius,
                "pressure": round(n.pressure, 4),
                "local_evidence": round(n.local_evidence, 4),
                "path_mean": round(n.path_mean, 4),
                "burned": n.burned,
                "created_tick": n.created_tick,
                "text": text,
                "orthogonal": nid in orthogonal,
                "network_root": network_roots.get(nid),
            })
        influence = graph.audit_edge_influence()
        edges = [
            {
                "from": a,
                "to": b,
                "total": round(v["total"], 6),
                "count": v["count"],
                "avg": round(v["total"] / v["count"], 6),
            }
            for (a, b), v in influence.items()
        ]
        tokens, score = graph.best_path()
        return {
            "nodes": nodes,
            "edges": edges,
            "traversal_count": len(graph.traversals),
            "best_path": self.tokenizer.decode(tokens),
            "best_score": round(score, 4),
        }

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run one fresh FluxGraph session end to end and return its tick history.

        One graph, one topology, one continuous line of real model calls,
        for the whole run -- root displacement (see FluxGraph._reroot) only
        ever changes which node is treated as anchor. Nothing is ever
        removed from the graph, so every node in every snapshot is tagged
        "orthogonal": true/false, purely as a display hint.

        Publishes to the module-level _run_progress after every tick (see
        _publish_run_progress) -- this call still blocks and returns the
        complete history at the end exactly as before, but a concurrent
        GET /api/run/progress can now see live stats while it's building
        instead of nothing until the whole thing finishes.
        """
        with self._run_lock:
            graph, seed_text, resolved_params = self.build_graph(params)
            ticks = max(0, int(params.get("ticks", 6)))
            run_id = params.get("run_id") or seed_text
            _publish_run_progress(
                active=True, run_id=run_id, seed_text=seed_text,
                total_ticks=ticks, tick=0, latest=None, error=None,
            )

            try:
                history: List[Dict[str, Any]] = []
                graph.spawn_first_children()
                snapshot = {"tick": 0, **self.snapshot_graph(graph)}
                history.append(snapshot)
                _publish_run_progress(tick=0, latest=snapshot)

                for t in range(1, ticks + 1):
                    graph.tick()
                    snapshot = {"tick": t, **self.snapshot_graph(graph)}
                    history.append(snapshot)
                    _publish_run_progress(tick=t, latest=snapshot)
            except Exception as e:  # noqa: BLE001 -- surfaced to the poller, then re-raised for _handle_run
                _publish_run_progress(active=False, error=str(e))
                raise
            else:
                _publish_run_progress(active=False)

            return {
                "run_id": run_id,
                "seed_text": seed_text,
                "params": {**resolved_params, "ticks": ticks},
                "history": history,
            }


class LiveSession:
    """Ticks one FluxGraph continuously in a background thread until stopped.

    Only ever one graph, doing exactly what a normal tick() call always
    safely does -- this is not the FluxForest mistake (no new graphs are
    ever spun up). "Continuous" is kept bounded on both axes that matter:
    a fixed-size rolling window of tick snapshots (a deque with maxlen --
    appending past capacity silently drops the oldest, i.e. "burns the
    oldest tick" for free) so memory never grows across an indefinitely
    long session, and a floor on the tick interval so a mistaken near-zero
    value can't spin the GPU as fast as physically possible.
    """

    MIN_INTERVAL_S = 0.2
    MAX_WINDOW = 200

    def __init__(self, bundle: "ModelBundle", params: Dict[str, Any]):
        self.bundle = bundle
        self.graph, self.seed_text, resolved_params = bundle.build_graph(params)
        self.window = max(1, min(int(params.get("window", 30)), self.MAX_WINDOW))
        interval_s = float(params.get("interval_ms", 800)) / 1000.0
        self.interval = max(self.MIN_INTERVAL_S, interval_s)
        self.params = {**resolved_params, "window": self.window, "interval_ms": self.interval * 1000.0}

        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.window)
        self.tick_index = 0
        self.error: Optional[str] = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="flux-live-tick")

    @classmethod
    def resume(cls, bundle: "ModelBundle", saved: Dict[str, Any]) -> "LiveSession":
        """Reconstruct a LiveSession from to_dict() output -- e.g. after a server restart.

        Rebuilds the graph shell the same way a fresh LiveSession would
        (same params, so the same config/choice_policy), but installs the
        previously-saved node state (FluxGraph.restore_state) instead of
        seeding a brand-new anchor -- the graph picks up exactly where it
        left off, not from scratch.
        """
        session = cls.__new__(cls)
        session.bundle = bundle
        params = saved["params"]
        graph, _ = bundle._build_empty_graph(params)
        saved_graph = saved["graph"]
        nodes = {int(nid): _node_from_dict(nd) for nid, nd in saved_graph["nodes"].items()}
        graph.restore_state(
            nodes, saved_graph["anchor_id"], saved_graph["anchor_tokens"], saved_graph["tick_count"],
            anchor_local_evidence=saved_graph.get("anchor_local_evidence", 0.0),
        )
        session.graph = graph
        session.seed_text = saved["seed_text"]
        session.window = max(1, min(int(params.get("window", 30)), cls.MAX_WINDOW))
        interval_s = float(params.get("interval_ms", 800)) / 1000.0
        session.interval = max(cls.MIN_INTERVAL_S, interval_s)
        session.params = params
        session.history = deque(saved["history"], maxlen=session.window)
        session.tick_index = saved["tick_index"]
        session.error = None
        session._state_lock = threading.Lock()
        session._stop_event = threading.Event()
        session._thread = threading.Thread(target=session._run_loop, daemon=True, name="flux-live-tick")
        return session

    def to_dict(self) -> Dict[str, Any]:
        """Full snapshot for persistence -- resume() rebuilds this session exactly from it.

        Only safe to call when no concurrent tick is mutating self.graph
        -- true from within _run_loop itself right after a tick completes
        and before the next one starts (the only place this is actually
        called from), and true before the background thread has started
        at all.
        """
        graph = self.graph
        return {
            "seed_text": self.seed_text,
            "params": self.params,
            "tick_index": self.tick_index,
            "history": list(self.history),
            "graph": {
                "anchor_id": graph.anchor_id,
                "anchor_tokens": list(graph.anchor_tokens),
                "anchor_local_evidence": graph.anchor_local_evidence,
                "tick_count": graph.tick_count,
                "nodes": {str(nid): _node_to_dict(n) for nid, n in graph.nodes.items()},
            },
        }

    def start(self) -> None:
        self.graph.spawn_first_children()
        with self._state_lock:
            self.history.append({"tick": 0, **self.bundle.snapshot_graph(self.graph)})
        self._thread.start()

    def start_resumed(self) -> None:
        """Like start(), but the graph is already fully grown from saved state -- just resume ticking."""
        self._thread.start()

    def _run_loop(self) -> None:
        try:
            # wait() first: the initial tick-0 snapshot from start() is
            # already the freshest state, no need to immediately tick again.
            while not self._stop_event.wait(self.interval):
                with self.bundle._run_lock:
                    self.graph.tick()
                self.tick_index += 1
                with self._state_lock:
                    self.history.append({"tick": self.tick_index, **self.bundle.snapshot_graph(self.graph)})
                _autosave()
        except Exception as e:  # noqa: BLE001 -- surfaced via snapshot_state(), not swallowed
            with self._state_lock:
                self.error = str(e)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def snapshot_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "seed_text": self.seed_text,
                "params": self.params,
                "tick_index": self.tick_index,
                "running": self._thread.is_alive() and not self._stop_event.is_set(),
                "error": self.error,
                "history": list(self.history),
            }


_live_session: Optional[LiveSession] = None
_live_session_lock = threading.Lock()


def _stop_live_session_locked() -> None:
    """Caller must hold _live_session_lock."""
    global _live_session
    if _live_session is not None:
        _live_session.stop()
        _live_session = None


_last_params: Optional[Dict[str, Any]] = None
_state_file_lock = threading.Lock()


def _write_state_file(state: Dict[str, Any]) -> None:
    """Atomic write (temp file + os.replace) so a kill mid-write can't leave a corrupt file behind."""
    with _state_file_lock:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, STATE_FILE)


def _read_state_file() -> Dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        with _state_file_lock:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 -- a bad/partial state file should never block startup
        print(f"[flux-radar] warning: could not read saved state ({e}); starting fresh.")
        return {}


def _autosave() -> None:
    """Write current last_params + (if any) live session state to disk.

    Called after every /api/run, every /api/live/start, and every single
    live tick -- not just on a graceful shutdown -- so a kill at any
    point loses at most the in-flight request, never previously-completed
    progress.
    """
    state: Dict[str, Any] = {"last_params": _last_params}
    with _live_session_lock:
        session = _live_session
    if session is not None:
        state["live_session"] = session.to_dict()
    _write_state_file(state)


def _clear_persisted_live_session() -> None:
    """Drop just the live_session half of the saved state (keeps last_params).

    Called on an explicit POST /api/live/stop -- that's an intentional
    end, not a kill, so a restart shouldn't resurrect it.
    """
    state = _read_state_file()
    state.pop("live_session", None)
    _write_state_file(state)


# Two independent caches, deliberately kept separate: engine handles
# (tokenizer+model, the expensive one -- can mean pulling gigabytes from
# HuggingFace) keyed only by engine name, and ModelBundles (dictionary/
# trie infra, cheap by comparison but still real work) keyed by the full
# (engine, dictionary params) combination. A ModelBundle for a new
# dictionary combination reuses an already-loaded engine handle rather
# than reloading the model -- see ModelBundle.__init__.
_engine_handles: Dict[str, Any] = {}
_engine_locks: Dict[str, threading.Lock] = {}
_engines_lock = threading.Lock()

_bundles: Dict[Tuple[str, int, int, Optional[int], bool], ModelBundle] = {}
_bundles_lock = threading.Lock()
_default_engine: str = DEFAULT_ENGINE

# One-shot POST /api/run blocks until every tick finishes, which for a
# real model + many ticks can be a long, silent wait -- this is the same
# tick-by-tick snapshot LiveSession already publishes for continuous mode,
# just for whichever one-shot run is currently in flight, so the frontend
# can poll GET /api/run/progress and show live stats while a run builds
# instead of staring at a blank loading spinner. Module-level rather than
# per-ModelBundle since only one run is ever meaningfully "in flight" from
# the UI's perspective regardless of which engine/bundle is doing it, and
# ModelBundle.run() already serializes against its engine's _run_lock, so
# there's never real concurrent-write contention here in practice.
_run_progress: Dict[str, Any] = {"active": False}
_run_progress_lock = threading.Lock()


def _publish_run_progress(**fields: Any) -> None:
    with _run_progress_lock:
        _run_progress.update(fields)

DEFAULT_DICTIONARY_SIZE = 20000
DEFAULT_MIN_WORD_LEN = 2


def set_engine(engine: str) -> None:
    """Choose the default engine get_bundle() falls back to when a request doesn't name one.

    Must be called before the first request -- changing it afterward
    wouldn't un-load anything already cached, just quietly stop being
    the default, which is confusing enough to just disallow outright.
    """
    global _default_engine
    if _engine_handles:
        raise RuntimeError("an engine has already been loaded; set_engine() must be called before the first request")
    _default_engine = engine


def _get_engine(name: str) -> Tuple[Any, threading.Lock]:
    """Return the (lazily loaded, cached forever) engine handle and its shared run-lock.

    One handle per engine name, regardless of how many different
    dictionary-param ModelBundles end up built on top of it. The lock is
    shared the same way: every ModelBundle for this engine name does its
    real GPU work through the same underlying model, so they all need to
    serialize against each other, not just against themselves.
    """
    with _engines_lock:
        if name not in _engine_handles:
            print(f"[flux-radar] loading {name!r} model+tokenizer (first use of this engine only)...")
            handle = load_engine(name)
            handle.preload()
            _engine_handles[name] = handle
            _engine_locks[name] = threading.Lock()
            print(f"[flux-radar] {name!r} model ready.")
    return _engine_handles[name], _engine_locks[name]


def get_bundle(params: Optional[Dict[str, Any]] = None) -> ModelBundle:
    """Return the (lazily loaded, cached) ModelBundle for the given request params.

    Reads engine/dictionary_size/min_word_len/max_word_len/
    dictionary_enabled out of ``params`` (each falling back to the same
    default as ModelBundle/FluxGraphConfig itself); everything else in
    ``params`` is irrelevant here (see ModelBundle.build_graph instead).
    One ModelBundle per distinct combination, kept forever once built --
    live-editing any of these from the UI costs a one-time rebuild the
    first time that exact combination is used, not a rebuild every
    request. Not hard-bounded: a user free-editing dictionary_size could
    in principle accumulate many cached combinations over a long
    session, but each one is modest, CPU-side (never GPU memory), so this
    is deliberately not treated as a resource-safety concern the way
    engine/model loading is.
    """
    params = params or {}
    engine_name = params.get("engine") or _default_engine
    if engine_name not in ENGINES:
        raise ValueError(f"Unknown engine {engine_name!r}. Available engines: {sorted(ENGINES)}")

    dictionary_size = int(params.get("dictionary_size", DEFAULT_DICTIONARY_SIZE))
    min_word_len = int(params.get("min_word_len", DEFAULT_MIN_WORD_LEN))
    max_word_len_raw = params.get("max_word_len", 0)
    max_word_len = int(max_word_len_raw) if max_word_len_raw else None
    dictionary_enabled = bool(params.get("dictionary_enabled", True))

    key = (engine_name, dictionary_size, min_word_len, max_word_len, dictionary_enabled)
    with _bundles_lock:
        if key not in _bundles:
            engine_handle, run_lock = _get_engine(engine_name)
            print(f"[flux-radar] building dictionary/trie for {key!r} (first use of this combination)...")
            _bundles[key] = ModelBundle(
                engine_handle, run_lock, engine_name,
                dictionary_size=dictionary_size, min_word_len=min_word_len,
                max_word_len=max_word_len, dictionary_enabled=dictionary_enabled,
            )
            print(f"[flux-radar] {key!r} ready.")
    return _bundles[key]


def _reset_everything() -> None:
    """Stop any live session and drop every cached engine/bundle -- a clean software restart.

    This is the fast path for "start over," not a fix for a genuine
    hang: a request truly wedged inside a synchronous model call (a
    stuck CUDA kernel, say) is a live thread this process cannot signal,
    cancel, or otherwise interrupt -- Python has no mechanism for that,
    and this function doesn't pretend to. What it *does* guarantee: any
    NEW request made after this returns gets entirely fresh engine/lock
    objects, so it is never blocked by whatever an old, still-stuck
    thread happens to be holding, even though that old thread keeps
    running (and keeps its own GPU memory allocated) until the process
    itself exits. For a request that's actually hung and not responding
    at all, see flux_radar_restart.py instead -- killing and restarting
    the OS process is the only fully reliable fix, and is safe to do at
    any time given autosaved state (see _autosave).
    """
    with _live_session_lock:
        _stop_live_session_locked()
    _clear_persisted_live_session()
    with _bundles_lock:
        _bundles.clear()
    with _engines_lock:
        _engine_handles.clear()
        _engine_locks.clear()
    with _run_progress_lock:
        _run_progress.clear()
        _run_progress["active"] = False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- best-effort; absence of torch/CUDA is not an error here
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print("[flux-radar]", fmt % args)

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self._serve_file(STATIC_DIR / "index.html")
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):]
            self._serve_file(STATIC_DIR / rel)
        elif self.path == "/api/live/state":
            self._handle_live_state()
        elif self.path == "/api/run/progress":
            self._handle_run_progress()
        elif self.path == "/api/engines":
            self._send_json(200, {"engines": sorted(ENGINES), "default": _default_engine})
        elif self.path == "/api/last_params":
            self._send_json(200, {"params": _last_params})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            self._handle_run()
        elif self.path == "/api/live/start":
            self._handle_live_start()
        elif self.path == "/api/live/stop":
            self._handle_live_stop()
        elif self.path == "/api/live/physics":
            self._handle_live_physics()
        elif self.path == "/api/reset":
            self._handle_reset()
        else:
            self.send_error(404)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        return json.loads(body) if body else {}

    def _handle_run(self) -> None:
        global _last_params
        try:
            params = self._read_json_body()
            bundle = get_bundle(params)
            result = bundle.run(params)
            _last_params = result["params"]
            _autosave()
            self._send_json(200, result)
        except Exception as e:  # noqa: BLE001 -- surfaced to the browser, not swallowed
            self._send_json(500, {"error": str(e)})

    def _handle_live_start(self) -> None:
        global _live_session, _last_params
        try:
            params = self._read_json_body()
            bundle = get_bundle(params)
            with _live_session_lock:
                _stop_live_session_locked()
                session = LiveSession(bundle, params)
                session.start()
                _live_session = session
                state = session.snapshot_state()
            _last_params = state["params"]
            _autosave()
            self._send_json(200, state)
        except Exception as e:  # noqa: BLE001 -- surfaced to the browser, not swallowed
            self._send_json(500, {"error": str(e)})

    def _handle_live_stop(self) -> None:
        with _live_session_lock:
            _stop_live_session_locked()
        _clear_persisted_live_session()
        self._send_json(200, {"stopped": True})

    def _handle_reset(self) -> None:
        try:
            _reset_everything()
            self._send_json(200, {"reset": True})
        except Exception as e:  # noqa: BLE001 -- surfaced to the browser, not swallowed
            self._send_json(500, {"error": str(e)})

    def _handle_live_state(self) -> None:
        with _live_session_lock:
            session = _live_session
        if session is None:
            self._send_json(404, {"error": "no live session running"})
            return
        self._send_json(200, session.snapshot_state())

    def _handle_live_physics(self) -> None:
        """A foreign physics domain (the browser's interface sim) publishing
        its latest per-tick observations for the live graph.

        Doesn't take the engine's _run_lock: absorb_external_physics is a
        single atomic reference swap (see its docstring), so blocking this
        request behind a multi-second GPU tick would buy nothing -- the
        payload just lands and whichever tick runs next reads it.
        """
        with _live_session_lock:
            session = _live_session
        if session is None:
            self._send_json(404, {"error": "no live session running"})
            return
        try:
            payload = self._read_json_body()
            domain = str(payload.pop("domain", "client"))
            ring = payload.get("ring_proximity")
            if isinstance(ring, dict):
                # JSON object keys are always strings; node ids are ints.
                payload["ring_proximity"] = {int(k): float(v) for k, v in ring.items()}
            session.graph.absorb_external_physics(domain, payload)
            self._send_json(200, {"absorbed": domain})
        except Exception as e:  # noqa: BLE001 -- surfaced to the browser, not swallowed
            self._send_json(500, {"error": str(e)})

    def _handle_run_progress(self) -> None:
        with _run_progress_lock:
            payload = dict(_run_progress)
        self._send_json(200, payload)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        if STATIC_DIR.resolve() not in resolved.parents and resolved != STATIC_DIR.resolve():
            self.send_error(403)
            return
        if not resolved.is_file():
            self.send_error(404)
            return
        data = resolved.read_bytes()
        content_type = _CONTENT_TYPES.get(resolved.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _resume_saved_state() -> None:
    """Load flux_radar_state.json (if any) and pick up where the last run left off.

    Always restores last_params (harmless even if stale -- it's just
    form pre-fill). Resuming a saved live session is best-effort: a bad
    or incompatible save (edited by hand, from an engine that's no
    longer registered, a config field that no longer exists, ...) should
    never block startup -- log a warning and start with no live session
    instead of crashing.
    """
    global _last_params, _live_session
    saved = _read_state_file()
    _last_params = saved.get("last_params")

    live_saved = saved.get("live_session")
    if live_saved is None:
        return
    try:
        bundle = get_bundle(live_saved["params"])
        session = LiveSession.resume(bundle, live_saved)
        session.start_resumed()
        _live_session = session
        print(f"[flux-radar] resumed live session from saved state (tick {session.tick_index}).")
    except Exception as e:  # noqa: BLE001 -- a bad save must never block startup
        print(f"[flux-radar] warning: could not resume saved live session ({e}); starting without one.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument(
        "--engine", default=DEFAULT_ENGINE, choices=sorted(ENGINES),
        help=f"which model engine to load (default: {DEFAULT_ENGINE!r})",
    )
    parser.add_argument(
        "--preload", action="store_true",
        help="load the engine and the dictionary/trie infrastructure at startup instead of on first request",
    )
    args = parser.parse_args()

    set_engine(args.engine)
    if args.preload:
        get_bundle()

    _resume_saved_state()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[flux-radar] serving http://127.0.0.1:{args.port}/ (engine: {args.engine})")
    if not args.preload:
        print("[flux-radar] first run will take a while (loading the engine + dictionary). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with _live_session_lock:
            _stop_live_session_locked()
        server.server_close()


if __name__ == "__main__":
    main()
