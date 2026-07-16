#!/usr/bin/env python3
"""Standalone local web server for live FluxGraph visualization.

Runs a real FluxGraph session on demand, driven by parameters submitted
from the browser (seed text, ticks, budget, branch, sampling alpha/beta),
and serves a radar-style live view of it: concentric rings by generation
depth, forward and backward as two mirrored hemispheres, node size by
pressure, brightness by score, spring-clustered children. Root
displacement (see FluxGraph._reroot) can change which node is anchor at
any time -- the graph itself never loses or detaches anything; nodes
that no longer read as a clean two-hemisphere layout are just tagged
"orthogonal" (see FluxGraph.orthogonal_node_ids) so the frontend can
choose not to force them into the radar. The frontend lives in
speaktome/flux_radar/index.html.

Deliberately stdlib-only (http.server, no Flask/FastAPI) so this needs no
new dependency -- see AGENTS_DO_NOT_PIP_MANUALLY.md.

Usage:
    python -m speaktome.flux_radar_server [--port 8877] [--preload]
Then open http://127.0.0.1:8877/ in a browser. The first /api/run call
loads GPT-2 and the dictionary/trie infrastructure (slow, one time);
--preload does that at startup instead of on first request.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from tensors.torch_backend import PyTorchTensorOperations
from .core.scorer import Scorer
from .core.model_abstraction import PyTorchModelWrapper
from .core.writing_token_filter import WritingTokenFilter
from .core.token_filters import DictionaryTokenFilter, CombinedTokenFilter
from .core.implicit_backpath import ImplicitBackpathScorer
from .core.choice_policy import AlphaBetaPolicy
from .core.flux_graph import FluxGraph, FluxGraphConfig
from .core.word_trie import WordTrie
from .core.noodle_explorer import Direction
# --- END HEADER ---

STATIC_DIR = Path(__file__).parent / "flux_radar"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
}


class ModelBundle:
    """Loads GPT-2 and the dictionary/trie infrastructure once, reused across runs.

    Building the curated dictionary and both WordTrie instances is a real,
    multi-second cost (see demo_flux_graph.py) -- doing it once here, at
    first request rather than per FluxGraph run, is what makes repeated
    /api/run calls from the browser fast after the initial load.
    """

    def __init__(self, dictionary_size: int = 20000):
        scorer_obj = Scorer()
        self.tokenizer = scorer_obj.tokenizer
        self.model = scorer_obj.model
        self.device = next(self.model.parameters()).device
        self.wrapper = PyTorchModelWrapper(self.model)

        writing_filter = WritingTokenFilter(self.tokenizer)
        dictionary_filter = DictionaryTokenFilter.from_curated_wordlist(self.tokenizer, n=dictionary_size)
        self.candidate_filter = CombinedTokenFilter([writing_filter, dictionary_filter])
        self.candidate_filter.mask_as_list(self.tokenizer.vocab_size)
        self.word_trie = WordTrie.from_curated_wordlist(n=dictionary_size)
        self.backward_word_trie = WordTrie.from_curated_wordlist(n=dictionary_size, reverse=True)
        self.ops = PyTorchTensorOperations(track_time=False)
        # A real FluxGraph run does GPU work; only one at a time keeps this
        # simple and correct for a single-user local tool rather than
        # trying to be a real multi-tenant inference server.
        self._run_lock = threading.Lock()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run one fresh FluxGraph session end to end and return its tick history.

        One graph, one topology, one continuous line of real model calls,
        for the whole run -- root displacement (see FluxGraph._reroot) only
        ever changes which node is treated as anchor. Nothing is ever
        removed from the graph, so every node in every snapshot is tagged
        "orthogonal": true/false (see FluxGraph.orthogonal_node_ids) purely
        as a display hint -- the frontend can choose not to force those
        into the two-hemisphere layout, but they're still live data, still
        eligible to become anchor themselves later.
        """
        with self._run_lock:
            backpath = ImplicitBackpathScorer(self.wrapper, self.tokenizer, writing_filter=self.candidate_filter)
            config = FluxGraphConfig(
                compute_budget_per_tick=int(params.get("budget", 3)),
                branch_factor=int(params.get("branch", 3)),
                verbose=False,
                backward_left_context=[self.tokenizer.eos_token_id],
                no_repeat_ngram_size=3,
                word_trie=self.word_trie,
                backward_word_trie=self.backward_word_trie,
                max_subword_steps=8,
                max_expand_elements=400_000_000,
                anchor_can_decay=bool(params.get("anchor_can_decay", False)),
            )
            choice_policy = AlphaBetaPolicy(
                alpha=float(params.get("alpha", 0.8)),
                beta=float(params.get("beta", 1.0)),
                seed=params.get("rng_seed"),
            )
            graph = FluxGraph(self.wrapper, backpath, choice_policy, self.ops, config=config, device=self.device)

            seed_text = str(params.get("seed") or "the ocean")
            seed_ids = self.tokenizer.encode(seed_text)
            graph.seed(seed_ids)

            def snapshot_graph() -> Dict[str, Any]:
                orthogonal = graph.orthogonal_node_ids()
                nodes = []
                for nid, n in graph.nodes.items():
                    # A node's own tokens are empty only while it's the
                    # current anchor -- anchor_tokens is whatever it
                    # represents (the original seed, or a former anchor's
                    # own word span once demoted).
                    text = self.tokenizer.decode(n.tokens) if n.tokens else self.tokenizer.decode(graph.anchor_tokens)
                    direction = None if n.direction is None else (
                        "forward" if n.direction is Direction.FORWARD else "backward"
                    )
                    nodes.append({
                        "id": nid,
                        "parent_id": n.parent_id,
                        "direction": direction,
                        "height": n.height,
                        "depth": n.depth,
                        "pressure": round(n.pressure, 4),
                        "local_evidence": round(n.local_evidence, 4),
                        "path_mean": round(n.path_mean, 4),
                        "burned": n.burned,
                        "created_tick": n.created_tick,
                        "text": text,
                        "orthogonal": nid in orthogonal,
                    })
                tokens, score = graph.best_path()
                return {"nodes": nodes, "best_path": self.tokenizer.decode(tokens), "best_score": round(score, 4)}

            history: List[Dict[str, Any]] = []

            def record_tick(tick_idx: int) -> None:
                history.append({"tick": tick_idx, **snapshot_graph()})

            graph.spawn_first_children()
            record_tick(0)

            ticks = max(0, int(params.get("ticks", 6)))
            for t in range(1, ticks + 1):
                graph.tick()
                record_tick(t)

            return {
                "run_id": params.get("run_id") or seed_text,
                "seed_text": seed_text,
                "params": {
                    "ticks": ticks,
                    "budget": config.compute_budget_per_tick,
                    "branch": config.branch_factor,
                    "alpha": choice_policy.alpha,
                    "beta": choice_policy.beta,
                    "anchor_can_decay": config.anchor_can_decay,
                },
                "history": history,
            }


_bundle: Optional[ModelBundle] = None
_bundle_lock = threading.Lock()


def get_bundle() -> ModelBundle:
    global _bundle
    with _bundle_lock:
        if _bundle is None:
            print("[flux-radar] loading GPT-2 and dictionary/trie infrastructure (first request only)...")
            _bundle = ModelBundle()
            print("[flux-radar] ready.")
    return _bundle


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print("[flux-radar]", fmt % args)

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self._serve_file(STATIC_DIR / "index.html")
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):]
            self._serve_file(STATIC_DIR / rel)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body) if body else {}
            bundle = get_bundle()
            result = bundle.run(params)
            self._send_json(200, result)
        except Exception as e:  # noqa: BLE001 -- surfaced to the browser, not swallowed
            self._send_json(500, {"error": str(e)})

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument(
        "--preload", action="store_true",
        help="load GPT-2 and the dictionary/trie infrastructure at startup instead of on first request",
    )
    args = parser.parse_args()

    if args.preload:
        get_bundle()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[flux-radar] serving http://127.0.0.1:{args.port}/")
    if not args.preload:
        print("[flux-radar] first run will take a while (loading GPT-2 + dictionary). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
