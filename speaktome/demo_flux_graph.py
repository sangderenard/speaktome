#!/usr/bin/env python3
"""Demo: grow a bidirectional flux graph from a seed, using real GPT-2.

Unlike demo_bidirectional_diffusion.py (which commits to a single string),
this never commits to anything. It builds a graph: nodes hang off an
anchor in both directions, pressure propagates through the graph each
tick like current through a resistor network, compute expands the
highest-pressure nodes, and starved extremities burn off. At every tick
this prints the live "best path right now" -- a snapshot, not an answer --
alongside basic graph stats, so you can watch the graph grow and reshape
itself rather than watching a single string get appended to.

Usage:
    python -m speaktome.demo_flux_graph ["seed text"] [--ticks N] [--budget K] [--branch B]
"""
from __future__ import annotations

import argparse
import time

from tensors.torch_backend import PyTorchTensorOperations

from .core.scorer import Scorer
from .core.model_abstraction import PyTorchModelWrapper
from .core.writing_token_filter import WritingTokenFilter
from .core.implicit_backpath import ImplicitBackpathScorer
from .core.choice_policy import AlphaBetaPolicy
from .core.flux_graph import FluxGraph, FluxGraphConfig
# --- END HEADER ---

DEFAULT_SEED = " brown fox jumps"


def describe_tick(graph: FluxGraph, tokenizer) -> None:
    live = [n for n in graph.nodes.values() if not n.burned]
    burned = [n for n in graph.nodes.values() if n.burned]
    tokens, score = graph.best_path()
    text = tokenizer.decode(tokens)
    print(f"  nodes: {len(live)} live, {len(burned)} burned")
    print(f"  best path (mean log-prob/token={score:.3f}): {text!r}")


def run_demo(
    seed_text: str, ticks: int, budget: int, branch_factor: int,
    alpha: float, beta: float, seed: int | None,
) -> None:
    print("Loading GPT-2 ...")
    t0 = time.time()
    scorer_obj = Scorer()
    tokenizer = scorer_obj.tokenizer
    model = scorer_obj.model
    device = next(model.parameters()).device
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")

    wrapper = PyTorchModelWrapper(model)
    writing_filter = WritingTokenFilter(tokenizer)
    backpath = ImplicitBackpathScorer(wrapper, tokenizer, writing_filter=writing_filter)
    ops = PyTorchTensorOperations(track_time=False)

    config = FluxGraphConfig(
        compute_budget_per_tick=budget,
        branch_factor=branch_factor,
        verbose=True,
        backward_left_context=[tokenizer.eos_token_id],
    )
    # TopKPolicy (deterministic top-k) makes forward expansion structurally
    # favor whichever child the model itself ranks highest -- the same
    # attractor plain argmax greedy decoding falls into. AlphaBetaPolicy
    # samples instead: alpha=1 reduces to the model's own ranking, alpha<1
    # mixes in real uniform exploration so the graph doesn't just retrace
    # the greedy path with minor decoration.
    choice_policy = AlphaBetaPolicy(alpha=alpha, beta=beta, seed=seed)
    graph = FluxGraph(wrapper, backpath, choice_policy, ops, config=config, device=device)

    seed_ids = tokenizer.encode(seed_text)
    print(f"\nSeed: {seed_text!r} -> {seed_ids}")
    graph.seed(seed_ids)

    print("\n--- initial expansion (one forward + one backward child off the anchor) ---")
    t0 = time.time()
    graph.spawn_first_children()
    print(f"  [{time.time() - t0:.1f}s]")
    describe_tick(graph, tokenizer)

    for step in range(ticks):
        print(f"\n--- tick {step + 1} ---")
        t0 = time.time()
        graph.tick()
        print(f"  [{time.time() - t0:.1f}s]")
        describe_tick(graph, tokenizer)

    print("\nFinal graph:")
    describe_tick(graph, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", default=DEFAULT_SEED, help="seed text to grow from")
    parser.add_argument("--ticks", type=int, default=4, help="number of discrete flux ticks to run")
    parser.add_argument("--budget", type=int, default=2, help="nodes expanded per tick (compute budget)")
    parser.add_argument("--branch", type=int, default=3, help="children per expansion")
    parser.add_argument("--alpha", type=float, default=0.8, help="0=uniform exploration, 1=pure model ranking")
    parser.add_argument("--beta", type=float, default=1.0, help="softmax temperature within the model-ranking term")
    parser.add_argument("--seed-rng", type=int, default=None, help="seed for reproducible sampling (default: random)")
    args = parser.parse_args()
    run_demo(args.seed, args.ticks, args.budget, args.branch, args.alpha, args.beta, args.seed_rng)


if __name__ == "__main__":
    main()
