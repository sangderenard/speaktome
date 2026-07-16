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
    python -m speaktome.demo_flux_graph --visualize            # live spring-layout window
    python -m speaktome.demo_flux_graph --poetic                # bias candidacy toward rhyme
"""
from __future__ import annotations

import argparse
import time

from tensors.torch_backend import PyTorchTensorOperations

from .core.scorer import Scorer
from .core.model_abstraction import PyTorchModelWrapper
from .core.writing_token_filter import WritingTokenFilter
from .core.token_filters import DictionaryTokenFilter, CombinedTokenFilter
from .core.implicit_backpath import ImplicitBackpathScorer
from .core.choice_policy import AlphaBetaPolicy
from .core.flux_graph import FluxGraph, FluxGraphConfig
from .core.poetic_attractor import PoeticAttractor
from .core.word_trie import WordTrie
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
    alpha: float, beta: float, seed: int | None, no_repeat_ngram_size: int | None,
    visualize: bool, poetic: PoeticAttractor | None, poetic_scale: float, poetic_shortlist_k: int,
    dictionary_file: str | None, auto_dictionary: bool, dictionary_size: int,
    word_growth: bool, max_subword_steps: int,
    auxin_suppression: float, auxin_decay: float, head_pressure_coefficient: float,
    max_expand_elements: int | None,
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
    word_trie = None
    backward_word_trie = None
    if dictionary_file:
        t0 = time.time()
        dictionary_filter = DictionaryTokenFilter.from_file(tokenizer, dictionary_file)
        print(f"  [dictionary: {dictionary_file}, {time.time() - t0:.1f}s]")
        if word_growth:
            word_trie = WordTrie.from_file(dictionary_file)
            # Backward growth discovers a word from its end toward its
            # start (each new token gets prepended), so it needs a trie
            # built over reversed word strings to ask "is this a valid
            # suffix-so-far" instead of "is this a valid prefix-so-far" --
            # see WordTrie's reverse parameter.
            backward_word_trie = WordTrie.from_file(dictionary_file, reverse=True)
    elif auto_dictionary:
        t0 = time.time()
        # Real dictionary (nltk) narrowed to its dictionary_size most common
        # words (wordfreq rank) -- see word_sources.curated_english_wordlist
        # for why neither source alone is enough. Built once here; both the
        # filter and the tries draw from the exact same word list.
        dictionary_filter = DictionaryTokenFilter.from_curated_wordlist(tokenizer, n=dictionary_size)
        print(f"  [auto dictionary: {dictionary_size} words (real dictionary x popularity rank), {time.time() - t0:.1f}s]")
        if word_growth:
            t0 = time.time()
            word_trie = WordTrie.from_curated_wordlist(n=dictionary_size)
            backward_word_trie = WordTrie.from_curated_wordlist(n=dictionary_size, reverse=True)
            print(f"  [word tries built from the same auto dictionary (forward + reversed), {time.time() - t0:.1f}s]")
    else:
        dictionary_filter = None
        print("  [NO DICTIONARY FILTER ACTIVE -- backward will score the full junk-filtered vocab "
              "(tens of thousands of candidates). Pass --auto-dictionary or --dictionary-file to narrow it.]")

    if dictionary_filter is not None:
        candidate_filter = CombinedTokenFilter([writing_filter, dictionary_filter])
        # Force the (one-time, cached) classification pass now rather than
        # on the first backward expansion, so its cost shows up here
        # instead of looking like part of tick 1's timing.
        t0 = time.time()
        kept = sum(candidate_filter.mask_as_list(tokenizer.vocab_size))
        print(f"  [candidate filter: {kept}/{tokenizer.vocab_size} vocab ids kept, {time.time() - t0:.1f}s]")
        if word_trie is not None:
            print("  [word growth enabled: edges are whole words, pruned/bounded by the same dictionary]")
    else:
        candidate_filter = writing_filter
    backpath = ImplicitBackpathScorer(wrapper, tokenizer, writing_filter=candidate_filter)
    ops = PyTorchTensorOperations(track_time=False)

    config = FluxGraphConfig(
        compute_budget_per_tick=budget,
        branch_factor=branch_factor,
        verbose=True,
        backward_left_context=[tokenizer.eos_token_id],
        no_repeat_ngram_size=no_repeat_ngram_size,
        poetic_attractor=poetic,
        poetic_scale=poetic_scale,
        poetic_shortlist_k=poetic_shortlist_k,
        word_trie=word_trie,
        backward_word_trie=backward_word_trie,
        max_subword_steps=max_subword_steps,
        auxin_suppression=auxin_suppression,
        auxin_decay=auxin_decay,
        head_pressure_coefficient=head_pressure_coefficient,
        max_expand_elements=max_expand_elements,
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

    visualizer = None
    if visualize:
        from .core.graph_visualizer import FluxGraphVisualizer
        visualizer = FluxGraphVisualizer(graph)
        visualizer.start()
        print("  [visualizer started on its own thread]")

    try:
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
    finally:
        if visualizer is not None:
            visualizer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", default=DEFAULT_SEED, help="seed text to grow from")
    parser.add_argument("--ticks", type=int, default=4, help="number of discrete flux ticks to run")
    parser.add_argument("--budget", type=int, default=2, help="nodes expanded per tick (compute budget)")
    parser.add_argument("--branch", type=int, default=3, help="children per expansion")
    parser.add_argument("--alpha", type=float, default=0.8, help="0=uniform exploration, 1=pure model ranking")
    parser.add_argument("--beta", type=float, default=1.0, help="softmax temperature within the model-ranking term")
    parser.add_argument("--seed-rng", type=int, default=None, help="seed for reproducible sampling (default: random)")
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, default=3,
        help="block a candidate if it would recreate an n-gram already in the node's known context (0 disables)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="open a live pygame/OpenGL spring-layout window on its own thread",
    )
    parser.add_argument("--poetic", action="store_true", help="bias candidacy toward rhyme/alliteration")
    parser.add_argument("--poetic-rhyme-weight", type=float, default=0.6)
    parser.add_argument("--poetic-slant-weight", type=float, default=0.3)
    parser.add_argument("--poetic-alliteration-weight", type=float, default=0.2)
    parser.add_argument("--poetic-internal-weight", type=float, default=0.4)
    parser.add_argument(
        "--poetic-scale", type=float, default=1.0,
        help="weight of the poetic bonus relative to real model evidence when re-ranking",
    )
    parser.add_argument(
        "--poetic-shortlist-k", type=int, default=20,
        help="how many of the model's own top candidates are eligible for poetic re-ranking",
    )
    parser.add_argument(
        "--dictionary-file", type=str, default=None,
        help="newline-separated word list; shrinks the backward candidate pool to just these words "
             "(composed with the usual junk/control-character filter, not a replacement for it)",
    )
    parser.add_argument(
        "--auto-dictionary", action="store_true", default=True,
        help="build the dictionary automatically (real nltk dictionary words, narrowed to the "
             "N most common via wordfreq rank -- see word_sources.py). On by default; ignored if "
             "--dictionary-file is given instead. Requires 'pip install nltk wordfreq'. "
             "Pass --no-auto-dictionary to run with no dictionary filter at all.",
    )
    parser.add_argument("--no-auto-dictionary", dest="auto_dictionary", action="store_false")
    parser.add_argument(
        "--dictionary-size", type=int, default=20000,
        help="how many words in the auto-built dictionary (only used with --auto-dictionary)",
    )
    parser.add_argument(
        "--word-growth", action="store_true", default=True,
        help="with --dictionary-file/--auto-dictionary, make each graph edge a whole word (trie-pruned, "
             "bounded subword beam search) instead of one BPE token; pass --no-word-growth to disable",
    )
    parser.add_argument("--no-word-growth", dest="word_growth", action="store_false")
    parser.add_argument(
        "--max-subword-steps", type=int, default=8,
        help="safety cap on how many subtokens one word's growth can consume (only relevant with word growth)",
    )
    parser.add_argument(
        "--auxin-suppression", type=float, default=0.0,
        help="apical-dominance-style branch suppression: a strong, uncontested tip dampens branch_factor "
             "at competing branches elsewhere in the graph, attenuated by hop distance. 0 disables it "
             "(branch_factor stays constant everywhere, the pre-auxin default).",
    )
    parser.add_argument(
        "--auxin-decay", type=float, default=0.6,
        help="attenuation per hop as auxin propagates away from its source (1.0 = no attenuation)",
    )
    parser.add_argument(
        "--head-pressure", type=float, default=0.0,
        help="cost subtracted from a node's pressure proportional to |height| (token-distance from the "
             "anchor, same in either direction) -- sustaining flow further out costs more. 0 disables it.",
    )
    parser.add_argument(
        "--max-expand-elements", type=int, default=400_000_000,
        help="memory-budget cap (rows x row_len x vocab_size) for any single model forward call inside "
             "a tick's expand batch -- the row count per call shrinks as context grows instead of staying "
             "flat, which is what actually caused a real CUDA OOM on a long run (expand-batch-chunk-size "
             "alone doesn't account for row_len growing over time). Default (400,000,000) is ~1.6GB per "
             "call in float32; lower it on a smaller GPU, or pass 0 to disable and fall back to the flat "
             "--expand-batch-chunk-size cap only (the pre-this-safeguard behavior).",
    )
    args = parser.parse_args()
    no_repeat = args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None
    max_expand_elements = args.max_expand_elements if args.max_expand_elements > 0 else None
    poetic = None
    if args.poetic:
        poetic = PoeticAttractor(
            rhyme_weight=args.poetic_rhyme_weight,
            slant_rhyme_weight=args.poetic_slant_weight,
            alliteration_weight=args.poetic_alliteration_weight,
            internal_rhyme_weight=args.poetic_internal_weight,
        )
    run_demo(
        args.seed, args.ticks, args.budget, args.branch, args.alpha, args.beta, args.seed_rng, no_repeat,
        args.visualize, poetic, args.poetic_scale, args.poetic_shortlist_k,
        args.dictionary_file, args.auto_dictionary, args.dictionary_size,
        args.word_growth, args.max_subword_steps, args.auxin_suppression, args.auxin_decay,
        args.head_pressure, max_expand_elements,
    )


if __name__ == "__main__":
    main()
