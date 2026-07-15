#!/usr/bin/env python3
"""Demo: extend a seed string in both directions using real GPT-2 scoring.

Forward extension is ordinary next-token stepping. Backward extension uses
``ImplicitBackpathScorer`` -- for each candidate previous token, prepend it
to the current left edge and read off how much more likely that makes the
rest of the string, in one batched forward pass over the (filtered)
vocabulary. Neither direction just commits to the top pick silently: at
every step this prints a wide slice of the landscape (top-N, bottom-N, and
summary statistics over the *entire* scored pool), so you can see the
actual shape of the model's belief -- not just the single choice made, and
not just a top-5 that could hide whether the distribution is peaked or
flat.

Backward candidates are scored with a real ``<|endoftext|>`` document-
boundary token as left context, not zero context -- every backward
candidate is, by construction, being scored with nothing of its own to its
left, and zero context is a regime the model rarely saw cleanly during
training (most training windows are mid-document, not document starts).

No new model, no training -- this is the existing forward-trained GPT-2,
probed from both directions.

Usage:
    python -m speaktome.demo_bidirectional_diffusion ["seed text"] [--steps N] [--show-k K]
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Tuple

from tensors import AbstractTensor
from tensors.torch_backend import PyTorchTensorOperations

from .core.scorer import Scorer
from .core.model_abstraction import PyTorchModelWrapper
from .core.writing_token_filter import WritingTokenFilter
from .core.implicit_backpath import ImplicitBackpathScorer
from .core.choice_policy import TopKPolicy
# --- END HEADER ---

DEFAULT_SEED = " brown fox jumps"


def forward_step(
    model_wrapper: PyTorchModelWrapper, tokens: AbstractTensor, k: int
) -> Tuple[List[float], List[int]]:
    """Return the top-k (log-prob, token id) continuations after ``tokens``."""
    backend_cls = type(tokens)
    device = tokens.get_device()
    row = tokens.tolist()

    batch = backend_cls.tensor([row], dtype=tokens.long_dtype, device=device)
    mask = backend_cls.tensor([[1] * len(row)], dtype=tokens.long_dtype, device=device)

    outputs = model_wrapper.forward(input_ids=batch.data, attention_mask=mask.data)
    logits = batch.ensure_tensor(outputs["logits"])  # [1, T, vocab]
    last_logits = logits[0, -1, :].unsqueeze(0)  # [1, vocab]

    scores, indices = TopKPolicy().choose(last_logits, k=k)
    return scores.tolist()[0], indices.tolist()[0]


def backward_landscape(
    backpath: ImplicitBackpathScorer,
    ops: AbstractTensor,
    suffix_tokens: AbstractTensor,
    max_batch_size: int,
    left_context: List[int],
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Return (scores, candidate_ids) for the *entire* filtered candidate pool."""
    pool = backpath.candidate_pool(
        ops, vocab_size=backpath.tokenizer.vocab_size, device=suffix_tokens.get_device()
    )
    scores = backpath.score_candidates(
        suffix_tokens, pool, max_batch_size=max_batch_size, left_context=left_context
    )
    return scores, pool


def print_wide_landscape(label: str, tokenizer, scores: List[float], ids: List[int], top_n: int, bottom_n: int) -> None:
    paired = sorted(zip(scores, ids), key=lambda p: p[0], reverse=True)
    n = len(paired)
    mean = statistics.mean(scores)
    stdev = statistics.pstdev(scores) if n > 1 else 0.0
    print(f"  {label} landscape: {n} candidates, mean={mean:.3f}, stdev={stdev:.3f}, "
          f"min={paired[-1][0]:.3f}, max={paired[0][0]:.3f}")
    print(f"    top {top_n}:")
    for score, tid in paired[:top_n]:
        text = tokenizer.decode([tid]).replace("\n", "\\n")
        print(f"      {score:9.3f}  {text!r}")
    if bottom_n > 0 and n > top_n:
        print(f"    bottom {bottom_n}:")
        for score, tid in paired[-bottom_n:]:
            text = tokenizer.decode([tid]).replace("\n", "\\n")
            print(f"      {score:9.3f}  {text!r}")


def print_top_landscape(label: str, tokenizer, scores: List[float], ids: List[int]) -> None:
    print(f"  {label} landscape (top {len(ids)}):")
    for score, tid in zip(scores, ids):
        text = tokenizer.decode([tid]).replace("\n", "\\n")
        print(f"    {score:8.3f}  {text!r}")


def run_demo(
    seed_text: str, steps_each_side: int, show_k: int, max_batch_size: int,
    landscape_top: int, landscape_bottom: int,
) -> None:
    print("Loading GPT-2 ...")
    t0 = time.time()
    scorer_obj = Scorer()
    tokenizer = scorer_obj.tokenizer
    model = scorer_obj.model
    print(f"  loaded in {time.time() - t0:.1f}s on {next(model.parameters()).device}")

    wrapper = PyTorchModelWrapper(model)
    writing_filter = WritingTokenFilter(tokenizer)
    backpath = ImplicitBackpathScorer(wrapper, tokenizer, writing_filter=writing_filter)
    ops = PyTorchTensorOperations(track_time=False)
    device = next(model.parameters()).device
    left_context = [tokenizer.eos_token_id]

    seed_ids = tokenizer.encode(seed_text)
    print(f"\nSeed: {seed_text!r} -> {seed_ids}")

    left = ops.tensor(seed_ids, dtype=ops.long_dtype, device=device)
    right = ops.tensor(seed_ids, dtype=ops.long_dtype, device=device)

    for step in range(steps_each_side):
        print(f"\n--- step {step + 1} ---")

        fwd_scores, fwd_ids = forward_step(wrapper, right, k=max(show_k, landscape_top))
        print_top_landscape("forward", tokenizer, fwd_scores[:show_k], fwd_ids[:show_k])
        chosen_fwd = fwd_ids[0]
        right = ops.tensor(right.tolist() + [chosen_fwd], dtype=ops.long_dtype, device=device)
        print(f"  -> forward commits {chosen_fwd} ({tokenizer.decode([chosen_fwd])!r})")

        t0 = time.time()
        all_scores_t, pool = backward_landscape(backpath, ops, left, max_batch_size, left_context)
        print(f"  [{time.time() - t0:.1f}s scoring {pool.shape[0]} candidates]")
        print_wide_landscape(
            "backward", tokenizer, all_scores_t.tolist(), pool.tolist(),
            top_n=landscape_top, bottom_n=landscape_bottom,
        )
        top_scores, top_idx = AbstractTensor.topk(all_scores_t, k=1, dim=0)
        chosen_bwd = int(pool[top_idx.tolist()[0]].item())
        left = ops.tensor([chosen_bwd] + left.tolist(), dtype=ops.long_dtype, device=device)
        print(f"  -> backward commits {chosen_bwd} ({tokenizer.decode([chosen_bwd])!r})")

        full_ids = left.tolist() + right.tolist()[len(seed_ids):]
        print(f"  string so far: {tokenizer.decode(full_ids)!r}")

    full_ids = left.tolist() + right.tolist()[len(seed_ids):]
    print(f"\nFinal: {tokenizer.decode(full_ids)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", default=DEFAULT_SEED, help="seed text to extend")
    parser.add_argument("--steps", type=int, default=3, help="steps to grow on each side")
    parser.add_argument("--show-k", type=int, default=5, help="forward candidates to display per step")
    parser.add_argument(
        "--max-batch-size", type=int, default=1024,
        help="chunk size for the backward candidate sweep (memory vs. speed)",
    )
    parser.add_argument("--landscape-top", type=int, default=20, help="backward: how many top candidates to show")
    parser.add_argument("--landscape-bottom", type=int, default=5, help="backward: how many bottom candidates to show")
    args = parser.parse_args()
    run_demo(args.seed, args.steps, args.show_k, args.max_batch_size, args.landscape_top, args.landscape_bottom)


if __name__ == "__main__":
    main()
