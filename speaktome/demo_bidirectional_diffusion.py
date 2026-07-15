#!/usr/bin/env python3
"""Demo: extend a seed string in both directions using real GPT-2 scoring.

Forward extension is ordinary next-token stepping. Backward extension uses
``ImplicitBackpathScorer`` -- for each candidate previous token, prepend it
to the current left edge and read off how much more likely that makes the
rest of the string, in one batched forward pass over the (filtered)
vocabulary. Neither direction just commits to the top pick silently: at
every step this prints the top-k candidates and their scores, so you can
see the shape of the model's belief at that point (a sharp peak vs. a flat
plateau) rather than only the single choice made.

No new model, no training -- this is the existing forward-trained GPT-2,
probed from both directions.

Usage:
    python -m speaktome.demo_bidirectional_diffusion ["seed text"] [--steps N] [--show-k K]
"""
from __future__ import annotations

import argparse
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


def backward_step(
    backpath: ImplicitBackpathScorer,
    ops: AbstractTensor,
    suffix_tokens: AbstractTensor,
    k: int,
    max_batch_size: int,
) -> Tuple[List[float], List[int]]:
    """Return the top-k (score, token id) predecessors of ``suffix_tokens``."""
    pool = backpath.candidate_pool(
        ops, vocab_size=backpath.tokenizer.vocab_size, device=suffix_tokens.get_device()
    )
    scores = backpath.score_candidates(suffix_tokens, pool, max_batch_size=max_batch_size)
    top_scores, top_idx = AbstractTensor.topk(scores, k=k, dim=0)
    candidate_ids = [int(pool[i].item()) for i in top_idx.tolist()]
    return top_scores.tolist(), candidate_ids


def print_landscape(label: str, tokenizer, scored: Tuple[List[float], List[int]]) -> None:
    scores, ids = scored
    print(f"  {label} landscape (top {len(ids)}):")
    for score, tid in zip(scores, ids):
        text = tokenizer.decode([tid]).replace("\n", "\\n")
        print(f"    {score:8.3f}  {text!r}")


def run_demo(seed_text: str, steps_each_side: int, show_k: int, max_batch_size: int) -> None:
    print(f"Loading GPT-2 ...")
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

    seed_ids = tokenizer.encode(seed_text)
    print(f"\nSeed: {seed_text!r} -> {seed_ids}")

    left = ops.tensor(seed_ids, dtype=ops.long_dtype, device=device)
    right = ops.tensor(seed_ids, dtype=ops.long_dtype, device=device)

    for step in range(steps_each_side):
        print(f"\n--- step {step + 1} ---")

        fwd_scores, fwd_ids = forward_step(wrapper, right, k=show_k)
        print_landscape("forward", tokenizer, (fwd_scores, fwd_ids))
        chosen_fwd = fwd_ids[0]
        right = ops.tensor(right.tolist() + [chosen_fwd], dtype=ops.long_dtype, device=device)
        print(f"  -> forward commits {chosen_fwd} ({tokenizer.decode([chosen_fwd])!r})")

        t0 = time.time()
        bwd_scores, bwd_ids = backward_step(backpath, ops, left, k=show_k, max_batch_size=max_batch_size)
        print_landscape("backward", tokenizer, (bwd_scores, bwd_ids))
        chosen_bwd = bwd_ids[0]
        left = ops.tensor([chosen_bwd] + left.tolist(), dtype=ops.long_dtype, device=device)
        print(f"  -> backward commits {chosen_bwd} ({tokenizer.decode([chosen_bwd])!r}) [{time.time() - t0:.1f}s]")

        full_ids = left.tolist() + right.tolist()[len(seed_ids):]
        print(f"  string so far: {tokenizer.decode(full_ids)!r}")

    full_ids = left.tolist() + right.tolist()[len(seed_ids):]
    print(f"\nFinal: {tokenizer.decode(full_ids)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", default=DEFAULT_SEED, help="seed text to extend")
    parser.add_argument("--steps", type=int, default=3, help="steps to grow on each side")
    parser.add_argument("--show-k", type=int, default=5, help="candidates to display per step")
    parser.add_argument(
        "--max-batch-size", type=int, default=1024,
        help="chunk size for the backward candidate sweep (memory vs. speed)",
    )
    args = parser.parse_args()
    run_demo(args.seed, args.steps, args.show_k, args.max_batch_size)


if __name__ == "__main__":
    main()
