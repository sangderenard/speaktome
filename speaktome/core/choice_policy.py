#!/usr/bin/env python3
"""Pluggable per-step token choice policies for beam/noodle expansion.

Every step of a beam or "noodle" walk needs to turn a row of logits into a
short list of child tokens to actually grow into. Today that step is
hardcoded as ``tensor_ops.topk(logprobs, k=..., dim=-1)`` inside
``LookaheadController.run`` -- always follow the model's own argmax-ish
ranking. This module gives that step a name and a swappable interface so a
future "graph traversal itinerary" (forced choices, or a stochastic walk
that reveals the shape of the model's belief rather than just its peak) can
sit behind the same call site without rewriting the surrounding loop.

Wiring ``ChoicePolicy`` into ``LookaheadController`` is a follow-up step;
this module only establishes the interface and a first concrete policy
pair so both can be tested in isolation.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from tensors import AbstractTensor
# --- END HEADER ---


class ChoicePolicy(ABC):
    """Decide which candidate tokens survive a single expansion step.

    Mirrors ``AbstractTensor.topk``'s contract: given raw logits shaped
    ``[batch, vocab]``, return ``(scores, indices)`` each shaped
    ``[batch, k]``. ``scores`` are always the model's own log-probabilities
    for the chosen tokens, even for policies that *select* using some other
    distribution -- the recorded score should reflect what the model
    actually believes, not the exploration bias used to pick it.
    """

    @abstractmethod
    def choose(
        self, logits: AbstractTensor, k: int
    ) -> Tuple[AbstractTensor, AbstractTensor]:
        """Return ``(scores, indices)`` for the chosen tokens per row."""
        raise NotImplementedError


class TopKPolicy(ChoicePolicy):
    """Deterministic top-k over the model's own ranking (today's default).

    Equivalent to the inline ``tensor_ops.topk(log_softmax(logits / temp),
    k=k, dim=-1)`` calls this policy is meant to replace.
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def choose(
        self, logits: AbstractTensor, k: int
    ) -> Tuple[AbstractTensor, AbstractTensor]:
        log_probs = (logits / self.temperature).log_softmax(dim=-1)
        return AbstractTensor.topk(log_probs, k=k, dim=-1)


class AlphaBetaPolicy(ChoicePolicy):
    """Interpolate between uniform exploration and belief-following sampling.

    ``P(token) = alpha * softmax(logits / beta) + (1 - alpha) * uniform``

    ``alpha=0`` draws uniformly at random among all candidates; ``alpha=1``
    draws from the model's own distribution at temperature ``beta``.
    Intermediate values blend the two, so the resulting sample population
    reflects the actual shape of the model's belief rather than always
    chasing its peak -- the "wide truth" a noodle ensemble needs to be
    useful.

    Sampling is without replacement, via the Efraimidis-Spirakis weighted
    reservoir trick: draw ``u ~ Uniform(0, 1)`` per candidate and keep the
    ``k`` candidates with the largest ``u ** (1 / weight)``. This still
    runs row-by-row because ``AbstractTensor`` has no sampling primitive
    yet, and extending the tensor abstraction is explicitly out of scope
    for this pass -- see the bidirectional-diffusion vision brief. Pulling
    a row of probabilities out via ``.tolist()`` is a deliberate, honest
    cost, not a shortcut around a solvable problem.

    The per-row arithmetic (mixed_weights, the sampling key, and the
    final ranking) is vectorized with plain numpy rather than a Python
    loop -- real profiling on a GPT-2 run showed this doing a full-vocab
    (~50k) ``pow()`` + Python-level sort on *every* choose() call, even
    though only ``k`` candidates ever survive, and that alone cost more
    wall-clock time than any GPU work in the same tick. This is CPU-only
    housekeeping around an unchanged algorithm, not an extension of
    ``AbstractTensor`` itself: same formula, same number and order of
    ``self._rng.random()`` draws (only for candidates with weight > 0,
    exactly like the original), same tie-breaking (numpy's stable sort
    matches Python's stable ``sort(reverse=True)`` for equal keys) --
    verified bit-for-bit against the original over hundreds of seeded
    trials, including zero/negative-weight edge cases, before landing.
    """

    def __init__(self, alpha: float, beta: float = 1.0, seed: Optional[int] = None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.alpha = alpha
        self.beta = beta
        self._rng = random.Random(seed)

    def choose(
        self, logits: AbstractTensor, k: int
    ) -> Tuple[AbstractTensor, AbstractTensor]:
        log_probs = (logits / self.beta).log_softmax(dim=-1)
        rows = log_probs.tolist()
        vocab_size = len(rows[0]) if rows else 0
        k = min(k, vocab_size)
        uniform_p = 1.0 / vocab_size if vocab_size else 0.0

        chosen_scores: List[List[float]] = []
        chosen_indices: List[List[int]] = []
        for row in rows:
            row_arr = np.asarray(row, dtype=np.float64)
            # np.power(e, x), not np.exp(x): the two aren't guaranteed
            # bit-identical (verified they can differ by ~1ulp), and
            # np.power matches the original per-element pow() exactly.
            mixed_weights = self.alpha * np.power(2.718281828459045, row_arr) + (
                1.0 - self.alpha
            ) * uniform_p
            picked = self._weighted_sample_without_replacement(mixed_weights, k)
            # Report the true model log-probability, sorted descending so
            # the output shape/order matches AbstractTensor.topk's contract.
            picked.sort(key=lambda i: row[i], reverse=True)
            chosen_indices.append(picked)
            chosen_scores.append([row[i] for i in picked])

        backend_cls = type(log_probs)
        device = log_probs.get_device()
        scores_t = backend_cls.tensor(
            chosen_scores, dtype=log_probs.float_dtype, device=device
        )
        indices_t = backend_cls.tensor(
            chosen_indices, dtype=log_probs.long_dtype, device=device
        )
        return scores_t, indices_t

    def _weighted_sample_without_replacement(
        self, weights: np.ndarray, k: int
    ) -> List[int]:
        """Vectorized Efraimidis-Spirakis top-k, bit-identical to the original loop.

        ``self._rng.random()`` is drawn only for weight > 0 candidates, in
        ascending index order -- exactly the subset and sequence the
        original per-element loop consumed -- so the RNG stream a given
        seed produces is unaffected. ``np.argsort(..., kind="stable")``
        mirrors Python's ``list.sort(reverse=True)`` tie-breaking (equal
        keys keep ascending-index order), which matters when several
        weight <= 0 candidates tie at key=-inf.
        """
        n = weights.shape[0]
        keys = np.full(n, -np.inf, dtype=np.float64)
        positive_idx = np.flatnonzero(weights > 0.0)
        u = np.fromiter(
            (self._rng.random() for _ in range(positive_idx.shape[0])),
            dtype=np.float64,
            count=positive_idx.shape[0],
        )
        if positive_idx.size:
            keys[positive_idx] = u ** (1.0 / weights[positive_idx])
        order = np.argsort(-keys, kind="stable")
        return order[:k].tolist()
