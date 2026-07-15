#!/usr/bin/env python3
"""Probe a forward-only model for its implicit backward beliefs.

A causal language model was never trained to answer "what comes before
this suffix" -- but it can still answer a related question: for a
candidate previous token, how much more likely does the suffix become once
that candidate is prepended? Scoring every (filtered) candidate this way,
in a single batched forward pass, gives the full shape of that answer
rather than a greedy guess at it -- the "implicit backpath" landscape
described in the forward/backward diffusion vision brief.

This is a single-shot dense query, not an iterative beam controller: given
a suffix and a pool of candidate predecessors, it returns one score per
candidate by prepending each candidate to the suffix and reading off the
teacher-forced log-likelihood the (unmodified, forward-trained) model
assigns to the suffix. No new model, no training.
"""
from __future__ import annotations

from typing import Any, Optional

from tensors import AbstractTensor
from .model_abstraction import AbstractModelWrapper
from .writing_token_filter import WritingTokenFilter
# --- END HEADER ---


class ImplicitBackpathScorer:
    """Scores candidate predecessor tokens by how well they explain a suffix."""

    def __init__(
        self,
        model_wrapper: AbstractModelWrapper,
        tokenizer: Any,
        writing_filter: Optional[WritingTokenFilter] = None,
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.writing_filter = writing_filter

    def candidate_pool(
        self, tensor_ops: AbstractTensor, vocab_size: int, device: Any = None
    ) -> AbstractTensor:
        """Return the vocab ids to probe, filtered down to ordinary writing if configured."""
        backend_cls = type(tensor_ops)
        all_ids = AbstractTensor.arange(
            0, vocab_size, device=device, dtype=tensor_ops.long_dtype, cls=backend_cls
        )
        if self.writing_filter is None:
            return all_ids
        mask = self.writing_filter.build_mask(tensor_ops, vocab_size, device=device)
        return all_ids[mask]

    def score_candidates(
        self,
        suffix_tokens: AbstractTensor,
        candidate_ids: AbstractTensor,
        max_batch_size: Optional[int] = 2048,
    ) -> AbstractTensor:
        """Return one score per candidate: teacher-forced log-likelihood of ``suffix_tokens``
        under the forward model when that candidate is prepended.

        ``suffix_tokens`` is a 1-D tensor ``[L]``, ``candidate_ids`` a 1-D tensor ``[N]``.
        Returns a 1-D tensor ``[N]``.

        A full (filtered) vocabulary is tens of thousands of candidates, and
        each one expands into its own ``[1+L]`` row -- large enough to blow
        past GPU memory in a single forward pass. ``max_batch_size`` chunks
        the candidate pool and concatenates the per-chunk scores; pass
        ``None`` to force a single unchunked pass (mainly useful for tests
        with tiny candidate pools).
        """
        backend_cls = type(suffix_tokens)
        device = suffix_tokens.get_device()
        float_dtype = suffix_tokens.float_dtype

        N = candidate_ids.shape[0]
        L = suffix_tokens.shape[0]

        if N == 0:
            return backend_cls.tensor([], dtype=float_dtype, device=device)
        if L == 0:
            return backend_cls.tensor([0.0] * N, dtype=float_dtype, device=device)

        if max_batch_size is None or N <= max_batch_size:
            return self._score_batch(suffix_tokens, candidate_ids)

        chunks = []
        for start in range(0, N, max_batch_size):
            chunk_ids = candidate_ids[start : start + max_batch_size]
            chunks.append(self._score_batch(suffix_tokens, chunk_ids))
        return AbstractTensor.cat(chunks, dim=0)

    def _score_batch(
        self, suffix_tokens: AbstractTensor, candidate_ids: AbstractTensor
    ) -> AbstractTensor:
        """Score one batch of candidates in a single forward pass (see score_candidates)."""
        backend_cls = type(suffix_tokens)
        device = suffix_tokens.get_device()
        long_dtype = suffix_tokens.long_dtype
        float_dtype = suffix_tokens.float_dtype

        N = candidate_ids.shape[0]
        L = suffix_tokens.shape[0]

        suffix_list = suffix_tokens.tolist()
        candidate_list = candidate_ids.tolist()
        batch_rows = [[c] + suffix_list for c in candidate_list]

        batch_tokens = backend_cls.tensor(batch_rows, dtype=long_dtype, device=device)
        attention_mask = backend_cls.tensor(
            [[1] * (1 + L)] * N, dtype=long_dtype, device=device
        )

        # AbstractModelWrapper.forward operates on raw backend tensors, not
        # AbstractTensor wrappers -- unwrap going in, rewrap coming out.
        outputs = self.model_wrapper.forward(
            input_ids=batch_tokens.data, attention_mask=attention_mask.data
        )
        logits = batch_tokens.ensure_tensor(outputs["logits"])  # [N, 1+L, vocab]

        total = backend_cls.tensor([0.0] * N, dtype=float_dtype, device=device)
        for position in range(L):
            target_id = int(suffix_tokens[position].item())
            log_probs = logits[:, position, :].log_softmax(dim=-1)
            total = total + log_probs[:, target_id]

        return total
