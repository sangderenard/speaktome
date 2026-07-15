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

from typing import Any, List, Optional, Protocol

from tensors import AbstractTensor
from .model_abstraction import AbstractModelWrapper

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
# --- END HEADER ---


class TokenFilter(Protocol):
    """Structural type for anything that can classify a vocabulary into a keep-mask.

    WritingTokenFilter, DictionaryTokenFilter, and CombinedTokenFilter
    (token_filters.py) all satisfy this without inheriting from anything --
    ``candidate_pool`` only ever calls ``build_mask``.
    """

    def build_mask(self, tensor_ops: AbstractTensor, vocab_size: int, device: Any = None) -> AbstractTensor: ...


class ImplicitBackpathScorer:
    """Scores candidate predecessor tokens by how well they explain a suffix."""

    def __init__(
        self,
        model_wrapper: AbstractModelWrapper,
        tokenizer: Any,
        writing_filter: Optional[TokenFilter] = None,
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
        left_context: Optional[List[int]] = None,
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

        ``left_context`` prepends fixed tokens before the candidate (e.g. a
        real GPT-2's own ``<|endoftext|>`` document-boundary token). Without
        it, every candidate is scored as the literal first token the model
        has ever seen -- a regime the model rarely saw cleanly during
        training, since training windows are usually mid-document, not
        document starts. This shifts every score but does not by itself
        make rare/specific candidates outscore common/generic ones -- that
        is a real property of single-token marginal scoring (it averages
        over every way a document could continue with the suffix, and
        frequent generic words accumulate more of that marginal mass than a
        word that's only right in one specific completion), not a bug.
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
            return self._score_batch(suffix_tokens, candidate_ids, left_context)

        chunks = []
        for start in range(0, N, max_batch_size):
            chunk_ids = candidate_ids[start : start + max_batch_size]
            chunks.append(self._score_batch(suffix_tokens, chunk_ids, left_context))
        return AbstractTensor.cat(chunks, dim=0)

    # ########## STUB: FluxGraph GPU Efficiency Pass ##########
    # PURPOSE: _score_batch (and FluxGraph._expand_batch/_grow_forward_word/
    #   _grow_backward_word in flux_graph.py) got their batching *shape*
    #   fixed already -- one padded/chunked model call per tick across every
    #   selected node, forward and backward together, instead of one call
    #   per node. The tensor *lifecycle* within that shape is still naive
    #   and is the next thing to fix.
    # EXPECTED BEHAVIOR: Fewer tensor allocations per tick (persistent/
    #   reused buffers instead of backend_cls.tensor(python_list, ...) built
    #   fresh every call), fewer host/device sync points (every .tolist()/
    #   .item() forces a GPU-to-CPU copy -- audit and batch/defer these),
    #   and no CPU->GPU->CPU->GPU round-tripping.
    # INPUTS: The current flux_graph.py/implicit_backpath.py pipeline
    #   (post word-level-nodes, auxin, head-pressure work); real GPT-2 for
    #   timing verification; the existing dummy-model tests in
    #   tests/test_flux_graph.py and tests/test_implicit_backpath.py for
    #   correctness regression (new tensor-lifecycle code must produce
    #   bit-identical results to the current, already-tested behavior --
    #   see test_expand_batch_matches_expand_backward_for_a_single_backward_node
    #   and its siblings for the established pattern of proving that).
    # OUTPUTS: Reduced wall-clock time per tick, particularly for backward
    #   expansion (currently the dominant cost). Regression tests proving
    #   the new code paths match old behavior exactly.
    # KEY ASSUMPTIONS/DEPENDENCIES: PyTorch backend is what actually matters
    #   for GPU behavior. Stay behind AbstractTensor's generic interface --
    #   do not hardcode a concrete backend class into flux_graph.py/
    #   implicit_backpath.py consumer code, an explicit repeated correction
    #   earlier in this project's history.
    # TODO:
    #   - Profile a real GPT-2 run first (python -m speaktome.demo_flux_graph
    #     --auto-dictionary --ticks N) before optimizing blind.
    #   - Design a persistent-buffer strategy for this method's
    #     batch_tokens/attention_mask (row counts/lengths vary call to
    #     call, so likely needs a max-size buffer reused with slicing).
    #   - Batch FluxGraph._grow_backward_word's per-beam score_candidates
    #     calls together the same way _expand_batch's round-0 backward pass
    #     already batches across nodes (documented as a deliberate,
    #     low-risk scope-limit when word growth was built -- worth
    #     revisiting now).
    #   - Audit every .tolist()/.item() call site in both files for whether
    #     it can be deferred or batched.
    # NOTES: Performance-only pass -- no new features, no changes to graph
    #   semantics. Pressure/auxin/head-pressure/word-growth behavior must
    #   stay bit-identical before and after. If a proposed change would
    #   alter what gets computed and not just how, stop and check with the
    #   user first -- this project has a strong established pattern of
    #   wanting to be consulted before architectural changes.
    # See also: AGENTS/experience_reports/1784144799_DOC_FluxGraph_Word_Level_And_Physics_Extensions.md
    # ###########################################################################
    def _score_batch(
        self,
        suffix_tokens: AbstractTensor,
        candidate_ids: AbstractTensor,
        left_context: Optional[List[int]] = None,
    ) -> AbstractTensor:
        """Score one batch of candidates in a single forward pass (see score_candidates)."""
        backend_cls = type(suffix_tokens)
        device = suffix_tokens.get_device()
        long_dtype = suffix_tokens.long_dtype

        N = candidate_ids.shape[0]
        L = suffix_tokens.shape[0]
        prefix = list(left_context) if left_context else []
        offset = len(prefix)

        suffix_list = suffix_tokens.tolist()
        candidate_list = candidate_ids.tolist()
        batch_rows = [prefix + [c] + suffix_list for c in candidate_list]
        row_len = offset + 1 + L

        batch_tokens = backend_cls.tensor(batch_rows, dtype=long_dtype, device=device)
        attention_mask = backend_cls.tensor(
            [[1] * row_len] * N, dtype=long_dtype, device=device
        )

        # AbstractModelWrapper.forward operates on raw backend tensors, not
        # AbstractTensor wrappers -- unwrap going in, rewrap coming out.
        outputs = self.model_wrapper.forward(
            input_ids=batch_tokens.data, attention_mask=attention_mask.data
        )
        logits = batch_tokens.ensure_tensor(outputs["logits"])  # [N, row_len, vocab]

        # Score every suffix position in one shot rather than L separate
        # log_softmax + index calls: log_softmax is taken once over the
        # whole [N, row_len, vocab] tensor, then every (position, target)
        # pair is gathered in a single fancy-indexed pass.
        # AbstractTensor.__getitem__ unwraps AbstractTensor indices before
        # delegating to the backend, so this paired gather stays generic --
        # no need to drop to a concrete backend.
        log_probs = logits.log_softmax(dim=-1)  # [N, row_len, vocab]
        positions = AbstractTensor.arange(
            offset, offset + L, device=device, dtype=long_dtype, cls=backend_cls
        )
        total = log_probs[:, positions, suffix_tokens].sum(dim=1)  # [N]

        # Each call allocates a [N, row_len, vocab] logits tensor whose
        # shape varies from call to call (row_len grows as the graph
        # grows) -- repeated varying-size allocate/free cycles fragment
        # CUDA's caching allocator until a later, larger request fails
        # even with nominally enough total free memory. Drop the large
        # intermediates and hand cached-but-unallocated blocks back before
        # returning, rather than letting fragmentation accumulate across
        # many FluxGraph ticks.
        del logits, log_probs, outputs, batch_tokens, attention_mask
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total
