# Standard library imports
from __future__ import annotations
from typing import List, Tuple, Callable, Any, Set, TYPE_CHECKING

# Local application/library specific imports
if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .beam_search_instruction import BeamSearchInstruction
from ..tensors import AbstractTensorOperations
from .model_abstraction import AbstractModelWrapper
# --- END HEADER ---


class LookaheadConfig:
    """
    Bundles all lookahead hyperparameters:
      - instruction: BeamSearchInstruction (contains any custom rules)
      - lookahead_top_k: how many candidates to keep *after* each expansion step
      - lookahead_temp: temperature for softmax during lookahead
      - aggregate_fn: function mapping [N, width]→[N] to score each candidate
    """
    def __init__(
        self,
        instruction: BeamSearchInstruction | None, # Can be None for simpler setups
        lookahead_top_k: int,
        lookahead_temp: float,
        aggregate_fn: Callable[[Any], Any] # Any for tensor type
    ):
        self.instruction = instruction
        self.lookahead_top_k = lookahead_top_k
        self.lookahead_temp = lookahead_temp
        self.aggregate_fn = aggregate_fn


class LookaheadController:
    def __init__(
        self,
        lookahead_steps: int,
        max_len: int,
        device: Any, # Generic device type
        tokenizer: Any, # Tokenizer should have pad_token_id
        config: LookaheadConfig,
        tensor_ops: AbstractTensorOperations,
        model_wrapper: AbstractModelWrapper,
    ):
        self.lookahead_steps = lookahead_steps
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        self.tensor_ops = tensor_ops
        self.model_wrapper = model_wrapper

        # Unpack from config
        self.config = config
        self.top_k = config.lookahead_top_k
        self.temp = config.lookahead_temp
        self.aggregate_fn = config.aggregate_fn

        # pad token ID fallback
        self.pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0

    def run(
        self,
        prefix_tokens: Any,     # [B, prefix_width]
        prefix_scores: Any,    # [B, prefix_width]
        prefix_lengths: Any,    # [B]
        original_parent_beam_idxs: Any,  # [B]
    ) -> Tuple[
        Any,   # final_tokens: [K, final_width]
        Any,  # final_scores: [K, final_width]
        Any,   # final_lengths: [K]
        Any,   # final_parent_beam_idxs: [K]
        Any,   # final_parent_prefix_lengths: [K]
        List[int]           # pruned_original_parent_beam_idxs
    ]:
        B_initial = self.tensor_ops.shape(prefix_tokens)[0]
        initial_prefix_width = self.tensor_ops.shape(prefix_tokens)[1]
        final_width = min(self.max_len, initial_prefix_width + self.lookahead_steps)

        # 1) Initialize current_* tensors by copying prefix into padded buffers
        current_tokens = self.tensor_ops.full(
            (B_initial, final_width),
            self.pad_id,
            dtype=self.tensor_ops.get_dtype(prefix_tokens),
            device=self.device
        )
        current_scores = self.tensor_ops.zeros(
            (B_initial, final_width),
            dtype=self.tensor_ops.get_dtype(prefix_scores),
            device=self.device
        )
        for i in range(B_initial):
            l = self.tensor_ops.item(prefix_lengths[i])
            if l > 0:
                self.tensor_ops.assign_at_indices(current_tokens, i, slice(0, l), prefix_tokens[i, :l])
                self.tensor_ops.assign_at_indices(current_scores, i, slice(0, l), prefix_scores[i, :l])

        current_lengths = self.tensor_ops.to_device(self.tensor_ops.clone(prefix_lengths), self.device)
        current_parent_beam_idxs = self.tensor_ops.to_device(self.tensor_ops.clone(original_parent_beam_idxs), self.device)
        current_parent_prefix_lengths = self.tensor_ops.to_device(self.tensor_ops.clone(prefix_lengths), self.device)

        original_parents_set: Set[int] = set(self.tensor_ops.tolist(original_parent_beam_idxs))

        for step in range(self.lookahead_steps):
            B_cur = self.tensor_ops.shape(current_tokens)[0]
            if B_cur == 0:
                break

            effective_input_width = int(self.tensor_ops.item(self.tensor_ops.max(current_lengths))) if B_cur > 0 else 0
            if effective_input_width == 0 and step == 0 and B_initial > 0:
                effective_input_width = 1
            if B_cur > 0:
                effective_input_width = max(1, effective_input_width)

            tokens_for_lm = current_tokens[:, :effective_input_width]
            attention_mask = self.tensor_ops.long_cast(
                self.tensor_ops.not_equal(tokens_for_lm, self.pad_id)
            )

            outputs_dict = self.model_wrapper.forward(
                input_ids=tokens_for_lm, attention_mask=attention_mask
            )
            logits = outputs_dict['logits']

            last_indices = self.tensor_ops.clamp(
                self.tensor_ops._apply_operator("sub", current_lengths, 1), min_val=0
            )
            last_logits = self.tensor_ops.select_by_indices(
                logits,
                self.tensor_ops.arange(0, B_cur, device=self.device),
                last_indices,
            )

            logprobs = self.tensor_ops.log_softmax(
                self.tensor_ops._apply_operator("truediv", last_logits, self.temp), dim=-1
            )
            topk_scores, topk_indices = self.tensor_ops.topk(logprobs, k=self.top_k, dim=-1)

            num_parents = B_cur
            num_children = self.top_k
            N_total = num_parents * num_children

            expanded_parent_idxs = self.tensor_ops.repeat_interleave(current_parent_beam_idxs, num_children)
            expanded_parent_prefix_lens = self.tensor_ops.repeat_interleave(current_parent_prefix_lengths, num_children)

            next_tokens = self.tensor_ops.repeat_interleave(current_tokens, num_children, dim=0)
            next_scores = self.tensor_ops.repeat_interleave(current_scores, num_children, dim=0)
            next_lengths = self.tensor_ops.repeat_interleave(current_lengths, num_children)

            row_idx = self.tensor_ops.arange(0, N_total, device=self.device)
            col_idx = self.tensor_ops.clone(next_lengths)

            flat_new_ids = self.tensor_ops.view_flat(topk_indices)
            flat_new_scores = self.tensor_ops.view_flat(topk_scores)

            can_append = self.tensor_ops.less(col_idx, final_width)

            self.tensor_ops.assign_at_indices(
                next_tokens,
                self.tensor_ops.boolean_mask_select(row_idx, can_append),
                self.tensor_ops.boolean_mask_select(col_idx, can_append),
                self.tensor_ops.boolean_mask_select(flat_new_ids, can_append),
            )
            self.tensor_ops.assign_at_indices(
                next_scores,
                self.tensor_ops.boolean_mask_select(row_idx, can_append),
                self.tensor_ops.boolean_mask_select(col_idx, can_append),
                self.tensor_ops.boolean_mask_select(flat_new_scores, can_append),
            )
            self.tensor_ops.increment_at_indices(next_lengths, can_append)
            next_lengths = self.tensor_ops.clamp(next_lengths, max_val=final_width)

            candidate_aggregate_scores = self.aggregate_fn(next_scores)

            if N_total > self.top_k:
                _, keep_indices = self.tensor_ops.topk(candidate_aggregate_scores, k=self.top_k, dim=0)
            else:
                keep_indices = self.tensor_ops.arange(0, N_total, device=self.device)

            current_tokens = self.tensor_ops.index_select(next_tokens, 0, keep_indices)
            current_scores = self.tensor_ops.index_select(next_scores, 0, keep_indices)
            current_lengths = self.tensor_ops.index_select(next_lengths, 0, keep_indices)
            current_parent_beam_idxs = self.tensor_ops.index_select(expanded_parent_idxs, 0, keep_indices)
            current_parent_prefix_lengths = self.tensor_ops.index_select(expanded_parent_prefix_lens, 0, keep_indices)

        final_parents: Set[int] = set(self.tensor_ops.tolist(current_parent_beam_idxs))
        pruned_original_parents = list(original_parents_set - final_parents)

        return (
            current_tokens,
            current_scores,
            current_lengths,
            current_parent_beam_idxs,
            current_parent_prefix_lengths,
            pruned_original_parents
        )
