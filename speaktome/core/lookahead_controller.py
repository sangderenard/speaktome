from __future__ import annotations

# Standard library imports
from typing import List, Tuple, Callable, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .beam_search_instruction import BeamSearchInstruction
from tensors import AbstractTensor
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
        tensor_ops: AbstractTensor | None,
        model_wrapper: AbstractModelWrapper,
    ):
        self.lookahead_steps = lookahead_steps
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        if tensor_ops is None:
            tensor_ops = AbstractTensor.get_tensor()
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
        backend_cls = type(self.tensor_ops)

        B_initial = prefix_tokens.shape[0]
        initial_prefix_width = prefix_tokens.shape[1]
        final_width = min(self.max_len, initial_prefix_width + self.lookahead_steps)

        # 1) Initialize current_* tensors by copying prefix into padded buffers
        current_tokens = self.tensor_ops.full(
            (B_initial, final_width),
            self.pad_id,
            dtype=prefix_tokens.get_dtype(),
            device=self.device,
            cls=backend_cls,
        )
        current_scores = self.tensor_ops.zeros(
            (B_initial, final_width),
            dtype=prefix_scores.get_dtype(),
            device=self.device,
            cls=backend_cls,
        )
        for i in range(B_initial):
            l = prefix_lengths[i].item()
            if l > 0:
                current_tokens.assign_at_indices(i, slice(0, l), prefix_tokens[i, :l])
                current_scores.assign_at_indices(i, slice(0, l), prefix_scores[i, :l])

        current_lengths = prefix_lengths.clone().to_device(self.device)
        current_parent_beam_idxs = original_parent_beam_idxs.clone().to_device(self.device)
        current_parent_prefix_lengths = prefix_lengths.clone().to_device(self.device)

        original_parents_set: Set[int] = set(original_parent_beam_idxs.tolist())

        for step in range(self.lookahead_steps):
            B_cur = current_tokens.shape[0]
            if B_cur == 0:
                break

            effective_input_width = int(current_lengths.max().item()) if B_cur > 0 else 0
            if effective_input_width == 0 and step == 0 and B_initial > 0:
                effective_input_width = 1
            if B_cur > 0:
                effective_input_width = max(1, effective_input_width)

            tokens_for_lm = current_tokens[:, :effective_input_width]
            attention_mask = tokens_for_lm.not_equal(self.pad_id).long_cast()

            # AbstractModelWrapper.forward operates on raw backend tensors
            # (e.g. PyTorchModelWrapper type-hints torch.Tensor), not
            # AbstractTensor wrappers -- unwrap going in, rewrap coming out.
            outputs_dict = self.model_wrapper.forward(
                input_ids=tokens_for_lm.data, attention_mask=attention_mask.data
            )
            logits = self.tensor_ops.ensure_tensor(outputs_dict['logits'])

            last_indices = (current_lengths - 1).clamp(min=0)
            last_logits = logits.select_by_indices(
                AbstractTensor.arange(0, B_cur, device=self.device, cls=backend_cls),
                last_indices,
            )

            logprobs = (last_logits / self.temp).log_softmax(dim=-1)
            topk_scores, topk_indices = AbstractTensor.topk(logprobs, k=self.top_k, dim=-1)

            num_parents = B_cur
            num_children = self.top_k
            N_total = num_parents * num_children

            expanded_parent_idxs = current_parent_beam_idxs.repeat_interleave(num_children)
            expanded_parent_prefix_lens = current_parent_prefix_lengths.repeat_interleave(num_children)

            next_tokens = current_tokens.repeat_interleave(num_children, dim=0)
            next_scores = current_scores.repeat_interleave(num_children, dim=0)
            next_lengths = current_lengths.repeat_interleave(num_children)

            row_idx = AbstractTensor.arange(0, N_total, device=self.device, cls=backend_cls)
            col_idx = next_lengths.clone()

            flat_new_ids = topk_indices.view_flat()
            flat_new_scores = topk_scores.view_flat()

            can_append = col_idx.less(final_width)

            next_tokens.assign_at_indices(
                row_idx.boolean_mask_select(can_append),
                col_idx.boolean_mask_select(can_append),
                flat_new_ids.boolean_mask_select(can_append),
            )
            next_scores.assign_at_indices(
                row_idx.boolean_mask_select(can_append),
                col_idx.boolean_mask_select(can_append),
                flat_new_scores.boolean_mask_select(can_append),
            )
            next_lengths.increment_at_indices(can_append)
            next_lengths = next_lengths.clamp(max=final_width)

            candidate_aggregate_scores = self.aggregate_fn(next_scores)

            if N_total > self.top_k:
                _, keep_indices = AbstractTensor.topk(candidate_aggregate_scores, k=self.top_k, dim=0)
            else:
                keep_indices = AbstractTensor.arange(0, N_total, device=self.device, cls=backend_cls)

            current_tokens = next_tokens[keep_indices]
            current_scores = next_scores[keep_indices]
            current_lengths = next_lengths[keep_indices]
            current_parent_beam_idxs = expanded_parent_idxs[keep_indices]
            current_parent_prefix_lengths = expanded_parent_prefix_lens[keep_indices]

        final_parents: Set[int] = set(current_parent_beam_idxs.tolist())
        pruned_original_parents = list(original_parents_set - final_parents)

        return (
            current_tokens,
            current_scores,
            current_lengths,
            current_parent_beam_idxs,
            current_parent_prefix_lengths,
            pruned_original_parents
        )
