# Standard library imports
from __future__ import annotations
from typing import List, Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
import math

    # Third-party imports
    if TYPE_CHECKING:  # pragma: no cover - type hints only
        import torch

    from tensors.faculty import Faculty

    FACULTY_REQUIREMENT = Faculty.TORCH

    # Local application/library specific imports
    # Please adjust these import paths based on your actual project structure.
    from .. import config
    from ..config import GPU_LIMIT, LENGTH_LIMIT
    from .beam_graph_operator import BeamGraphOperator
    from .beam_search_instruction import BeamSearchInstruction
    from .scorer import Scorer
    from .beam_retirement_manager import BeamRetirementManager
    from .compressed_beam_tree import CompressedBeamTree
    from tensors import (
        AbstractTensor,
    )
    from .model_abstraction import (
        AbstractModelWrapper,
        PyTorchModelWrapper,
    )
    from .lookahead_controller import LookaheadConfig, LookaheadController
# --- END HEADER ---

class BeamSearch:
    """Core orchestrator for beam expansion and pruning."""
    def __init__(self, scorer: Scorer, beam_width: int = 5, gpu_limit: int = GPU_LIMIT, 
                 lookahead_steps: int = 1, initial_retirement_enabled: bool = True,
                 device=config.DEVICE, verbose=True, max_len=LENGTH_LIMIT, # General params
                 # Lookahead specific initial/default configurations
                 initial_lookahead_rules: Optional[BeamSearchInstruction] = None,
                 initial_aggregate_score_fn: Optional[Callable[[Any], Any]] = None,
                 max_candidates_per_lookahead_step: Optional[int] = None # This is lookahead_top_k
    ):
        self.scorer = scorer
        self.tensor_ops: AbstractTensor = scorer.tensor_ops
        self.max_len = max_len
        self.beam_width = beam_width
        self.gpu_limit = gpu_limit
        self.device = device
        self.initial_retirement_enabled = initial_retirement_enabled
        self.lookahead_steps = max(1, lookahead_steps) # Ensure at least 1 step
        self.verbose = verbose

        # Store initial/default lookahead configurations
        self.current_lookahead_rules: Optional[BeamSearchInstruction] = initial_lookahead_rules
        self.current_lookahead_aggregate_fn: Optional[Callable[[Any], Any]] = initial_aggregate_score_fn
        # This will be used as lookahead_top_k in LookaheadConfig
        self.max_candidates_per_lookahead_step = max_candidates_per_lookahead_step if max_candidates_per_lookahead_step is not None else beam_width 
        
        # ─── Auxin‐inspired thresholds ───
        self.DELTA_GROWTH_GAP      = 0.05   # absolute “bud if internal > child_sum + gap”
        self.ALPHA_OVERSHOOT_FACTOR = 1.2   # if child_sum > α * parent_score → suppress
        self.K_DEEPEN_LEAVES       = beam_width  # expand up to this many leaves each GNN pass

        # ─── NO MORE STATIC bins_config HERE ───

        self.tree = CompressedBeamTree(device=device, tokenizer=scorer.tokenizer)
        self.graph_op = BeamGraphOperator(self.tree, tensor_ops=self.tensor_ops)
 
        # Defaults, but will be overridden by each instruction
        self.pre_top_k  = self.scorer.default_pre_top_k
        self.pre_temp   = self.scorer.default_pre_temp
        self.cull_after = self.scorer.default_cull_after

        self.dead_end_indices = []
        self.retirement_manager = None
        self.active_leaf_indices = []

        # ←— “current_instruction” slot (initially None)
        self.current_instruction: BeamSearchInstruction = None

    def expand_internal_as_leaf(self, internal_node_pyg_id: int) -> Tuple[bool, int]:
        """
        EXPAND “bud” at an internal PyG node: attach new children under that node,
        without duplicating the entire prefix. Returns (success, total_active_leaves).
        """
        if self.verbose:
            print(f"[BeamSearch] BUDDING at internal PyG node {internal_node_pyg_id}")

        # 1) Look up the CompressedBeamTree node_idx corresponding to this PyG ID
        original_idx = self.tree.pyg_id_to_node_idx.get(internal_node_pyg_id)
        if original_idx is None:
            if self.verbose:
                print(f"  [Error] PyG node {internal_node_pyg_id} not found in compressed-tree map.")
            return False, len(self.active_leaf_indices)

        # 2) Ask the tree to extend from that original_idx, generating beam_width new children
        new_leaves = self.tree.extend_from_node(original_idx, num_children=self.beam_width)
        if not new_leaves:
            if self.verbose:
                print(f"  [Error] No new leaves returned when budding at node_idx {original_idx}.")
            return False, len(self.active_leaf_indices)

        # 3) Add the newly‐created leaves into active_leaf_indices
        for leaf_idx in new_leaves:
            if leaf_idx not in self.active_leaf_indices:
                self.active_leaf_indices.append(leaf_idx)

        if self.verbose:
           print(f"  [Success] Budded {len(new_leaves)} new leaves. Now {len(self.active_leaf_indices)} active leaves.")
        return True, len(self.active_leaf_indices)

    def apply_instruction(self, instr: BeamSearchInstruction):
        """
        1) Pull pre_top_k, pre_temp, cull_after
        2) Rebuild bins_config‐dict from instr.score_bins
        3) Initialise scorer-managed bins
        4) Store instr in self.current_instruction
        """
        # 1. Update pre-sampling & cull settings
        self.pre_temp   = instr.pre_temp
        self.pre_top_k  = instr.pre_top_k
        self.cull_after = instr.cull_after
        self.lookahead_steps = instr.lookahead_steps # Apply lookahead from instruction
        
        # Update lookahead-specific configurations from the new instruction
        self.current_lookahead_rules = instr # The whole instruction can serve as rules for lookahead
        self.current_lookahead_aggregate_fn = instr.lookahead_aggregate_fn

        # 2. Turn instr.score_bins (List[(fn, width, temp)]) → bins_dict
        bins_dict = {}
        for i, (fn, width, temp) in enumerate(instr.score_bins):
            name = f"bin_{i}_{fn.__name__}"
            bins_dict[name] = {
                'fn': fn,
                'width': width,
                # we’ll store temp as part of bin’s “params” so that _select_best can read it
                'params': {'temp': temp}
            }

        # 3. Configure scorer bins from scratch
        self.scorer.init_bins(
            bins_dict,
            max_len=self.max_len,
            device=self.device,
            cull_after=instr.cull_after,
        )

        # 4. Remember which instruction we’re using right now
        self.current_instruction = instr

        if self.verbose:
            print(f"[BeamSearch] Applied new instruction: {instr} | {instr.get_lookahead_config_summary()}")

    def get_all_beams_summary(self, n: int = 20) -> List[Tuple[int, str, float, str]]:
        """
        Retrieves a summary of all leaf beams in the tree, indicating their status.
        """
        all_leaf_beam_indices = list(self.tree.leaf_node_indices.keys())
        
        beam_details = []
        for beam_idx in all_leaf_beam_indices:
            tokens, scores, length = self.tree.get_beam_tensors_by_beam_idx(beam_idx, max_len=self.max_len, read_only=True)
            if length > 0: # Ensure length is a scalar for comparison
                # Use mean_logprob_score for ranking
                # Ensure tensors are correctly shaped for mean_logprob_score:
                # beams: [1, L], scores: [1, L], lengths: [1]
                current_beam_tokens = tokens[:length].unsqueeze(0)
                current_beam_scores = scores[:length].unsqueeze(0)
                current_beam_length_tensor = self.tensor_ops.tensor_from_list(
                    [length],
                    dtype=self.tensor_ops.long_dtype,
                    device=self.device,
                )

                path_score = self.scorer.default_scorer(beams=current_beam_tokens, scores=current_beam_scores, lengths=current_beam_length_tensor, tokenizer=self.scorer.tokenizer).item()
                path_text = self.scorer.tokenizer.decode(tokens[:length].tolist(), skip_special_tokens=True)
                status = "Unknown"
                if beam_idx in self.active_leaf_indices:
                    status = "Active"
                elif self.retirement_manager and beam_idx in self.retirement_manager._bucket.get(self.retirement_manager._prefix_hash(beam_idx), []):
                    status = "Retired"
                beam_details.append((beam_idx, path_text, path_score, status))
        
        beam_details.sort(key=lambda x: x[2], reverse=True) # Sort by score
        return beam_details[:n]

    def get_all_beams_full_summary(self) -> List[Tuple[int, str, float, str]]:
        """
        Retrieves a summary of ALL leaf beams in the tree, indicating their status, without truncation.
        """
        all_leaf_beam_indices = list(self.tree.leaf_node_indices.keys())
        
        beam_details = []
        for beam_idx in all_leaf_beam_indices:
            tokens, scores, length = self.tree.get_beam_tensors_by_beam_idx(beam_idx, max_len=self.max_len, read_only=True)
            if length > 0:
                current_beam_tokens = tokens[:length].unsqueeze(0)
                current_beam_scores = scores[:length].unsqueeze(0)
                current_beam_length_tensor = self.tensor_ops.tensor_from_list(
                    [length],
                    dtype=self.tensor_ops.long_dtype,
                    device=self.device,
                )  # Ensure device consistency

                path_score = self.scorer.default_scorer(beams=current_beam_tokens, scores=current_beam_scores, lengths=current_beam_length_tensor, tokenizer=self.scorer.tokenizer).item()
                path_text = self.scorer.tokenizer.decode(tokens[:length].tolist(), skip_special_tokens=True)
                status = "Unknown"
                if beam_idx in self.active_leaf_indices: status = "Active"
                elif self.retirement_manager and beam_idx in self.retirement_manager._bucket.get(self.retirement_manager._prefix_hash(beam_idx), []): status = "Retired"
                beam_details.append((beam_idx, path_text, path_score, status))
        
        beam_details.sort(key=lambda x: x[2], reverse=True) # Sort by score
        return beam_details # Return all details
    def get_top_retired_beams(self, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Retrieves the top N retired beams based on their scores.
        """
        if not self.retirement_manager or not hasattr(self.retirement_manager, '_bucket'):
            print("[Warning] Retirement manager not available or not initialized correctly.")
            return []

        all_retired_beam_indices = set()
        with self.retirement_manager._lock: # Ensure thread-safe access to _bucket
            for beam_idx_list in self.retirement_manager._bucket.values():
                all_retired_beam_indices.update(beam_idx_list)

        if not all_retired_beam_indices:
            return []

        scored_retired_beams = []
        for beam_idx in all_retired_beam_indices:
            tokens, scores, length = self.tree.get_beam_tensors_by_beam_idx(beam_idx, max_len=self.max_len)
            if length > 0:
                # Use sum of logprobs as the score for ranking
                path_score = scores[:length].sum().item()
                path_text = self.scorer.tokenizer.decode(tokens[:length], skip_special_tokens=True)
                scored_retired_beams.append((beam_idx, path_text, path_score))

        # Sort by score in descending order (higher score is better)
        scored_retired_beams.sort(key=lambda x: x[2], reverse=True)
        return scored_retired_beams[:n]


    def expand_specific_leaf_once(self, target_beam_idx: int) -> Tuple[bool, Optional[int]]:
        """
        Expands a single specified leaf beam.
        Returns (success: bool, new_active_leaf_count: Optional[int])
        Updates self.active_leaf_indices and self.dead_end_indices internally.
        """
        if target_beam_idx not in self.active_leaf_indices:
            # Try to recover it if it's a valid leaf
            if target_beam_idx in self.tree.leaf_node_indices:
                try:
                    self.graph_op.move_paths_to_device(
                        [self.tree.leaf_node_indices[target_beam_idx]],
                        device=self.device
                    )
                    self.active_leaf_indices.append(target_beam_idx)
                    if self.verbose:
                        print(f"[Recovery] Beam {target_beam_idx} elevated from retirement to active.")
                except Exception as e:
                    print(f"[ERROR] Could not move beam {target_beam_idx} to GPU: {e}")
                    return False, len(self.active_leaf_indices)
            else:
                print(f"[ERROR] Beam {target_beam_idx} no longer exists in tree. Cannot expand.")
                return False, len(self.active_leaf_indices)


        # Temporarily make only the target beam active for _expand_once
        # Store other active leaves to restore them later if needed, or manage active_leaf_indices carefully
        
        # Perform expansion
        kept_new_leaves, retired_new_leaves = self._expand_once([target_beam_idx])

        # The target_beam_idx is no longer a leaf itself, so remove it from active_leaf_indices
        if target_beam_idx in self.active_leaf_indices:
            self.active_leaf_indices.remove(target_beam_idx)

        # Add newly created and kept leaves to active_leaf_indices
        for leaf_idx in kept_new_leaves:
            if leaf_idx not in self.active_leaf_indices: # Avoid duplicates if any logic error
                self.active_leaf_indices.append(leaf_idx)
        
        # Handle retired new leaves (send to retirement manager)
        if retired_new_leaves:
            self.retire_leaves(retired_new_leaves)

        # Dead-ending is handled within _expand_once by moving to self.dead_end_indices
        # and offloading to CPU.

        if self.verbose:
            print(f"Expanded target beam {target_beam_idx}. New active leaves from this expansion: {len(kept_new_leaves)}. Total active: {len(self.active_leaf_indices)}")
        
        return True, len(self.active_leaf_indices)

    def get_active_leaves_pyg_nodes(self) -> List[int]:
        """Returns the PyG node IDs of the current active leaf beams."""
        pyg_node_ids = []
        for beam_idx in self.active_leaf_indices:
            original_node_idx = self.tree.leaf_node_indices.get(beam_idx)
            if original_node_idx is not None:
                pyg_id = self.tree.node_idx_to_pyg_id.get(original_node_idx)
                if pyg_id is not None:
                    pyg_node_ids.append(pyg_id)
        return pyg_node_ids




    def _select_best(self, beams, scores, lengths, tokenizer, **kwargs):
        # Make sure an instruction is in place
        if self.current_instruction is None:
            raise RuntimeError("No BeamSearchInstruction applied. Call apply_instruction(...) first.")

        # Update scorer-managed bins
        self.scorer.update_bins(beams, scores, lengths, tokenizer, **kwargs)

        bin_names   = list(self.scorer.bins.keys())
        score_matrix = self.tensor_ops.stack(
            [self.scorer.bins[name]['scores'] for name in bin_names], dim=0
        )
        num_candidates = score_matrix.shape()[1]
        k = min(self.gpu_limit, num_candidates)
        top_scores, top_idx = self.tensor_ops.topk(score_matrix, k=k, dim=1)

        if self.verbose:
            self.scorer.print_bins(tokenizer)

        return top_idx, score_matrix

    
    def _expand_once(self, active_leaf_beam_indices: List[int]) -> Tuple[List[int], List[int]]:
        tokenizer = self.scorer.tokenizer
        model     = self.scorer.model

        # Construct LookaheadConfig
        # Use current_lookahead_rules or a default if None
        lookahead_instr_for_config = self.current_lookahead_rules
        if lookahead_instr_for_config is None: # Create a default instruction if none is set
            lookahead_instr_for_config = BeamSearchInstruction(node_id=-1, action="lookahead_default", scorer=self.scorer)
        
        # Use current_lookahead_aggregate_fn or a default (e.g., RMS, handled by LookaheadController if None)
        # For LookaheadConfig, aggregate_fn is mandatory. Let's define a default RMS here if not provided.
        agg_fn_for_config = self.current_lookahead_aggregate_fn
        tensor_ops_instance: AbstractTensor = self.tensor_ops
        model_wrapper_instance = PyTorchModelWrapper(model)
        if agg_fn_for_config is None:
            def default_rms_aggregate_fn(score_matrix: Any) -> Any:
                clamped = tensor_ops_instance.clamp(score_matrix, min_val=-1e9)
                return tensor_ops_instance.sqrt(
                    tensor_ops_instance.mean(tensor_ops_instance.pow(clamped, 2), dim=0)
                )
            agg_fn_for_config = default_rms_aggregate_fn

        lookahead_config_obj = LookaheadConfig(
            instruction=lookahead_instr_for_config,
            lookahead_top_k=self.max_candidates_per_lookahead_step,
            lookahead_temp=self.pre_temp,
            aggregate_fn=agg_fn_for_config,
        )
        lookahead_controller = LookaheadController(
            self.lookahead_steps,
            self.max_len,
            self.device,
            tokenizer,
            lookahead_config_obj,
            tensor_ops_instance,
            model_wrapper_instance,
        )

        if not active_leaf_beam_indices:
            return [], []

        # ───────────────────────────────────────────────────────────
        # 1) Move all active leaf paths onto GPU so we can read them
        # ───────────────────────────────────────────────────────────
        active_leaf_node_indices = [
            self.tree.leaf_node_indices[idx]
            for idx in active_leaf_beam_indices
            if idx in self.tree.leaf_node_indices
        ]
        if active_leaf_node_indices:
            self.graph_op.move_paths_to_device(active_leaf_node_indices, device=self.device)

        # ───────────────────────────────────────────────────────────
        # 2) Fetch “prefix” tokens + scores + lengths from the tree
        # ───────────────────────────────────────────────────────────
        #    initial_beams_tokens:  [B, W0], where W0 = max prefix length among these beams
        #    initial_beams_scores:  [B, W0]
        #    initial_parent_tree_lengths: [B]  (the “true” prefix length for each row)
        initial_beams_tokens, initial_beams_scores, initial_parent_tree_lengths = (
            self.tree.get_batch_by_beam_indices(active_leaf_beam_indices, max_len=self.max_len)
        )
        if initial_beams_tokens.numel() == 0:
            return [], []

        # Keep track of which “beam_idx” (from the tree) each candidate ultimately came from:
        # This is passed to LookaheadController
        original_parent_beam_idxs_for_lookahead = self.tensor_ops.tensor_from_list(
            active_leaf_beam_indices,
            dtype=self.tensor_ops.long_dtype,
            device=self.device,
        )

        # Run the lookahead process
        (
            final_lookahead_tokens,             # [M, final_width]
            final_lookahead_scores,             # [M, final_width]
            final_lookahead_lengths,            # [M]
            final_lookahead_parent_beam_idxs,   # [M] (original parent beam_idx for each survivor)
            final_lookahead_parent_prefix_lengths, # [M] (original prefix_length for each survivor)
            pruned_original_parent_beam_idxs_to_retire # List[int]
        ) = lookahead_controller.run(
            prefix_tokens=initial_beams_tokens,
            prefix_scores=initial_beams_scores,
            prefix_lengths=initial_parent_tree_lengths,
            original_parent_beam_idxs=original_parent_beam_idxs_for_lookahead,
        )

        if self.verbose:
            print(f"Lookahead finished. {final_lookahead_tokens.size(0)} candidates returned. {len(pruned_original_parent_beam_idxs_to_retire)} original parents identified for potential retirement from internal lookahead pruning.")

        # Retire beams whose lookahead paths were all pruned internally by LookaheadController
        if pruned_original_parent_beam_idxs_to_retire:
            # These are original parent beam_idxs. They are no longer active leaves if all their children died in lookahead.
            # The actual beam_idx of the *pruned lookahead path itself* is not created in the tree yet.
            # So, we retire the *original parent* if it's in this list.
            # However, the parent might have other children that survived or were not part of this specific _expand_once call.
            # The `pruned_original_parent_beam_idxs_to_retire` list from LookaheadController
            # contains original parent beam_idxs whose *all* lookahead children got pruned *within that controller's run*.
            # These original parents should be moved to dead_end_indices or retired.
            # For now, let's assume these are effectively dead ends from this expansion.
            # The `retire_leaves` function expects beam_indices of *newly created leaves* that are to be retired.
            # This is different. These are *parents* whose expansions failed.
            
            # For now, we will handle the retirement of *newly created leaves* that don't make the final cut *after* _select_best.
            # The `pruned_original_parent_beam_idxs_to_retire` indicates that these original parents
            # did not yield any surviving lookahead paths. They should be removed from active_leaf_indices
            # and potentially marked as dead_ends if not handled by the main logic.
            # This interaction needs careful thought. The current `retire_leaves` is for *new* leaves.
            # Let's assume for now that if an original parent is in `pruned_original_parent_beam_idxs_to_retire`,
            # it means it effectively became a dead-end in this expansion.
            
            # The `active_leaf_beam_indices` passed to `_expand_once` are the parents.
            # If a parent from `active_leaf_beam_indices` is in `pruned_original_parent_beam_idxs_to_retire`,
            # it means none of its lookahead children survived.
            # These parents should be removed from `self.active_leaf_indices` later if they are not updated with new children.
            # The main logic of updating `active_leaf_indices` will handle this: parents are removed, new children are added.
            # If a parent has no new children added, it's implicitly no longer an active leaf.
            # Parents with no surviving lookahead children should be removed
            # from ``self.active_leaf_indices``.  The
            # ``pruned_original_parent_beam_idxs_to_retire`` list contains these
            # failed parents.
            failed_parents = [
                idx for idx in pruned_original_parent_beam_idxs_to_retire
                if idx in self.tree.leaf_node_indices
            ]
            if failed_parents:
                # Remove failed parents from active leaves
                self.active_leaf_indices = [
                    idx for idx in self.active_leaf_indices
                    if idx not in failed_parents
                ]
                if self.retirement_enabled:
                    # Offload failed parents to the retirement manager so they
                    # do not linger in GPU memory.
                    self.retire_leaves(failed_parents)
                else:
                    # Fast mode: simply mark them as dead ends.
                    self.dead_end_indices.extend(failed_parents)
                if self.verbose:
                    print(
                        f"[Lookahead] Retired {len(failed_parents)} parent beams with no surviving children."
                    )

        if final_lookahead_tokens.numel() == 0:
            if self.verbose:
                print("No candidates survived lookahead → returning empty")
            return [], []

        # ──────────────────────────────────────────────────────────────────────────────────────────
        # 6) Now run the scorer's full bin selection on these “final lookahead” M candidates
        # ──────────────────────────────────────────────────────────────────────────────────────────
        topk_idx_from_meta, score_matrix_from_meta = self._select_best(
            final_lookahead_tokens,
            final_lookahead_scores,
            final_lookahead_lengths,
            tokenizer
        )

        # Consolidate “one candidate per bin up to bin_width” again to get our final keep‐list
        num_bins    = topk_idx_from_meta.size(0)
        M_from_lookahead = final_lookahead_tokens.size(0) # Number of candidates from lookahead
        keep_final  = []
        seen_final  = set()
        bin_counts  = [0] * num_bins
        rank_in_bin = 0
        bin_names   = list(self.scorer.bins.keys())

        while (rank_in_bin < topk_idx_from_meta.size(1)) and any(
            bin_counts[b] < self.scorer.bins[bin_names[b]]['width']
            for b in range(num_bins)
        ):
            for b in range(num_bins):
                if bin_counts[b] >= self.scorer.bins[bin_names[b]]['width']:
                    continue
                idx_cand = topk_idx_from_meta[b, rank_in_bin].item()
                if idx_cand not in seen_final:
                    keep_final.append(idx_cand)
                    seen_final.add(idx_cand)
                    bin_counts[b] += 1
            rank_in_bin += 1

        final_idx_tensor = self.tensor_ops.tensor_from_list(
            keep_final,
            dtype=self.tensor_ops.long_dtype,
            device=self.device,
        )

        # ──────────────────────────────────────────────────────────────────────────────────────────
        # 7) “Hard‐cut” vs “Retired” logic exactly as before, but now we call
        #    extend_leaves_batch_lookahead(...) once with:
        #      * final_selected_beams_tokens: rows from current_beams_tokens[final_idx_tensor]
        #      * final_selected_beams_scores
        #      * final_selected_beams_lengths
        #      * final_selected_parent_idxs
        #      * final_selected_parent_prefix_lengths
        # ──────────────────────────────────────────────────────────────────────────────────────────
        if not self.retirement_enabled:
            if final_idx_tensor.numel() > 0:
                # Pick ONLY the “kept” rows
                sel_tokens   = final_lookahead_tokens[final_idx_tensor]
                sel_scores   = final_lookahead_scores[final_idx_tensor]
                sel_lengths  = final_lookahead_lengths[final_idx_tensor]
                sel_parents  = final_lookahead_parent_beam_idxs[final_idx_tensor]
                sel_parents_plen = final_lookahead_parent_prefix_lengths[final_idx_tensor]

                new_kept_beam_idxs = self.tree.extend_leaves_batch_lookahead(
                    sel_parents,
                    sel_tokens,
                    sel_scores,
                    sel_lengths,
                    initial_lengths_of_parents=sel_parents_plen
                )
                kept_final = new_kept_beam_idxs.tolist()
            else:
                kept_final = []
            retired_final = []
        else:
            # The “retirement enabled” path: we have to pass *all* M lookahead candidates in,
            # then mask out which ones are “kept” vs “retired.”
            # M_from_lookahead is the total number of candidates from lookahead controller
            mask_keep = self.tensor_ops.zeros(
                (M_from_lookahead,),
                dtype=self.tensor_ops.bool_dtype,
                device=self.device,
            )
            if final_idx_tensor.numel() > 0:
                mask_keep[final_idx_tensor] = True

            all_new_idxs = self.tree.extend_leaves_batch_lookahead(
                final_lookahead_parent_beam_idxs,    # Parent beam_idxs for each of the M candidates
                final_lookahead_tokens,              # Tokens for each of the M candidates
                final_lookahead_scores,              # Scores for each of the M candidates
                final_lookahead_lengths,             # Lengths for each of the M candidates
                initial_lengths_of_parents=final_lookahead_parent_prefix_lengths # Original prefix lengths
            )
            # Convert to Python lists:
            kept_final    = all_new_idxs[mask_keep.cpu()].tolist()
            retired_final = all_new_idxs[~mask_keep.cpu()].tolist()

        return kept_final, retired_final

    # --- Start: Refactored BeamSearch methods ---
    def initialize_search(self, seed_text: str):
        """Initializes the beam tree with a seed sequence."""
        tokenizer = self.scorer.tokenizer
        seed_tokens_list = tokenizer.encode(seed_text)
        seed_scores_list = [0.0] * len(seed_tokens_list)
        
        initial_leaf_beam_idx = self.tree.add_root_beam(seed_tokens_list, seed_scores_list)
        self.active_leaf_indices = [initial_leaf_beam_idx]
        self.dead_end_indices = [] # Reset for each search
        
        # Use the initial_retirement_enabled state set during __init__
        self.retirement_enabled = self.initial_retirement_enabled 

        if self.retirement_enabled:
            self.retirement_manager = BeamRetirementManager(tree=self.tree, prefix_len=8, tokenizer=tokenizer)
            if self.verbose: print("Retirement manager initialized.")
        else:
            self.retirement_manager = None
            if self.verbose: print("Retirement is disabled for this search.")
        if self.verbose:
            print(f"Search initialized with seed. Active beams: {len(self.active_leaf_indices)}")

    def expand_and_score_once(self):
        """Performs one step of beam expansion (possibly with lookahead) and scoring."""
        if not hasattr(self, 'active_leaf_indices') or not self.active_leaf_indices:
            if self.verbose: print("No active beams to expand.")
            return

        kept_leaves, retired_leaves = self._expand_once(self.active_leaf_indices)
        self.active_leaf_indices = kept_leaves

        if retired_leaves:
            self.retire_leaves(retired_leaves)
        
        if self.verbose:
            print(f"Expansion complete. Active beams: {len(self.active_leaf_indices)}, Retired this step: {len(retired_leaves)}")

    def promote_if_needed(self):
        """Promotes beams from retirement manager if active beams are below GPU limit."""
        if not hasattr(self, 'retirement_manager') or not hasattr(self, 'active_leaf_indices'):
            if self.verbose: print("Retirement manager or active_leaf_indices not initialized. Cannot promote.")
            return

        if len(self.active_leaf_indices) < self.gpu_limit and self.retirement_manager and len(self.retirement_manager) > 0:
            needed_on_gpu = self.gpu_limit - len(self.active_leaf_indices)
            promoted_beam_indices, _ = self.retirement_manager.get_promoted_beam_indices(
                volume_limit=needed_on_gpu * self.max_len
            )
            if promoted_beam_indices:
                if self.verbose:
                    print(f"Promoting {len(promoted_beam_indices)} beams from retirement.")
                promoted_node_indices = [
                    self.tree.leaf_node_indices[idx] for idx in promoted_beam_indices if idx in self.tree.leaf_node_indices
                ]
                if promoted_node_indices:
                    self.graph_op.move_paths_to_device(promoted_node_indices, device=self.device)
                self.active_leaf_indices.extend(promoted_beam_indices)
        elif self.verbose:
            if len(self.active_leaf_indices) >= self.gpu_limit:
                print("Promotion skipped: GPU limit reached.")
            elif not self.retirement_manager or len(self.retirement_manager) == 0:
                print("Promotion skipped: Retirement manager is empty.")


    def retire_leaves(self, retired_leaf_beam_indices: List[int]):
        """Handles the retirement of specified leaf beam indices."""
        if self.retirement_manager is None:
            if self.verbose: print("Retirement manager not initialized. Cannot retire leaves.")
            return

        if not self.retirement_enabled:
            # If retirement is disabled, simply mark them as dead ends,
            # but do NOT offload or add to retirement_manager.
            if self.verbose:
                print(f"[FastMode] Retirement disabled; marking {len(retired_leaf_beam_indices)} beams as dead ends.")
            self.dead_end_indices.extend(
                [idx for idx in retired_leaf_beam_indices if idx in self.tree.leaf_node_indices]
            )
            return

        # Otherwise, do the usual offload + retirement_manager.add_batch:
        if retired_leaf_beam_indices:
            retired_node_indices_to_cpu = [
                self.tree.leaf_node_indices[idx] for idx in retired_leaf_beam_indices
                if idx in self.tree.leaf_node_indices
            ]
            if retired_node_indices_to_cpu:
                self.graph_op.move_paths_to_device(retired_node_indices_to_cpu, device="cpu")
            self.retirement_manager.add_batch(
                [idx for idx in retired_leaf_beam_indices if idx in self.tree.leaf_node_indices]
            )

    def shutdown_retirement_manager(self):
        """Shuts down the retirement manager thread."""
        if hasattr(self, 'retirement_manager') and self.retirement_manager:
            self.retirement_manager.shutdown()
            if self.verbose: print("Retirement manager shut down.")
    # --- End: Refactored BeamSearch methods ---
