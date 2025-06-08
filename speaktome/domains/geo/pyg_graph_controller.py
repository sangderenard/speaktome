# Standard library imports
from typing import Optional, List, TYPE_CHECKING

# Third-party imports
import torch

from ...util.lazy_loader import lazy_import
if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData

# Local application/library specific imports
from ...core.beam_search import BeamSearch
from .pygeo_mind import PyGeoMind
# Local application/library specific imports
from ...core.human_scorer_policy_manager import HumanScorerPolicyManager
from ...core.beam_search_instruction import BeamSearchInstruction
# Ensure this import points to the new location of BeamTreeVisualizer
# Import both visualizers from beam_tree_visualizer.py
from ...core.beam_tree_visualizer import BeamTreeVisualizer
from ...config import get_sentence_transformer_model
# --- END HEADER ---
class PyGGraphController:
    def __init__(self, beam_search: BeamSearch, pygeomind_model: Optional[PyGeoMind] = None, human_in_control: bool = False):
        self.beam_search = beam_search
        self.tree = beam_search.tree
        self.human_scorer_policy_manager = HumanScorerPolicyManager(beam_search.scorer) 
        self.graph_op = beam_search.graph_op # Assuming BeamGraphOperator is already on beam_search.tree
        self.pygeomind_model = pygeomind_model
        self.tensor_ops = beam_search.scorer.tensor_ops
        self.targeted_pyg_node_for_expansion: Optional[int] = None
        self.human_in_control = human_in_control
        self.human_chosen_beam_idx_for_expansion: Optional[int] = None
        self.current_step_for_pygeomind: int = 0 # To help with symbolic_id
        # New attributes for auto-run mode
        self.auto_run_counter: int = 0
        self.auto_run_mode: Optional[str] = None
        self.initial_human_in_control = human_in_control
    def _convert_action_to_instruction_list(self, action: str):
        # Fetch current settings from beam_search.scorer and beam_search itself
        # to ensure human-generated instructions use current global defaults.
        current_scorer = self.beam_search.scorer
        current_lookahead = self.beam_search.lookahead_steps

        # Default policy values from the scorer
        default_bins = list(current_scorer.default_score_policy.get("score_bins", {}).values())
        default_pre_temp = current_scorer.default_score_policy.get("pre_temp", 1.5)
        default_pre_top_k = current_scorer.default_score_policy.get("pre_top_k", 50)
        default_cull_after = current_scorer.default_score_policy.get("cull_after", 3)
        # current_lookahead is already fetched from self.beam_search.lookahead_steps

        common_params = {
            "scorer": current_scorer,
            "score_bins": default_bins,
            "pre_temp": default_pre_temp,
            "pre_top_k": default_pre_top_k,
            "cull_after": default_cull_after,
            "lookahead_steps": current_lookahead
        }

        if action == "done":
            return []
        elif action == "expand_targeted" and hasattr(self, "targeted_pyg_node_for_expansion"):
            beam_idx = self.tree.get_beam_idx_from_pyg_node(self.targeted_pyg_node_for_expansion)
            if beam_idx is None:
                print("[Warning] No beam_idx found for targeted pyg node.")
                return [BeamSearchInstruction(action="expand_any", node_id=-1, justification="Fallback: invalid target for expand_targeted", **common_params)]
            return [BeamSearchInstruction(
                node_id=beam_idx,
                action="expand_targeted",
                **common_params
            )]
        elif action == "expand_targeted_human" and hasattr(self, "human_chosen_beam_idx_for_expansion"):
            return [BeamSearchInstruction(
                node_id=self.human_chosen_beam_idx_for_expansion,
                action="expand_targeted",
                **common_params
            )]
        elif action.startswith("promote:"):
            try:
                target_beam_idx = int(action.split(":")[1])
                return [BeamSearchInstruction(action="promote", node_id=target_beam_idx, **common_params)]
            except (IndexError, ValueError):
                print(f"[Warning] Malformed targeted promote action: {action}. Defaulting to general promote.")
                return [BeamSearchInstruction(action="promote", node_id=-1, justification="Fallback: malformed promote target", **common_params)]
        elif action == "promote":
            return [BeamSearchInstruction(action="promote", node_id=-1, **common_params)]
        elif action == "expand_any":
            return [BeamSearchInstruction(action="expand_any", node_id=-1, **common_params)]
        elif action == "noop_controller_switch": # Handle new no-op action
            return []
        else:
            print(f"[Warning] Unrecognized action: {action}")
            return []

    def decide_next_action(self, current_step_id: int) -> str:
        """
        Examine PyG state and decide what to do next.
        Returns one of: "expand", "promote", "done"
        """
        # Ensure active_leaf_indices exists; if not, maybe we need to initialize or it's truly done.
        if not hasattr(self.beam_search, 'active_leaf_indices'):
            return "done"  # Should be initialized by initialize_search

        # Check for auto-run mode
        if self.auto_run_counter > 0 and self.auto_run_mode is not None:
            self.auto_run_counter -= 1
            print(f"Auto-running '{self.auto_run_mode}', {self.auto_run_counter} rounds remaining.")
            return self.auto_run_mode

        if self.pygeomind_model and not self.human_in_control:  # PyGeoMind decides if not human_in_control
            pyg_data = self.tree.get_pyg_data()
            if pyg_data and pyg_data.num_nodes > 0:
                instructions = self.pygeomind_model.run(
                    pyg_data, beam_tree=self.tree, current_step_id=current_step_id
                )
                active_pyg_leaf_nodes = self.beam_search.get_active_leaves_pyg_nodes()

                for instr in sorted(
                    instructions,
                    key=lambda x: x.metadata.get("score", float('-inf')),
                    reverse=True
                ):
                    if instr.action == "expand":
                        pyg_node_to_expand = instr.node_id  # This is pyg_node_idx
                        # Check if this PyG node is an active leaf
                        if pyg_node_to_expand in active_pyg_leaf_nodes:
                            # Map pyg_node_idx to beam_idx
                            beam_idx_to_expand = pyg_data.pyg_node_to_beam_idx[pyg_node_to_expand].item()
                            if (
                                beam_idx_to_expand != -1
                                and beam_idx_to_expand in self.beam_search.active_leaf_indices
                            ):
                                self.targeted_pyg_node_for_expansion = pyg_node_to_expand
                                print(
                                    f"PyGeoMind decided to expand PyG node "
                                    f"{pyg_node_to_expand} (beam_idx {beam_idx_to_expand})"
                                )
                                return "expand_targeted"
            else:
                print(
                    "PyGeoMind: No PyG data or empty graph, falling back to default logic."
                )

        if self.human_in_control:
            print("\n=== Human Decision Point ===")
            pyg_data = self.tree.get_pyg_data()
            active_leaves_info = []

            if pyg_data and pyg_data.num_nodes > 0:
                print("Active Leaf Beams:")
                for beam_idx in self.beam_search.active_leaf_indices:
                    original_node_idx = self.tree.leaf_node_indices.get(beam_idx)
                    pyg_node_id_val = -1
                    if original_node_idx is not None:
                        pyg_node_id_val = self.tree.node_idx_to_pyg_id.get(original_node_idx, -1)

                    tokens, _, length = self.tree.get_beam_tensors_by_beam_idx(
                        beam_idx, max_len=15
                    )
                    snippet = self.beam_search.scorer.tokenizer.decode(
                        tokens[:length], skip_special_tokens=True
                    )
                    active_leaves_info.append(
                        {'beam_idx': beam_idx, 'pyg_id': pyg_node_id_val, 'snippet': snippet}
                    )
                    print(
                        f"  Beam Idx: {beam_idx}, PyG Node: {pyg_node_id_val}, "
                        f"Text: '{snippet[:50]}...'"
                    )
            else:
                print("No active leaf beams or no PyG data.")

            while True:
                print("\nChoose action:")
                print("  e <beam_idx>      - Expand specific leaf beam")
                print("  p [beam_idx]      - Promote any or specific retired beam")
                print("  a [rounds]        - Expand any (default logic) [for N rounds]")
                print("  u <beam_idx>      - Move up one level to parent")
                print("  x <beam_idx>      - Extend siblings of a beam (lateral)")
                print("  search <text>     - Find best match for tokenized input")
                print("  snap <text>       - Snap full sentence beam into tree with score trace")
                print("  trace <beam_idx>  - Trace beam path from given beam to root")
                print("  policy            - Configure scorer policy")
                print("  ra | rt           - View top-N retired or active beams")
                print("  viz               - Visualization menu (graph export views)")
                print("  s <pyg_id>        - Visualize subtree from PyG node ID")
                print("  pca               - Visualize sentence embeddings (PCA)")
                print("  summary           - View summary of all beams")
                print(f"  la [steps]        - View/Set lookahead steps (current: {self.beam_search.lookahead_steps})")
                print(
                    "  t                 - Toggle retirement on/off "
                    f"(currently {'ON' if self.beam_search.retirement_enabled else 'OFF'})"
                )
                print("  d                 - Done with search")

                human_choice = input("Your choice: ").strip().lower()

                if human_choice.startswith("la"):
                    parts = human_choice.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        new_steps = int(parts[1])
                        if new_steps >= 1:
                            self.beam_search.lookahead_steps = new_steps
                            print(f"Lookahead steps set to {new_steps}.")
                        else:
                            print("Lookahead steps must be >= 1.")
                    else:
                        print(f"Current lookahead steps: {self.beam_search.lookahead_steps}")
                    continue


                # ─── Toggle retirement on/off ───
                if human_choice == "t":
                    self.beam_search.retirement_enabled = not self.beam_search.retirement_enabled
                    status = "ON" if self.beam_search.retirement_enabled else "OFF"
                    print(f"[Human] Retirement is now {status}.")
                    continue  # Re-print the menu

                elif human_choice.startswith("e "):
                    try:
                        target = int(human_choice.split(" ")[1])
                        if any(info['beam_idx'] == target for info in active_leaves_info):
                            self.human_chosen_beam_idx_for_expansion = target
                            return "expand_targeted_human"
                        else:
                            print("Invalid beam_idx. Not active.")
                    except Exception as e:
                        print("Invalid input for 'e':", e)

                elif human_choice.startswith("p "):
                    try:
                        target = int(human_choice.split(" ")[1])
                        return f"promote:{target}"
                    except Exception:
                        print("Invalid format for promote. Use 'p <beam_idx>'.")

                elif human_choice.startswith("p"):  # Handles "p" and "p <idx>"
                    return "promote"

                elif human_choice.startswith("a"):
                    parts = human_choice.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        self.auto_run_counter = int(parts[1]) - 1  # -1 because this choice counts as one round
                        self.auto_run_mode = "expand_any"
                        print(f"Auto-running 'expand_any' for {int(parts[1])} rounds.")
                    else:
                        self.auto_run_counter = 0
                        self.auto_run_mode = None
                    return "expand_any"

                elif human_choice == "a":
                    return "expand_any"

                elif human_choice.startswith("u "):
                    try:
                        child_idx = int(human_choice.split(" ")[1])
                        parent_idx = self.tree.get_parent_beam_idx(child_idx)
                        if parent_idx is not None:
                            self.human_chosen_beam_idx_for_expansion = parent_idx
                            return "expand_targeted_human"
                        else:
                            print("No parent found.")
                    except Exception:
                        print("Use 'u <beam_idx>'.")

                elif human_choice.startswith("x "):
                    try:
                        sib_idx = int(human_choice.split(" ")[1])
                        sibling_idxs = self.tree.get_sibling_beam_indices(sib_idx)
                        print(f"Sibling beam candidates: {sibling_idxs}")
                        # Return a special code or store them for next action
                        self.human_chosen_sibling_group = sibling_idxs
                        return "expand_siblings"
                    except Exception:
                        print("Use 'x <beam_idx>'.")

                elif human_choice.startswith("search "):
                    query = human_choice[len("search "):].strip()
                    results = self.tree.search_closest_beams(
                        query, tokenizer=self.beam_search.scorer.tokenizer
                    )
                    for idx, score, text in results:
                        print(f"[{idx}] Score: {score:.3f} | {text}")
                    continue

                elif human_choice.startswith("snap "):
                    text = human_choice[len("snap "):].strip()
                    tokenizer = self.beam_search.scorer.tokenizer
                    model_for_scoring = self.beam_search.scorer.model
                    device_for_scoring = model_for_scoring.device

                    snapped_tokens_list: List[int] = []
                    snapped_scores_list: List[float] = []

                    try:
                        if not text:
                            print("Cannot snap empty text.")
                            continue

                        input_ids_tensor = tokenizer.encode(
                            text, return_tensors="pt"
                        ).to(device_for_scoring)

                        if input_ids_tensor.size(1) > 0:
                            first_token_id = input_ids_tensor[0, 0].item()
                            snapped_tokens_list.append(first_token_id)
                            snapped_scores_list.append(0.0)  # Score for the first token

                            current_context_ids = input_ids_tensor[:, 0:1]

                            for i in range(1, input_ids_tensor.size(1)):
                                next_token_id_item = input_ids_tensor[0, i].item()
                                with torch.no_grad():
                                    outputs = model_for_scoring(input_ids=current_context_ids)
                                    logits = outputs.logits[:, -1, :]
                                    log_probs = self.tensor_ops.log_softmax(logits, dim=-1)

                                token_logprob = log_probs[0, next_token_id_item].item()
                                snapped_tokens_list.append(next_token_id_item)
                                snapped_scores_list.append(token_logprob)
                                current_context_ids = self.tensor_ops.cat(
                                    [current_context_ids, input_ids_tensor[:, i:i+1]], dim=1
                                )

                        if snapped_tokens_list:
                            final_beam_idx = self.tree.snap_beam_path(
                                tokens=snapped_tokens_list,
                                scores=snapped_scores_list,
                                device_str=str(self.tree.device)
                            )
                            if final_beam_idx != -1:
                                self.beam_search.active_leaf_indices.append(final_beam_idx)
                                print(
                                    f"Snapped beam path. Final beam idx: {final_beam_idx} (now active)"
                                )
                            else:
                                print("Snap operation resulted in an invalid beam index.")
                        else:
                            print("No tokens to snap after processing.")
                    except Exception as e:
                        print(f"Snap failed: {e}")
                    continue

                elif human_choice.startswith("trace "):
                    try:
                        beam_idx = int(human_choice.split(" ")[1])
                        lineage = self.tree.trace_beam_path(
                            beam_idx, tokenizer=self.beam_search.scorer.tokenizer
                        )
                        print("\n--- Beam Trace ---")
                        if not lineage:
                            print(f"Beam index {beam_idx} not found.")
                        for i, (tok, score) in enumerate(lineage):
                            decoded = self.beam_search.scorer.tokenizer.decode(
                                [tok], skip_special_tokens=True
                            )
                            print(
                                f"{i:02}: Token '{decoded}' (id={tok}) | Score: {score:.4f}"
                            )
                    except Exception as e:
                        print(f"Trace failed: {e}")
                    continue

                elif human_choice == "policy":
                    self.human_scorer_policy_manager.set_default_policy_interactive()
                    continue  # Re-prompt after setting policy

                elif human_choice in {"ra", "rt"}:
                    top_retired = self.beam_search.get_top_retired_beams(n=10)
                    print("\n-- Top Retired Beams --")
                    for beam_idx, text, score in top_retired:
                        print(f"[{beam_idx}] Score: {score:.2f} | {text[:60]}...")
                    continue

                elif human_choice == "summary":
                    all_beams = self.beam_search.get_all_beams_summary(n=20)
                    print("\n-- All Beams Summary --")
                    for beam_idx, text, score, status in all_beams:
                        print(
                            f"[{beam_idx}] Score: {score:.2f} | Status: {status:10} | {text[:50]}..."
                        )
                    continue

                elif human_choice == "viz":
                    visualizers = self.tree.get_visualization_plugins()
                    print("Available Visualizations:")
                    for i, plugin in enumerate(visualizers):
                        print(f"  {i}: {plugin.name} - {plugin.description}")
                    try:
                        selected = int(input("Select plugin number: "))
                        visualizers[selected].render(self.tree.export_for_visualizer())
                    except Exception as e:
                        print(f"Visualization error: {e}")
                    continue

                elif human_choice == "pca":
                    if self.tree and self.tree.nodes:
                        print("\nVisualizing node sentence embeddings (PCA)...")
                        BeamTreeVisualizer().visualize_sentence_embeddings(
                            self.tree,
                            get_sentence_transformer_model(),
                            self.beam_search.scorer.tokenizer,
                        )
                    else:
                        print("Tree is empty. Skipping PCA sentence embedding visualization.")
                    continue

                elif human_choice.startswith("g") and self.pygeomind_model:
                    parts = human_choice.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        num_rounds = int(parts[1])
                        if num_rounds <= 0:
                            print("Number of rounds must be positive.")
                            continue
                        self.auto_run_counter = num_rounds # Total N rounds
                        self.auto_run_mode = "pygeomind_decide"
                        print(f"PyGeoMind will run for {self.auto_run_counter} round(s) starting now.")
                    else:
                        self.auto_run_counter = 1
                        self.auto_run_mode = "pygeomind_decide"
                        print(f"PyGeoMind will run for 1 round starting now.")
                    self.human_in_control = False # PyGeoMind takes over
                    return "noop_controller_switch" # Signal run_loop to let PyGeoMind run this step


                elif human_choice.startswith("s "):
                    try:
                        target_pyg_id = int(human_choice.split(" ")[1])
                        if pyg_data and 0 <= target_pyg_id < pyg_data.num_nodes:
                            BeamTreeVisualizer().visualize_subtree(
                                pyg_data,
                                self.beam_search.scorer.tokenizer,
                                self.beam_search,
                                target_pyg_id,
                                title=f"Subtree from PyG Node {target_pyg_id} (Step {current_step_id})"
                            )
                        else:
                            print("Invalid PyG node ID or no graph data.")
                    except Exception as e:
                        print("Invalid format:", e)
                    continue

                elif human_choice == "d":
                    return "done"

                else:
                    print("Unrecognized command.")

        # ─── Fallback / Default logic if PyGeoMind doesn't provide a valid expansion target ───
        active_count = len(self.beam_search.active_leaf_indices)
        can_promote = (
            hasattr(self.beam_search, 'retirement_manager')
            and len(self.beam_search.retirement_manager) > 0
        )

        if active_count == 0:
            return "promote" if can_promote else "done"

        if active_count < self.beam_search.gpu_limit and can_promote:
            return "promote"

        if self.auto_run_mode == "pygeomind_decide":
            # If auto-running PyGeoMind, let it decide
            self.human_in_control = False
            action = self.decide_next_action(current_step_id)
            self.human_in_control = True
            return action

        return "expand_any"  # Expand any available active leaf


    def run_loop(self, seed_text: str, max_steps: int = 100, beam_width_override: Optional[int] = None):
        # 1) If the user asked to override beam_width, just change it here
        if beam_width_override is not None:
            self.beam_search.beam_width = beam_width_override
            # Bins are rebuilt via the scorer when we apply the first instruction.

        # 2) Initialize the beam search
        self.beam_search.initialize_search(seed_text)

        for step in range(max_steps):
            print(f"\n--- Controller Step {step+1}/{max_steps} ---")

            # 3) Query PyGeoMind (or human) → but now it's a LIST of instructions
            # Check if we are in PyGeoMind auto-run mode
            # PyGeoMind runs if:
            # 1. It's the default actor (model exists, human not in control, no other auto_run_mode like "expand_any")
            # 2. Or, auto_run_mode is "pygeomind_decide" and counter > 0 (human_in_control would be false here)
            is_pygeomind_default_turn = self.pygeomind_model and \
                                        not self.human_in_control and \
                                        (self.auto_run_mode is None or self.auto_run_mode == "pygeomind_decide")
            
            is_pygeomind_auto_run = self.auto_run_mode == "pygeomind_decide" and self.auto_run_counter > 0

            is_pygeomind_turn = is_pygeomind_default_turn or is_pygeomind_auto_run
            
            if is_pygeomind_turn:
                if self.auto_run_mode == "pygeomind_decide" and self.auto_run_counter > 0: # Manage counter for auto-run
                    self.auto_run_counter -= 1
                    print(f"PyGeoMind auto-running. {self.auto_run_counter} round(s) remaining after this.")

                # If auto-running PyGeoMind, ensure human_in_control is false for decide_next_action
                pyg_data = self.tree.get_pyg_data()
                if pyg_data is not None and pyg_data.num_nodes > 0:
                    instr_list = self.pygeomind_model.run(pyg_data, self.tree, current_step_id=step)
                else:
                    instr_list = [BeamSearchInstruction(
                        node_id=-1,
                        action="expand_any",
                        **self._get_common_instr_params(step, "Controller fallback: empty graph", f"controller_step{step}_fallback_empty_graph")
                    )]
                if self.auto_run_mode == "pygeomind_decide" and self.auto_run_counter == 0:
                    self.auto_run_mode = None # Reset auto-run mode
                    if self.initial_human_in_control: # Revert to human control if it was originally set
                        self.human_in_control = True

            elif self.human_in_control:
                next_action = self.decide_next_action(current_step_id=step)
                instr_list = self._convert_action_to_instruction_list(next_action)

                # If next_action was "noop_controller_switch", instr_list is []. PyGeoMind will run due to human_in_control=False.

            else: # Not PyGeoMind's turn, and not Human's turn for decision.
                  # This handles 'expand_any' auto-run or other fallbacks.
                if self.auto_run_mode == "expand_any" and self.auto_run_counter > 0:
                    self.auto_run_counter -= 1
                    print(f"Auto-running 'expand_any'. {self.auto_run_counter} round(s) remaining after this.")
                    instr_list = self._convert_action_to_instruction_list("expand_any")
                    if self.auto_run_counter == 0:
                        self.auto_run_mode = None # Reset auto-run mode
                        if self.initial_human_in_control: # Revert to human control
                            self.human_in_control = True
                else: # Default fallback if no other agent is active

                    instr_list = [BeamSearchInstruction(
                        node_id=-1,
                        action="expand_any",
                        **self._get_common_instr_params(step, "Controller default", f"controller_step{step}_default_fallback")
                    )]


            # Ensure we have a LIST
            if not isinstance(instr_list, list):
                instr_list = [instr_list]

            # 4) Apply each instruction in the returned list
            for instr in instr_list:
                self.beam_search.apply_instruction(instr)
                print(f"[Applied Instruction] {instr}")

                # 5) Dispatch actions
                if instr.action == "expand_targeted" and instr.node_id is not None:
                    self.beam_search.expand_specific_leaf_once(instr.node_id)
                elif instr.action == "expand_internal_as_leaf" and instr.node_id is not None:
                    self.beam_search.expand_internal_as_leaf(instr.node_id)
                elif instr.action == "expand_any":
                    self.beam_search.expand_and_score_once()
                elif instr.action == "promote":
                    self.beam_search.promote_if_needed()
                elif instr.action == "done":
                    print("Controller decided to end search.")
                    self.beam_search.done = True
                    break

            # Loop returns to top for next GNN pass
            if getattr(self.beam_search, "done", False):
                break

        # 6) Clean up
        self.beam_search.shutdown_retirement_manager()
        print("PyGGraphController run_loop finished.")
        return self.beam_search.tree

    def _get_common_instr_params(self, step: int, justification: str, symbolic_id_suffix: str) -> dict:
        """Helper to get common parameters for BeamSearchInstruction."""
        params = {
            "scorer": self.beam_search.scorer,
            "score_bins": list(self.beam_search.scorer.default_score_policy.get("score_bins", {}).values()),
            "pre_temp": self.beam_search.scorer.default_score_policy.get("pre_temp", 1.5),
            "pre_top_k": self.beam_search.scorer.default_score_policy.get("pre_top_k", 50),
            "cull_after": self.beam_search.scorer.default_score_policy.get("cull_after", 3),
            "lookahead_steps": self.beam_search.lookahead_steps,
            "justification": justification,
            "symbolic_id": f"controller_step{step}_{symbolic_id_suffix}"
        }
        # Ensure priority is included if it's a common parameter, or handle it separately.
        # For these fallbacks, a default priority is fine.
        params["priority"] = 0.0 
        return params
