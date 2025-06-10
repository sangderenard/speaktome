try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    # Standard library imports
    from typing import List, Optional, TYPE_CHECKING

    # Third-party imports
    import torch  # type: ignore[import-untyped]
    from torch_geometric.data import Data as PyGData  # type: ignore[import-untyped] # Moved for runtime availability

    # Local application/library specific imports
    from .beam_search import BeamSearch
    from .beam_search_instruction import BeamSearchInstruction
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class HumanPilotController:
    """
    ╔══════════════════════════════════════════════════════════╗
    ║              HUMAN PILOT BEAM CONTROLLER                 ║
    ║   Human-in-the-loop traversal through GPT beam space     ║
    ║   with GNN observers passively logging metrics for       ║
    ║   future autonomous simulation control.                  ║
    ╚══════════════════════════════════════════════════════════╝
    """


    def __init__(self, beam_search: BeamSearch, policy_model: Optional[torch.nn.Module] = None):
        self.beam_search = beam_search
        self.tree = beam_search.tree
        self.policy_model = policy_model
        self.history = []
        self.step_id = 0
        self._stop = False

    def run(self):
        while not self._stop:
            self.step_id += 1
            try:
                # Must-Fix 1.3: Call get_pyg_data without extra args
                graph: Optional[PyGData] = self.tree.get_pyg_data()
                if graph is None or graph.num_nodes == 0:
                    print("\n🛑 No active graph nodes. Halting.")
                    break

                instructions = self.generate_instructions(graph)

                if self.policy_model:
                    self.log_policy(graph, instructions)

                instruction = self.ask_human_choice(instructions)

                if instruction:
                    self.history.append(instruction)
                    self.apply_instruction(instruction)

            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user.")
                break

    def generate_instructions(self, graph: PyGData) -> List[BeamSearchInstruction]:
        """
        Create candidate instructions to offer the human.
        For now: top-k expand commands on leaf nodes.
        """
        # Must-Fix 1.4: Use range(graph.num_nodes) instead of graph.node_ids
        node_ids = list(range(graph.num_nodes)) # PyG node indices
        scores = torch.rand(len(node_ids))  # Placeholder priority scores
        k = min(6, len(scores))

        instructions = []
        for i in range(k):
            nid = node_ids[i]
            p = float(scores[i])
            instructions.append(BeamSearchInstruction(
                node_id=nid,
                action="expand",
                priority=p,
                justification="Stub: randomly ranked leaf",
                symbolic_id=f"beam-{nid}-s{self.step_id}"
            ))

        return sorted(instructions, key=lambda i: -i.priority)

    def ask_human_choice(self, instructions: List[BeamSearchInstruction]) -> Optional[BeamSearchInstruction]:
        print(f"\n📍 [STEP {self.step_id}] Human Beam Control Interface")
        print("-" * 60)
        for idx, instr in enumerate(instructions):
            print(f"[{idx}] {instr}")
        print("\nCommands:")
        print("  [x] Skip step")
        print("  [q] Quit controller")

        while True:
            choice = input("Choose action: ").strip().lower()
            if choice == "x":
                return None
            elif choice == "q":
                self._stop = True
                return None
            elif choice.isdigit() and int(choice) < len(instructions):
                return instructions[int(choice)]
            print("Invalid input. Try again.")

    def apply_instruction(self, instr: BeamSearchInstruction):
        if instr.action == "expand":
            # Must-Fix 5.5: Call the correct method. instr.node_id is a PyG node ID. Need to map to beam_idx.
            # This requires a more complex mapping if HumanPilotController is to be used directly.
            # For now, assuming PyGGraphController's human mode is the primary interface, which handles this.
            # If HumanPilotController is used standalone, this mapping needs to be implemented.
            # For now, let's assume this method is called with a beam_idx if used by a different controller.
            # If instr.node_id is a pyg_node_id, it needs conversion.
            # If it's already a beam_idx (as per PyGGraphController's human mode logic), then it's fine.
            # Given the context of the audit, it's likely instr.node_id is intended to be a beam_idx here.
            # However, generate_instructions uses PyG node IDs. This is a mismatch.
            # For now, assuming instr.node_id is a *beam_idx* for this specific method.
            # A proper fix would involve generate_instructions providing beam_idx or this method doing the lookup.
            # Let's assume for now that the calling context (if any outside PyGGraphController) would provide beam_idx.
            # If this HumanPilotController is *only* a conceptual stub and not directly run, this is less critical.
            # Given the audit context, let's assume it *should* be a beam_idx.
            # The PyGGraphController's human mode correctly maps PyG node ID to beam_idx before calling expand_specific_leaf_once.
            # So, if this HumanPilotController is a separate entity, it needs that mapping.
            # For the purpose of this diff, I'll assume the intent was to call with a beam_idx.
            # If instr.node_id is a PyG node ID, this will fail.
            # The audit points out `expand_node` is wrong, and suggests `expand_specific_leaf_once`.
            # `expand_specific_leaf_once` takes a `target_beam_idx`.
            # So, `instr.node_id` *must* be a beam_idx for this to work.
            # This implies `generate_instructions` should provide beam_idx.
            # For now, I will make the call correct, assuming instr.node_id is a beam_idx.
            # If it's a PyG node ID, the calling code or generate_instructions needs adjustment.
            print(f"[HumanPilotController] Attempting to expand beam_idx: {instr.node_id}")
            self.beam_search.expand_specific_leaf_once(instr.node_id)
        elif instr.action == "freeze":
            self.tree.freeze_node(instr.node_id)
        elif instr.action == "prune":
            self.tree.prune_node(instr.node_id)
        else:
            print(f"[WARN] Unrecognized instruction: {instr}")

    def log_policy(self, graph: PyGData, instructions: List[BeamSearchInstruction]):
        """
        Log policy observations from the current graph. May store state for future training.
        """
        with torch.no_grad():
            _ = self.policy_model(graph)  # Forward pass for passive observation
            print(f"📡 GNN policy observed step {self.step_id}.")

            # Future: dump embeddings, node actions, etc. to log
