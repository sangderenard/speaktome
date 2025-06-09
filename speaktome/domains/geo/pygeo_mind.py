# Standard library imports
import collections
from typing import Dict, List, Tuple, TYPE_CHECKING

# Third-party imports
import torch

from ...tensors.faculty import Faculty

FACULTY_REQUIREMENT = Faculty.PYGEO

from ...util.lazy_loader import lazy_install
if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    import torch_geometric.nn as pyg_nn

# Local application/library specific imports
from ...core.beam_search_instruction import BeamSearchInstruction
from ...core.compressed_beam_tree import CompressedBeamTree
from ...core.human_scorer_policy_manager import HumanScorerPolicyManager
from ...core.scorer import Scorer
# --- END HEADER ---


class PyGeoMind(torch.nn.Module):
    """
    ┌────────────────────────────────────────────┐
    │              The Grand Hall                │
    │        Pilot Seat for Beam Dominion        │
    └────────────────────────────────────────────┘

    This is the central chamber of decision for the PyGeo engine.
    It receives the full graph state as structured memory and 
    outputs action plans to guide the simulation's future.

    Here lies the seat of attention, memory, and motion across
    GPT-inspired space traversals—arbitrary direction, 
    with authority granted by learned policy.
    """

    def __init__(self, scorer: Scorer, input_dim: int = 768, hidden_dim: int = 512, beam_width: int = 5, **kwargs):
        super().__init__()

        pyg_nn = lazy_install('torch_geometric.nn', 'torch_geometric')

        # ══════ Encoder & GNN ══════
        self.encoder       = torch.nn.Linear(input_dim, hidden_dim)
        self.message_gnn   = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        # Ensure GRU hidden_dim matches the output of message_gnn and input of policy_head
        self.integrator    = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head   = torch.nn.Linear(hidden_dim, 1)

        # Use the tensor abstraction provided by the scorer for backend agnostic
        # operations throughout this module.
        self.tensor_ops = scorer.tensor_ops

        # ══════ Auxin‐inspired hyperparameters ══════
        self.DELTA_GROWTH_GAP      = 0.05       # bud if parent_score > children_sum + Δ
        self.ALPHA_OVERSHOOT_FACTOR = 1.2       # suppress if children_sum > α * parent_score
        self.K_DEEPEN_LEAVES       = beam_width # expand up to this many leaves per GNN pass

        # ══════ Memory for GRU hidden state ══════
        self.context_state = None

        # ══════ Store beam_width for use in expand_internal_as_leaf ══════
        self.beam_width    = beam_width

        if scorer is None:
            raise ValueError("PyGeoMind requires a Scorer instance.")
        self.scorer = scorer
        # HumanScorerPolicyManager might not be needed directly if PyGeoMind uses the Scorer's defaults
        
        self.scorer = scorer
        self.human_scorer_policy_manager = HumanScorerPolicyManager(self.scorer)
        

    def _get_leaves_under_internal_node(self, internal_node_pyg_id: int, data: 'PyGData') -> List[int]:
        """
        Return a list of all PyG‐node IDs that are leaves in the subtree rooted at `internal_node_pyg_id`.
        Uses a cached adjacency list for efficiency (built once per run() call).
        """
        # Build or reuse adjacency list: node_id → list of children
        if not hasattr(self, "_adj") or self._adj_size != data.num_nodes:
            adj = [[] for _ in range(data.num_nodes)]
            for idx in range(data.edge_index.shape[1]):
                parent = data.edge_index[0, idx].item()
                child  = data.edge_index[1, idx].item()
                adj[parent].append(child)
            self._adj      = adj
            self._adj_size = data.num_nodes
        else:
            adj = self._adj

        # BFS from internal_node_pyg_id down to collect leaves
        leaves_under_internal: List[int] = []
        q = collections.deque()

        # Initialize queue with direct children
        for child in adj[internal_node_pyg_id]:
            q.append(child)

        visited = {internal_node_pyg_id}
        while q:
            curr = q.popleft()
            if curr in visited:
                continue
            visited.add(curr)

            if data.pyg_node_is_leaf[curr]:
                leaves_under_internal.append(curr)
            else:
                for child in adj[curr]:
                    if child not in visited:
                        q.append(child)

        return leaves_under_internal


    def _get_ancestors(self, pyg_node_id: int, tree: CompressedBeamTree) -> List[int]:
        """
        Return a list of PyG‐node IDs corresponding to all strict ancestors of `pyg_node_id`
        in the CompressedBeamTree. (Does not include the node itself.)
        """
        ancestor_ids: List[int] = []
        original_idx = tree.pyg_id_to_node_idx.get(pyg_node_id)
        if original_idx is None:
            return []

        cur = tree.nodes[original_idx].parent_node_idx
        while cur is not None:
            anc_pyg = tree.node_idx_to_pyg_id.get(cur)
            if anc_pyg is not None:
                ancestor_ids.append(anc_pyg)
            cur = tree.nodes[cur].parent_node_idx

        return ancestor_ids

    def run(self, data: 'PyGData', beam_tree: CompressedBeamTree, current_step_id: int = 0) -> List[BeamSearchInstruction]:
        """
        Produce up to K_DEEPEN_LEAVES+1 instructions (one optional BUD + up to K_DEEPEN_LEAVES leaf expansions).
        """
        # 0) If the graph is empty, return a single “expand_any” fallback
        if data is None or data.num_nodes == 0:
            return [BeamSearchInstruction(
                node_id=-1,
                action="expand_any",
                scorer=self.scorer, # Pass the scorer instance
                score_bins=list(self.scorer.default_score_policy["score_bins"].values()),
                pre_temp=self.scorer.default_score_policy["pre_temp"],
                pre_top_k=self.scorer.default_score_policy["pre_top_k"],
                cull_after=self.scorer.default_score_policy["cull_after"],
                lookahead_steps=self.scorer.default_score_policy.get("lookahead_steps", 1),
                justification="PyGeoMind: empty graph fallback",
                priority=0.0,
                symbolic_id=f"pygeomind_step{current_step_id}_empty"
            )]

        # 1) GNN pass → h_integ → raw node scores s(n)
        x          = data.x
        edge_index = data.edge_index

        h = self.encoder(x)
        h = self.message_gnn(h, edge_index)

        if self.context_state is None:
            # initial hidden state for GRU
            self.context_state = self.tensor_ops.zeros(
                (1, 1, h.size(1)),
                dtype=self.tensor_ops.get_dtype(h),
                device=h.device,
            )
        
        batch_in = h.unsqueeze(0)  # shape (1, num_nodes, hidden_dim)

        # Ensure context_state is compatible with GRU's expected input for num_layers * num_directions
        # For a single layer, unidirectional GRU, context_state should be (1, batch_size_for_gru, hidden_size)
        # Here, batch_in is (1, num_nodes, hidden_dim), so GRU's batch_size is 1 (as batch_first=True).
        # The context_state should be (1, 1, hidden_dim)
        batch_size = batch_in.size(0)
        hidden_size = self.integrator.hidden_size # Use hidden_size from GRU
        device = batch_in.device
        if self.context_state is None or self.context_state.size(1) != batch_size:
            self.context_state = self.tensor_ops.zeros(
                (1, batch_size, hidden_size),
                dtype=self.tensor_ops.get_dtype(h),
                device=device,
            )

        next_ctx, _ = self.integrator(batch_in, self.context_state)
        
        self.context_state = next_ctx

        h_integ = next_ctx.squeeze(0)  # shape (num_nodes, hidden_dim)
        node_scores_s_n = self.policy_head(h_integ).squeeze(-1)  # shape [num_nodes]

        instructions: List[BeamSearchInstruction] = []

        # 2) Identify top‐scoring internal node for possible BUD
        top_internal = -1
        max_int_score = -float('inf')
        for i in range(data.num_nodes):
            if not data.pyg_node_is_leaf[i]:
                sc = node_scores_s_n[i].item()
                if sc > max_int_score:
                    max_int_score = sc
                    top_internal = i

        if top_internal != -1:
            s_I = max_int_score
            leaves_under_I = self._get_leaves_under_internal_node(top_internal, data)
            P_I = sum(node_scores_s_n[leaf].item() for leaf in leaves_under_I)
            if s_I > P_I + self.DELTA_GROWTH_GAP:
                # Emit one “bud” instruction
                instructions.append(BeamSearchInstruction(
                    node_id=top_internal,
                    action="expand_internal_as_leaf",
                    metadata={"raw_internal_score": s_I, "children_sum": P_I},
                    priority=s_I,
                    justification=f"PyGeoMind BUD at {top_internal}: {s_I:.2f} > {P_I:.2f}",
                    symbolic_id=f"pygeomind_step{current_step_id}_bud",
                    scorer=self.scorer,
                    score_bins=list(self.scorer.default_score_policy["score_bins"].values()),
                    pre_temp=self.scorer.default_score_policy["pre_temp"],
                    pre_top_k=self.scorer.default_score_policy["pre_top_k"],
                    cull_after=self.scorer.default_score_policy["cull_after"],
                    lookahead_steps=self.scorer.default_score_policy.get("lookahead_steps", 1)
                ))
                # (Do not return early—still compute leaf expansions below.)

        # 3) DEEPEN logic: suppression + inherited‐score ranking
        # Note: _get_leaves_under_internal_node builds/uses an adjacency list for efficient BFS.

        # 3a) Initialize each leaf’s “effective” score to its raw s(n)
        effective_leaf_scores: Dict[int, float] = {}
        for i in range(data.num_nodes):
            if data.pyg_node_is_leaf[i]:
                effective_leaf_scores[i] = node_scores_s_n[i].item()

        # 3b) For each internal node I, if its leaves’ sum > alpha * s(I), suppress
        for i in range(data.num_nodes):
            if not data.pyg_node_is_leaf[i]:
                sI = node_scores_s_n[i].item()
                leaves_i = self._get_leaves_under_internal_node(i, data)
                if not leaves_i:
                    continue
                P = sum(node_scores_s_n[leaf].item() for leaf in leaves_i)
                if P > sI * self.ALPHA_OVERSHOOT_FACTOR and P > 1e-9:
                    factor = sI / P
                    for leaf in leaves_i:
                        raw_leaf = node_scores_s_n[leaf].item()
                        effective_leaf_scores[leaf] = raw_leaf * factor

        # 3c) Compute “inherited” score for each leaf = max(effective_leaf, max(ancestors))
        inherited_list: List[Tuple[float,int]] = []
        for leaf in range(data.num_nodes):
            if data.pyg_node_is_leaf[leaf]:
                raw_leaf = node_scores_s_n[leaf].item()
                s_eff = effective_leaf_scores.get(leaf, raw_leaf)
                ancestors = self._get_ancestors(leaf, beam_tree)
                anc_scores = [node_scores_s_n[a].item() for a in ancestors]
                max_anc = max(anc_scores) if anc_scores else -float('inf')
                inh = max(s_eff, max_anc)
                inherited_list.append((inh, leaf))

        # If no leaves found (rare), fallback to an “expand_any” instruction
        if not inherited_list:
            instructions.append(BeamSearchInstruction(
                node_id=-1,
                action="expand_any",
                scorer=self.scorer,
                score_bins=list(self.scorer.default_score_policy["score_bins"].values()),
                pre_temp=self.scorer.default_score_policy["pre_temp"],
                pre_top_k=self.scorer.default_score_policy["pre_top_k"],
                cull_after=self.scorer.default_score_policy["cull_after"],
                lookahead_steps=self.scorer.default_score_policy.get("lookahead_steps", 1),
                justification="PyGeoMind: no leaves to DEEPEN",
                priority=0.0,
                symbolic_id=f"pygeomind_step{current_step_id}_nodeep"
            ))
            return instructions

        # 4) Sort leaves by inherited score, descending
        inherited_list.sort(key=lambda x: x[0], reverse=True)

        # 5) Emit up to K_DEEPEN_LEAVES “expand_targeted” instructions
        num_to_expand = min(self.K_DEEPEN_LEAVES, len(inherited_list))
        for idx in range(num_to_expand):
            inh_score, pyg_leaf = inherited_list[idx]
            beam_idx = data.pyg_node_to_beam_idx[pyg_leaf].item()
            if beam_idx == -1:
                # fallback if the PyG leaf no longer has a beam index
                instructions.append(BeamSearchInstruction(
                    node_id=-1,
                    action="expand_any",
                    scorer=self.scorer,
                    score_bins=list(self.scorer.default_score_policy["score_bins"].values()),
                    pre_temp=self.scorer.default_score_policy["pre_temp"],
                    pre_top_k=self.scorer.default_score_policy["pre_top_k"],
                    cull_after=self.scorer.default_score_policy["cull_after"],
                    lookahead_steps=self.scorer.default_score_policy.get("lookahead_steps", 1),
                    justification=f"PyGeoMind: leaf {pyg_leaf} no beam_idx",
                    priority=inh_score,
                    symbolic_id=f"pygeomind_step{current_step_id}_nobeam{idx}"
                ))
            else:
                instructions.append(BeamSearchInstruction(
                    node_id=beam_idx,
                    action="expand_targeted",
                    metadata={"raw_leaf_score": node_scores_s_n[pyg_leaf].item(),
                              "inherited_leaf_score": inh_score},
                    priority=inh_score,
                    justification=f"PyGeoMind DEEPEN leaf {pyg_leaf} (beam {beam_idx}) score={inh_score:.2f}",
                    symbolic_id=f"pygeomind_step{current_step_id}_deepen{idx}",
                    scorer=self.scorer,
                    score_bins=list(self.scorer.default_score_policy["score_bins"].values()),
                    pre_temp=self.scorer.default_score_policy["pre_temp"],
                    pre_top_k=self.scorer.default_score_policy["pre_top_k"],
                    cull_after=self.scorer.default_score_policy["cull_after"],
                    lookahead_steps=self.scorer.default_score_policy.get("lookahead_steps", 1)
                ))

        # 6) Return the entire bundle of instructions
        return instructions


    def reset_context(self):
        self.context_state = None
