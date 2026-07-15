#!/usr/bin/env python3
"""A graph that grows bidirectionally from an anchor and is never "done."

Nodes hang into the substrate from an anchor sequence, one token per edge,
growing forward (append) and backward (prepend) at once. There is no
committed sentence and no finalization step -- at any point you can ask
for the current best path, but the graph keeps growing and re-scoring for
as long as you keep ticking it.

Every node gets pressure -- a live, continuously recomputed value, not an
accumulated score. Each discrete tick recomputes every node's pressure
from its own local evidence (how likely its token was when it was
created) plus flux received from its neighbors, the way current flows
through a resistor network: edges with good scores conduct flux easily,
edges with bad scores resist it. A node's pressure is what it currently
supports and is supported by -- not a running total. There is no blanket
decay; a node only loses standing if it stops having good lines running
through it (no supportive flux from neighbors, and weak local evidence of
its own), at which point it starves and, if that persists, burns off the
extremity. A node with a strong descendant never starves, because that
descendant's pressure flows back to it every tick.

Compute is the resource that actually grows the network: each tick, a
bounded number of the highest-pressure not-yet-expanded nodes get spent
generating real children (via ordinary forward next-token stepping, or
ImplicitBackpathScorer's prepend-and-rescore probe for backward nodes).
This does not require ever scoring a candidate against the *entire*
opposite-direction subtree -- a new node is scored once, locally, against
the best currently-known context on the other side, and correctness
diffuses through the graph via the tick update as both sides keep
growing, rather than needing to be exactly right at the moment a node is
created.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tensors import AbstractTensor
from .choice_policy import ChoicePolicy
from .implicit_backpath import ImplicitBackpathScorer
from .model_abstraction import AbstractModelWrapper
from .noodle_explorer import Direction
# --- END HEADER ---


@dataclass
class FluxNode:
    """One hung token, or the anchor itself (token=None, direction=None)."""

    id: int
    token: Optional[int]
    direction: Optional[Direction]
    parent_id: Optional[int]
    depth: int
    local_evidence: float  # log-prob of this token given its context at creation
    children_ids: List[int] = field(default_factory=list)
    pressure: float = 0.0
    low_pressure_ticks: int = 0
    created_tick: int = 0
    burned: bool = False
    expanded: bool = False

    @property
    def local_value(self) -> float:
        """exp(local_evidence): 1.0 for the anchor, in (0, 1] for real tokens."""
        return math.exp(self.local_evidence)


@dataclass
class FluxGraphConfig:
    found_bonus: float = 0.05
    damping: float = 0.5
    starvation_floor: float = 0.08
    burn_after_ticks: int = 3
    compute_budget_per_tick: int = 4
    branch_factor: int = 3
    max_context_tokens: int = 64
    verbose: bool = False


class FluxGraph:
    """Grows a bidirectional token graph from an anchor under a flux/pressure dynamic."""

    def __init__(
        self,
        model_wrapper: AbstractModelWrapper,
        backward_scorer: ImplicitBackpathScorer,
        choice_policy: ChoicePolicy,
        tensor_ops: AbstractTensor,
        config: Optional[FluxGraphConfig] = None,
        device: Any = None,
    ):
        self.model_wrapper = model_wrapper
        self.backward_scorer = backward_scorer
        self.choice_policy = choice_policy
        self.tensor_ops = tensor_ops
        self.device = device
        self.config = config or FluxGraphConfig()

        self.nodes: Dict[int, FluxNode] = {}
        self.anchor_id: Optional[int] = None
        self.anchor_tokens: List[int] = []
        self.tick_count = 0
        self._next_id = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def seed(self, anchor_tokens: List[int]) -> int:
        """Create the anchor node holding the fixed seed sequence."""
        node_id = self._alloc_id()
        self.anchor_tokens = list(anchor_tokens)
        self.nodes[node_id] = FluxNode(
            id=node_id,
            token=None,
            direction=None,
            parent_id=None,
            depth=0,
            local_evidence=0.0,
            pressure=1.0 + self.config.found_bonus,
            created_tick=0,
            expanded=True,  # the anchor doesn't get "expanded" itself
        )
        self.anchor_id = node_id
        return node_id

    def _alloc_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    # ------------------------------------------------------------------
    # Graph structure helpers
    # ------------------------------------------------------------------
    def _live_children(self, node_id: int) -> List[int]:
        return [c for c in self.nodes[node_id].children_ids if not self.nodes[c].burned]

    def _neighbors(self, node_id: int) -> List[int]:
        node = self.nodes[node_id]
        neighbors = list(self._live_children(node_id))
        if node.parent_id is not None and not self.nodes[node.parent_id].burned:
            neighbors.append(node.parent_id)
        return neighbors

    def path_tokens(self, node_id: int) -> Tuple[List[int], Optional[Direction]]:
        """Tokens hung between the anchor and ``node_id``, in growth order, plus that side's direction."""
        tokens: List[int] = []
        direction = self.nodes[node_id].direction
        cur = node_id
        while cur != self.anchor_id:
            node = self.nodes[cur]
            tokens.append(node.token)
            cur = node.parent_id
        tokens.reverse()
        return tokens, direction

    def full_sequence(self, forward_leaf: Optional[int], backward_leaf: Optional[int]) -> List[int]:
        """Assemble a full token sequence: backward tokens + anchor + forward tokens."""
        back_tokens: List[int] = []
        fwd_tokens: List[int] = []
        if backward_leaf is not None:
            back_tokens, _ = self.path_tokens(backward_leaf)
        if forward_leaf is not None:
            fwd_tokens, _ = self.path_tokens(forward_leaf)
        return back_tokens + self.anchor_tokens + fwd_tokens

    def best_leaf(self, direction: Optional[Direction]) -> Optional[int]:
        """The live leaf with the highest pressure on the given side (None = either side, incl. anchor)."""
        best_id = None
        best_pressure = -math.inf
        for node_id, node in self.nodes.items():
            if node.burned:
                continue
            if direction is not None and node.direction != direction:
                continue
            if direction is None and node_id == self.anchor_id:
                continue
            if self._live_children(node_id):
                continue  # only consider leaves
            if node.pressure > best_pressure:
                best_pressure = node.pressure
                best_id = node_id
        return best_id

    def best_path(self) -> Tuple[List[int], float]:
        """Live snapshot: the current best complete path and its total local-evidence score."""
        fwd_leaf = self.best_leaf(Direction.FORWARD)
        bwd_leaf = self.best_leaf(Direction.BACKWARD)
        tokens = self.full_sequence(fwd_leaf, bwd_leaf)

        total = 0.0
        for leaf in (fwd_leaf, bwd_leaf):
            cur = leaf
            while cur is not None and cur != self.anchor_id:
                total += self.nodes[cur].local_evidence
                cur = self.nodes[cur].parent_id
        return tokens, total

    # ------------------------------------------------------------------
    # Tick: pressure update, expansion, starvation
    # ------------------------------------------------------------------
    def tick(self) -> None:
        self.tick_count += 1
        self._update_pressures()
        self._expand_top_pressure_nodes()
        self._starve_and_burn()

    def _edge_conductance(self, node_id: int, neighbor_id: int) -> float:
        """Conductance of the edge between ``node_id`` and ``neighbor_id``.

        Every edge carries exactly one token, on whichever side of the
        edge is farther from the anchor (the "child side"). Conductance
        is always gated by that token's own evidence -- not by whichever
        node happens to be upstream -- so a direct child of the anchor
        still has to earn its own support rather than inheriting the
        anchor's perfect certainty for free.
        """
        node = self.nodes[node_id]
        if neighbor_id == node.parent_id:
            return node.local_value
        return self.nodes[neighbor_id].local_value

    def _update_pressures(self) -> None:
        cfg = self.config
        previous = {nid: n.pressure for nid, n in self.nodes.items() if not n.burned}
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            neighbors = self._neighbors(node_id)
            if neighbors:
                inflow = sum(
                    self._edge_conductance(node_id, n) * previous.get(n, 0.0) for n in neighbors
                ) / (1 + len(neighbors))
            else:
                inflow = 0.0
            intrinsic = node.local_value + cfg.found_bonus
            node.pressure = intrinsic + cfg.damping * inflow

    def _expandable_nodes(self) -> List[FluxNode]:
        return [
            n for n in self.nodes.values()
            if not n.burned and not n.expanded and n.id != self.anchor_id
        ]

    def _expand_top_pressure_nodes(self) -> None:
        candidates = sorted(self._expandable_nodes(), key=lambda n: n.pressure, reverse=True)
        for node in candidates[: self.config.compute_budget_per_tick]:
            self._expand_node(node.id)
            node.expanded = True

    def _starve_and_burn(self) -> None:
        cfg = self.config
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            if self._live_children(node_id):
                continue  # only extremities (current leaves) are candidates to burn
            if node.pressure < cfg.starvation_floor:
                node.low_pressure_ticks += 1
            else:
                node.low_pressure_ticks = 0
            if node.low_pressure_ticks >= cfg.burn_after_ticks:
                self._burn(node_id)

    def _burn(self, node_id: int) -> None:
        node = self.nodes[node_id]
        node.burned = True
        if self.config.verbose:
            print(f"  [burn] node {node_id} (token={node.token}, dir={node.direction}) starved out")
        if node.parent_id is not None:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)

    # ------------------------------------------------------------------
    # Expansion: generate real children via the model
    # ------------------------------------------------------------------
    def _context_window(self, tokens: List[int]) -> List[int]:
        limit = self.config.max_context_tokens
        if len(tokens) <= limit:
            return tokens
        return tokens[-limit:]

    def _expand_node(self, node_id: int) -> None:
        node = self.nodes[node_id]
        if node.direction is Direction.FORWARD:
            self._expand_forward(node_id)
        elif node.direction is Direction.BACKWARD:
            self._expand_backward(node_id)
        else:
            raise ValueError(f"node {node_id} has no direction")

    def spawn_first_children(self) -> None:
        """Seed one forward and one backward child directly off the anchor."""
        self._expand_forward(self.anchor_id)
        self._expand_backward(self.anchor_id)
        self.nodes[self.anchor_id].expanded = True

    def _expand_forward(self, node_id: int) -> None:
        prefix_tokens, _ = self.path_tokens(node_id)
        full = self._context_window(self.anchor_tokens + prefix_tokens)

        ops = self.tensor_ops
        backend_cls = type(ops)
        row = backend_cls.tensor([full], dtype=ops.long_dtype, device=self.device)
        mask = backend_cls.tensor([[1] * len(full)], dtype=ops.long_dtype, device=self.device)
        outputs = self.model_wrapper.forward(input_ids=row.data, attention_mask=mask.data)
        logits = row.ensure_tensor(outputs["logits"])
        last_logits = logits[0, -1, :].unsqueeze(0)

        scores, indices = self.choice_policy.choose(last_logits, k=self.config.branch_factor)
        self._attach_children(node_id, Direction.FORWARD, scores.tolist()[0], indices.tolist()[0])

    def _expand_backward(self, node_id: int) -> None:
        prefix_tokens, _ = self.path_tokens(node_id)  # tokens between anchor and node_id
        fwd_leaf = self.best_leaf(Direction.FORWARD)
        fwd_tokens = self.path_tokens(fwd_leaf)[0] if fwd_leaf is not None else []
        suffix = self._context_window(prefix_tokens + self.anchor_tokens + fwd_tokens)

        ops = self.tensor_ops
        suffix_t = type(ops).tensor(suffix, dtype=ops.long_dtype, device=self.device)
        pool = self.backward_scorer.candidate_pool(
            ops, vocab_size=self.backward_scorer.tokenizer.vocab_size, device=self.device
        )
        if self.config.verbose:
            print(f"  [expand-backward] node {node_id}: scoring {pool.shape[0]} candidates ...")
        raw_scores = self.backward_scorer.score_candidates(suffix_t, pool)
        top_scores, top_idx = AbstractTensor.topk(raw_scores, k=self.config.branch_factor, dim=0)
        candidate_ids = [int(pool[i].item()) for i in top_idx.tolist()]
        self._attach_children(node_id, Direction.BACKWARD, top_scores.tolist(), candidate_ids)

    def _attach_children(
        self, parent_id: int, direction: Direction, scores: List[float], token_ids: List[int]
    ) -> None:
        parent = self.nodes[parent_id]
        for score, token_id in zip(scores, token_ids):
            child_id = self._alloc_id()
            self.nodes[child_id] = FluxNode(
                id=child_id,
                token=int(token_id),
                direction=direction,
                parent_id=parent_id,
                depth=parent.depth + 1,
                local_evidence=float(score),
                pressure=self.config.found_bonus + math.exp(float(score)),
                created_tick=self.tick_count,
            )
            parent.children_ids.append(child_id)
