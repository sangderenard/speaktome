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
    # Sum of local_evidence from the anchor to this node, inclusive -- set
    # once at creation (cheap: parent's cumulative + this node's own
    # evidence), not recomputed by walking the tree each tick.
    cumulative_evidence: float = 0.0
    # Best path_mean reachable through this node's own subtree, backed up
    # from leaves toward the anchor each tick by _digest(). Starts equal
    # to the node's own path_mean before any digestion has run.
    rollup_mean: float = 0.0

    @property
    def local_value(self) -> float:
        """exp(local_evidence): 1.0 for the anchor, in (0, 1] for real tokens."""
        return math.exp(self.local_evidence)

    @property
    def path_mean(self) -> float:
        """Length-normalized quality: mean local_evidence from anchor to here."""
        if self.depth <= 0:
            return 0.0
        return self.cumulative_evidence / self.depth


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
    # Circuit: how many relaxation sweeps to run per tick before trusting
    # pressure for decisions, and how small a max-change counts as settled.
    max_relaxation_iterations: int = 25
    relaxation_tolerance: float = 1e-4
    # Digestion: how much a node's expansion priority is boosted by its
    # parent's rollup_mean (the best mean-quality path known anywhere in
    # that neighborhood) -- gives siblings of a good discovery a reason to
    # get attention even if their own instantaneous pressure is ordinary.
    rollup_weight: float = 0.3
    # Exploitation game: how much expansion priority grows per tick a node
    # has sat eligible without being expanded, so a merely-mediocre-looking
    # node isn't ignored forever just because something else looked better
    # first.
    exploration_constant: float = 0.05
    # Real GPT-2's own document-boundary token (<|endoftext|>), or whatever
    # else the model wrapper's tokenizer uses for "start of something new".
    # Every backward candidate is, by construction, being scored with
    # nothing to its own left -- without this, that means literally zero
    # context, a regime the model rarely saw cleanly during training (most
    # training windows are mid-document, not document starts). Left as
    # None by default since it's model-specific, not a graph-algorithm
    # parameter; callers building a real GPT-2 graph should set it.
    backward_left_context: Optional[List[int]] = None


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
        """The live leaf with the best length-normalized path quality on the given side.

        Selected by path_mean (mean local_evidence from the anchor to this
        leaf), not raw pressure -- pressure is the exploit/explore signal
        that drives *where compute goes*, a different question from "what's
        actually the best thing found so far". Using pressure here made a
        long-but-mediocre chain look like it kept improving as best_path()
        just because pressure doesn't decay with depth; path_mean answers
        the quality question directly, independent of how long the path is.
        """
        best_id = None
        best_mean = -math.inf
        for node_id, node in self.nodes.items():
            if node.burned:
                continue
            if direction is not None and node.direction != direction:
                continue
            if direction is None and node_id == self.anchor_id:
                continue
            if self._live_children(node_id):
                continue  # only consider leaves
            if node.path_mean > best_mean:
                best_mean = node.path_mean
                best_id = node_id
        return best_id

    def best_path(self) -> Tuple[List[int], float]:
        """Live snapshot: the current best complete path and its mean per-token quality.

        The score is a length-normalized mean (average local_evidence per
        token across whichever side(s) currently exist), not a raw sum --
        a raw sum mechanically gets worse as a path gets longer regardless
        of whether the *quality* per token is improving, which made a
        growing path look like it was failing even when each new token was
        individually reasonable.
        """
        fwd_leaf = self.best_leaf(Direction.FORWARD)
        bwd_leaf = self.best_leaf(Direction.BACKWARD)
        tokens = self.full_sequence(fwd_leaf, bwd_leaf)

        total = 0.0
        count = 0
        for leaf in (fwd_leaf, bwd_leaf):
            cur = leaf
            while cur is not None and cur != self.anchor_id:
                total += self.nodes[cur].local_evidence
                count += 1
                cur = self.nodes[cur].parent_id
        mean_score = total / count if count > 0 else 0.0
        return tokens, mean_score

    # ------------------------------------------------------------------
    # Tick: pressure update, expansion, starvation
    # ------------------------------------------------------------------
    def tick(self) -> None:
        self.tick_count += 1
        self._settle_circuit()
        self._digest()
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

    def _update_pressures(self) -> float:
        """One relaxation sweep. Returns the largest pressure change seen."""
        cfg = self.config
        previous = {nid: n.pressure for nid, n in self.nodes.items() if not n.burned}
        max_delta = 0.0
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
            new_pressure = intrinsic + cfg.damping * inflow
            max_delta = max(max_delta, abs(new_pressure - node.pressure))
            node.pressure = new_pressure
        return max_delta

    def _settle_circuit(self) -> int:
        """Iterate relaxation until pressure stops moving (or the iteration cap).

        A single sweep is a half-propagated transient, not a solved
        circuit -- multi-hop support hasn't had a chance to arrive yet.
        This is pure local arithmetic (no model calls), so iterating it
        many times per tick is cheap; only the *decisions* made off of it
        (expansion, starvation) are expensive, and those should see a
        settled state, not a snapshot mid-flight.
        """
        cfg = self.config
        for i in range(cfg.max_relaxation_iterations):
            delta = self._update_pressures()
            if delta < cfg.relaxation_tolerance:
                return i + 1
        return cfg.max_relaxation_iterations

    def _digest(self) -> None:
        """Back up the best reachable path quality from leaves toward the anchor.

        Pressure alone never lets a distant discovery make its ancestors
        look better -- it only propagates gradually, hop by hop, and can
        get out-competed for compute along the way. Digestion is a direct,
        explicit rollup: every node learns the best mean-quality (length-
        normalized) path found anywhere in its own subtree, in one pass,
        so that information can inform expansion priority immediately
        rather than waiting on the circuit to carry it there.
        """
        live_nodes = [n for n in self.nodes.values() if not n.burned and n.id != self.anchor_id]
        for node in sorted(live_nodes, key=lambda n: n.depth, reverse=True):
            child_rollups = [self.nodes[c].rollup_mean for c in self._live_children(node.id)]
            node.rollup_mean = max([node.path_mean] + child_rollups)

        anchor = self.nodes[self.anchor_id]
        child_rollups = [self.nodes[c].rollup_mean for c in self._live_children(self.anchor_id)]
        anchor.rollup_mean = max(child_rollups) if child_rollups else 0.0

    def _expansion_priority(self, node: FluxNode) -> float:
        """Pressure (exploit) + neighborhood rollup (digested value) + wait-time bonus (explore).

        Pure top-pressure selection is greedy exploitation: whatever looks
        best right now always wins, forever. The neighborhood term uses
        the *parent's* rollup (not the candidate's own -- a fresh leaf's
        own rollup is trivially just itself) so siblings of a known-good
        discovery get a boost. The wait-time term grows with ticks spent
        eligible-but-unpicked, so a merely-ordinary node isn't starved of
        its turn forever just because something else looked better first.
        """
        cfg = self.config
        parent = self.nodes[node.parent_id] if node.parent_id is not None else None
        neighborhood_bonus = cfg.rollup_weight * math.exp(parent.rollup_mean) if parent is not None else 0.0
        wait = max(0, self.tick_count - node.created_tick)
        exploration_bonus = cfg.exploration_constant * math.sqrt(wait)
        return node.pressure + neighborhood_bonus + exploration_bonus

    def _expandable_nodes(self) -> List[FluxNode]:
        # Eligibility is "currently a leaf" (no live children right now),
        # not "has never been expanded" -- a node whose children all
        # burned off must be able to try again, or it becomes a permanent
        # dead end that can win best_path() forever without ever being
        # able to grow past it.
        return [
            n for n in self.nodes.values()
            if not n.burned and n.id != self.anchor_id and not self._live_children(n.id)
        ]

    def _expand_top_pressure_nodes(self) -> None:
        candidates = sorted(self._expandable_nodes(), key=self._expansion_priority, reverse=True)
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
        raw_scores = self.backward_scorer.score_candidates(
            suffix_t, pool, left_context=self.config.backward_left_context
        )
        top_scores, top_idx = AbstractTensor.topk(raw_scores, k=self.config.branch_factor, dim=0)
        candidate_ids = [int(pool[i].item()) for i in top_idx.tolist()]
        self._attach_children(node_id, Direction.BACKWARD, top_scores.tolist(), candidate_ids)

    def _attach_children(
        self, parent_id: int, direction: Direction, scores: List[float], token_ids: List[int]
    ) -> None:
        parent = self.nodes[parent_id]
        for score, token_id in zip(scores, token_ids):
            child_id = self._alloc_id()
            cumulative = parent.cumulative_evidence + float(score)
            new_depth = parent.depth + 1
            self.nodes[child_id] = FluxNode(
                id=child_id,
                token=int(token_id),
                direction=direction,
                parent_id=parent_id,
                depth=new_depth,
                local_evidence=float(score),
                pressure=self.config.found_bonus + math.exp(float(score)),
                created_tick=self.tick_count,
                cumulative_evidence=cumulative,
                rollup_mean=cumulative / new_depth,
            )
            parent.children_ids.append(child_id)
