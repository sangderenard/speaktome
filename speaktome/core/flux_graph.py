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

Even the anchor isn't permanent. If a non-anchor node's live pressure
ever exceeds the current anchor's, that node displaces it: the tree is
re-rooted there (see FluxGraph._reroot), forward/backward are
reinterpreted relative to the new root -- what was "toward the old
anchor" becomes the new root's opposite direction, and the old anchor
folds into that reinterpreted lineage, now a perfectly ordinary node,
newly subject to the same starvation/burning as anything else. This is
still one graph, one topology, one continuous line of real compute --
re-rooting only changes which node is treated as root. Nothing is ever
detached, extracted, or stopped: any subtree that branched off *between*
the new root and the old anchor, on the side not carrying the new root,
keeps its old direction and just sits there, still fully live, still
eligible to become anchor itself later if its own pressure ever earns
it. Such a subtree no longer reads as a clean two-hemisphere
forward/backward layout (see FluxGraph.orthogonal_node_ids) -- that's a
pure display question, not a reason to remove it from the graph.
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
from .poetic_attractor import PoeticAttractor
from .word_trie import WordTrie, TrieGate
from .word_boundary import starts_new_word
# --- END HEADER ---


@dataclass
class FluxNode:
    """One hung word (a span of one or more tokens), or the anchor itself (tokens=[], direction=None).

    A span longer than one token only ever occurs when FluxGraphConfig.word_trie
    is set -- without it, every edge is exactly one BPE token, same as before.
    With it, an edge is a whole word: the physics (pressure, expansion
    priority, starvation) never sees a mid-word fragment as a real, complete
    prediction, only the finished word once subword growth resolves it.
    """

    id: int
    tokens: List[int]
    direction: Optional[Direction]
    parent_id: Optional[int]
    depth: int
    local_evidence: float  # mean log-prob of this word's tokens given its context at creation
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
    # Strongest quality-weighted source within this node's own subtree,
    # decayed per hop -- the bottom-up half of the auxin computation
    # (_diffuse_auxin). Internal bookkeeping only; branching decisions read
    # auxin_level, not this.
    subtree_auxin: float = 0.0
    # Ambient suppression this node feels from *outside* its own subtree --
    # a strong tip elsewhere (a sibling's descendant, or further still)
    # reaches here attenuated by hop distance. Never includes this node's
    # own subtree's contribution, matching real apical dominance: a growing
    # tip doesn't inhibit itself, only lateral buds elsewhere on the plant.
    auxin_level: float = 0.0

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

    @property
    def height(self) -> float:
        """Signed token-distance from the anchor: +depth forward, -depth backward, 0 at the anchor.

        Pure topology -- no simulation, no dynamic force, just where a node
        sits in the tree. This is the single quantity both the live
        visualizer (pins a node's on-screen vertical position to it) and the
        real pressure equation (head_pressure_coefficient, in
        _update_pressures) read, so "what it looks like" and "what it costs"
        stay the same number instead of two things that happen to agree.
        """
        if self.direction is Direction.FORWARD:
            return float(self.depth)
        if self.direction is Direction.BACKWARD:
            return float(-self.depth)
        return 0.0


@dataclass
class FluxGraphConfig:
    found_bonus: float = 0.05
    damping: float = 0.5
    starvation_floor: float = 0.08
    burn_after_ticks: int = 3
    # Off by default: the anchor's own pressure is frozen (never
    # recomputed, never starves) the way every other node's is. Turning
    # this on makes the anchor a normal citizen of the pressure network --
    # if it goes without support for burn_after_ticks ticks (same floor,
    # same patience as an ordinary node), it can't just sit there forever;
    # _maybe_reroot force-picks the current highest-pressure live node as
    # the next anchor from whatever's actually in the graph right now,
    # rather than waiting for a challenger to out-score it fairly.
    anchor_can_decay: bool = False
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
    # Block a candidate token if using it would recreate an n-gram that
    # already exists in the node's known context (backward + anchor +
    # forward). Length-normalizing best_path (path_mean) fixes a *different*
    # failure mode -- a mechanical penalty on longer paths -- but does
    # nothing about induction-head repetition loops: once a phrase repeats
    # once, a transformer's own attention makes repeating it again *more*
    # confident, not less, so a length-normalized score still walks straight
    # into the loop. This is a hard, cheap veto on that exact case. 0 or
    # None disables it.
    no_repeat_ngram_size: Optional[int] = 3
    # Every node picked for expansion this tick -- forward and backward,
    # regardless of node -- is scored in one shared, padded, chunked batch
    # instead of one model call per node. This is the row cap per model
    # call within that shared batch (a backward node alone can contribute
    # tens of thousands of candidate rows), not a count of nodes.
    expand_batch_chunk_size: int = 2048
    # expand_batch_chunk_size alone is a *flat* row-count cap -- real GPU
    # memory for one forward call scales with rows x row_len x vocab_size,
    # and row_len grows over a run as context accumulates, so a flat row
    # cap that's safe early in a run can silently become a multi-GB
    # request later without expand_batch_chunk_size ever changing. This
    # bounds the *product* directly: the effective row cap for a given
    # chunk shrinks as that chunk's own row_len grows, so no single
    # forward call this triggers can exceed roughly this many
    # (row x vocab-position) elements, regardless of how long the graph's
    # context has grown. None (default) preserves the original flat-cap
    # behavior exactly -- this is opt-in, not a change to existing
    # configs. Also applied to backward word growth's per-step scoring
    # calls (see _grow_backward_word), which share the same underlying
    # concern at a smaller scale. A real GPT-2 run that OOM'd here even
    # with expand_batch_chunk_size respected (2048 rows x ~20 growing
    # row_len x 50257 vocab in float32 is already multi-GB) is what this
    # exists to prevent; tune to your GPU's actual headroom, not a
    # one-size-fits-all constant.
    max_expand_elements: Optional[int] = None
    # Optional rhyme/alliteration bias over candidate selection. None
    # disables it entirely (default -- this is a bias on top of the real
    # model, not a replacement for it). When set, it never touches
    # local_evidence/pressure -- see _apply_poetic_rerank -- it only
    # re-ranks which of the model's own top candidates get kept.
    poetic_attractor: Optional[PoeticAttractor] = None
    # How much weight the poetic bonus gets relative to real log-prob
    # evidence when re-ranking a shortlist (same units as local_evidence).
    poetic_scale: float = 1.0
    # How many of the model's own top candidates are even considered for
    # poetic re-ranking. Keeps the (comparatively expensive: decode text +
    # string heuristics per candidate) poetic scoring off the full 50k-token
    # backward pool and off the full forward vocab -- only the shortlist the
    # model already ranked highest is eligible to be re-ordered by rhyme.
    poetic_shortlist_k: int = 20
    # If set, an edge is a whole word, not one BPE token: the physics never
    # sees a mid-word fragment as a complete prediction. A round-0 candidate
    # (one subtoken) is grown by a bounded, branch_factor-wide beam search --
    # branching at every subword step, not committing greedily -- pruned by
    # this trie's is_prefix check (forward only; see _grow_backward_word for
    # why backward skips mid-growth pruning) and terminated by GPT-2's own
    # leading-space word-boundary convention (word_boundary.starts_new_word).
    # None (default) disables all of this: every edge stays exactly one
    # token, identical to pre-word-growth behavior.
    word_trie: Optional[WordTrie] = None
    # Backward growth discovers a word from its end toward its start (each
    # new token gets *prepended*), so validating against word_trie (built
    # prefix-wise) would ask a prefix question about what's actually a
    # suffix-in-progress -- it needs a second trie built over reversed
    # word strings (WordTrie(..., reverse=True)) to ask the right
    # question. None (default) means backward growth falls back to
    # scoring its whole candidate pool every step with no trie narrowing
    # or validation at all, matching pre-trie-gating behavior.
    backward_word_trie: Optional[WordTrie] = None
    # Hard cap on how many subtokens one word's growth can consume before
    # being force-finalized as-is, regardless of whether a boundary was
    # found -- safety against a pathological run of continuation pieces
    # that never resolves to a real word boundary.
    max_subword_steps: int = 8
    # Apical-dominance-style branching suppression. branch_factor today is
    # one constant applied identically everywhere, regardless of whether a
    # strong, uncontested tip already exists nearby -- auxin makes it
    # context-sensitive: a node's *final* child count shrinks the more
    # ambient auxin (see FluxNode.auxin_level) it feels from outside its own
    # subtree. 0 disables the effect entirely (branch_factor stays constant,
    # pre-auxin behavior); higher values suppress harder.
    auxin_suppression: float = 0.0
    # Attenuation per hop as auxin propagates away from its source, in both
    # the bottom-up (subtree) and top-down (ambient) passes. 1.0 = no
    # attenuation (a distant tip suppresses as hard as an adjacent one);
    # near 0 = only immediate neighbors feel any suppression at all.
    auxin_decay: float = 0.6
    # Head pressure: sustaining flow to a point further from the anchor
    # costs more than to a nearby one, modeled on real xylem transport
    # (lifting water against a taller column of it costs more head
    # pressure). Each node's pressure is reduced by
    # head_pressure_coefficient * abs(node.height) -- abs(), not signed,
    # so forward and backward pay the same elevation cost at equal
    # distance; this is a real resistance term in the pressure equation
    # itself, not just a reporting-time normalization the way path_mean's
    # length-normalization already is. 0 (default) disables it.
    head_pressure_coefficient: float = 0.0


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
        # A complete, self-consistent snapshot published wholesale at the
        # end of seed()/spawn_first_children()/tick() -- never touched
        # mid-mutation. An external reader (FluxGraphVisualizer) just reads
        # this reference directly, no lock needed: in CPython, a single
        # attribute read/write is already atomic under the GIL, so a reader
        # either sees the previous complete snapshot or the new complete
        # one, never a torn one. This replaces an earlier lock-based design
        # that had two real problems: spawn_first_children() never held the
        # lock at all (so a reader could freeze on a snapshot mid-way
        # through it -- forward child added, backward not yet), and even
        # during tick(), a non-blocking acquire racing a tight loop that
        # immediately re-locks between ticks could statistically starve the
        # reader for the whole run. Publishing removes the race instead of
        # tuning around it.
        self.published_snapshot: Optional[Dict[int, Tuple[Optional[int], bool, Optional[Direction], float]]] = None
        # One TrieGate per distinct WordTrie (forward vs backward-reversed),
        # built lazily on first use and reused for the graph's whole
        # lifetime -- the underlying candidate pool (which vocab ids are
        # dictionary/writing-clean at all) never changes tick to tick, so
        # there's no reason to redo that decode-and-classify pass more than
        # once. Keyed by id(trie) rather than the trie object itself since
        # WordTrie isn't hashable.
        self._trie_gates: Dict[int, TrieGate] = {}

    def _get_word_trie_gate(self, trie: WordTrie) -> TrieGate:
        key = id(trie)
        gate = self._trie_gates.get(key)
        if gate is None:
            ops = self.tensor_ops
            pool = self.backward_scorer.candidate_pool(
                ops, vocab_size=self.backward_scorer.tokenizer.vocab_size, device=self.device
            )
            gate = TrieGate(trie, self.backward_scorer.tokenizer, pool.tolist())
            self._trie_gates[key] = gate
        return gate

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def seed(self, anchor_tokens: List[int]) -> int:
        """Create the anchor node holding the fixed seed sequence."""
        node_id = self._alloc_id()
        self.anchor_tokens = list(anchor_tokens)
        self.nodes[node_id] = FluxNode(
            id=node_id,
            tokens=[],
            direction=None,
            parent_id=None,
            depth=0,
            local_evidence=0.0,
            pressure=1.0 + self.config.found_bonus,
            created_tick=0,
            expanded=True,  # the anchor doesn't get "expanded" itself
        )
        self.anchor_id = node_id
        self._publish_snapshot()
        return node_id

    def _alloc_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def _publish_snapshot(self) -> None:
        """Publish a complete, ready-to-read snapshot for external readers.

        Called at the end of seed()/spawn_first_children()/tick() -- every
        point where self.nodes is left in a consistent state -- never from
        mid-mutation. See published_snapshot's own docstring for why this
        replaces a lock.
        """
        self.published_snapshot = {
            nid: (n.parent_id, n.burned, n.direction, n.height) for nid, n in self.nodes.items()
        }

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
        """Tokens hung between the anchor and ``node_id``, in growth order, plus that side's direction.

        Walks child-to-parent (leaf up to the anchor), so the per-node spans
        are collected in reverse growth order -- reverse the list of spans,
        not each span's own internal token order, before flattening, or a
        multi-token word gets its own tokens scrambled.
        """
        spans: List[List[int]] = []
        direction = self.nodes[node_id].direction
        cur = node_id
        while cur != self.anchor_id:
            node = self.nodes[cur]
            spans.append(node.tokens)
            cur = node.parent_id
        spans.reverse()
        tokens = [t for span in spans for t in span]
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
        """Advance one discrete step."""
        self.tick_count += 1
        self._settle_circuit()
        self._maybe_reroot()
        self._digest()
        self._diffuse_auxin()
        self._expand_top_pressure_nodes()
        self._starve_and_burn()
        self._publish_snapshot()

    # ------------------------------------------------------------------
    # Re-rooting: the anchor itself can be displaced by a higher-pressure node
    # ------------------------------------------------------------------
    def _maybe_reroot(self) -> None:
        """Displace the anchor if some live node's pressure now exceeds it.

        Checked once per tick, right after pressure settles (so the
        comparison uses converged values, not a mid-relaxation snapshot)
        and before digestion/auxin/expansion/starvation (so those all see
        the post-rerooting tree -- correct anchor_id, correct depths --
        rather than operating on stale structure for one tick). If several
        live nodes exceed the anchor, the single highest-pressure one wins;
        ties are broken by whichever this dict iteration reaches first,
        which is insertion order (oldest node id) in practice -- not a
        documented guarantee, just a note for anyone tracing a specific run.

        If ``config.anchor_can_decay`` is on, the anchor itself is also
        tracked for starvation exactly like any other node (same
        starvation_floor/burn_after_ticks patience -- see _update_pressures'
        matching exemption toggle). Once it's gone that long without
        support, it can't just sit there forever the way an unsupported
        leaf can't: this force-picks the current highest-pressure live node
        as the next anchor from whatever's actually in the graph right now,
        rather than waiting for a challenger to fairly out-score it.
        """
        anchor = self.nodes[self.anchor_id]

        if self.config.anchor_can_decay:
            if anchor.pressure < self.config.starvation_floor:
                anchor.low_pressure_ticks += 1
            else:
                anchor.low_pressure_ticks = 0
            if anchor.low_pressure_ticks >= self.config.burn_after_ticks:
                replacement_id = self._best_replacement_anchor()
                if replacement_id is not None:
                    self._reroot(replacement_id)
                return

        challenger_id = None
        challenger_pressure = anchor.pressure
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            if node.pressure > challenger_pressure:
                challenger_pressure = node.pressure
                challenger_id = node_id
        if challenger_id is None:
            return
        self._reroot(challenger_id)

    def _best_replacement_anchor(self) -> Optional[int]:
        """The current highest-pressure live non-anchor node, or None if there isn't one.

        The forced pick used when the anchor itself has decayed away (see
        anchor_can_decay) -- unlike the normal challenger scan in
        _maybe_reroot, this doesn't require beating anything, just being
        the best of whatever the current node landscape actually has.
        """
        best_id = None
        best_pressure = -math.inf
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            if node.pressure > best_pressure:
                best_pressure = node.pressure
                best_id = node_id
        return best_id

    def _reroot(self, new_root_id: int) -> None:
        """Re-root the tree at ``new_root_id``.

        Walks from ``new_root_id`` up to the current anchor, reversing
        parent/child pointers and flipping direction along that path --
        what was "toward the old anchor" from the new root's own original
        side is now the new root's *opposite* direction. Every other node
        in the graph -- including branches whose direction no longer
        lines up with this new orientation -- is left exactly where it
        is: nothing is detached, extracted, or reset except the two nodes
        actually changing anchor status (new_root promoted, old anchor
        demoted). The graph stays one connected tree, and every node
        stays fully live and eligible to become anchor itself later (see
        _maybe_reroot's scan over all of self.nodes). Which nodes no
        longer read as a clean two-hemisphere forward/backward layout is
        a pure display question -- see orthogonal_node_ids -- not
        something this method needs to compute or care about.
        """
        old_anchor_id = self.anchor_id
        if new_root_id == old_anchor_id:
            return
        old_anchor_tokens = self.anchor_tokens

        new_root = self.nodes[new_root_id]
        original_direction = new_root.direction
        flipped_direction = (
            Direction.BACKWARD if original_direction is Direction.FORWARD else Direction.FORWARD
        )

        # Walk the path from new_root up to the old anchor, before any
        # pointer gets mutated.
        path: List[int] = [new_root_id]
        cur = new_root_id
        while cur != old_anchor_id:
            cur = self.nodes[cur].parent_id
            path.append(cur)

        # Reverse parent/child pointers along path[0..-1] -> path[-1].
        for i in range(len(path) - 1):
            cur_id, next_id = path[i], path[i + 1]
            cur_node, next_node = self.nodes[cur_id], self.nodes[next_id]
            if cur_id in next_node.children_ids:
                next_node.children_ids.remove(cur_id)
            cur_node.children_ids.append(next_id)
            next_node.parent_id = cur_id

        # Flip direction for every reversed-path node (all but the new
        # root itself, which becomes direction=None as the anchor).
        for node_id in path[1:]:
            self.nodes[node_id].direction = flipped_direction

        # Promote new_root to anchor. The demoted old anchor never had its
        # own single-edge token span (it *was* the fixed seed, not
        # something grown) -- without this, the entire original seed text
        # would silently vanish from any path that walks through it now
        # that it's an ordinary node. Its captured former anchor_tokens
        # becomes that node's own (multi-token, same as any word-growth
        # span) edge content.
        self.nodes[old_anchor_id].tokens = old_anchor_tokens
        self.anchor_tokens = self._reset_to_anchor_invariants(new_root)
        self.anchor_id = new_root_id
        self._recompute_depths_from(new_root_id)

    def _reset_to_anchor_invariants(self, node: FluxNode) -> List[int]:
        """Match seed()'s anchor construction, in place, for a node that's becoming the anchor.

        Returns the node's own tokens as they were just before this reset
        -- the caller uses that as the new anchor_tokens. Deliberately
        leaves pressure, children_ids, and created_tick alone -- pressure
        is why this node was promoted in the first place (no reason to
        discard that live signal), children_ids/created_tick are real
        facts about this node's own history that promotion doesn't change.
        """
        captured_tokens = node.tokens
        node.tokens = []
        node.direction = None
        node.parent_id = None
        node.depth = 0
        node.local_evidence = 0.0
        node.cumulative_evidence = 0.0
        node.low_pressure_ticks = 0
        node.expanded = True
        return captured_tokens

    def _recompute_depths_from(self, root_id: int) -> None:
        """BFS from ``root_id``, fixing depth/cumulative_evidence for the whole tree.

        Needed after re-rooting: both quantities are defined relative to
        whichever node is being treated as the anchor, and re-rooting
        shifts that anchor. The graph is always one connected tree (see
        _reroot), so a single BFS from the new anchor reaches every node,
        including branches now orthogonal to the new orientation.
        """
        root = self.nodes[root_id]
        root.depth = 0
        root.cumulative_evidence = 0.0
        stack = [root_id]
        while stack:
            cur_id = stack.pop()
            cur = self.nodes[cur_id]
            for child_id in cur.children_ids:
                child = self.nodes[child_id]
                child.depth = cur.depth + 1
                child.cumulative_evidence = cur.cumulative_evidence + child.local_evidence
                stack.append(child_id)

    def orthogonal_node_ids(self) -> "set[int]":
        """Node ids whose direction lineage no longer aligns with the current anchor.

        A pure, stateless display query -- never used by pressure,
        expansion, or starvation, and never mutates anything. Re-rooting
        (see _reroot) never removes or partitions nodes; it only changes
        which node is treated as anchor, so a subtree grown under a
        now-superseded orientation can end up with a direction that no
        longer matches its (new) parent's. The anchor's own two direct
        children are always exempt -- they're definitionally the roots of
        the current forward/backward hemispheres -- and orthogonality
        propagates downward: once a node disagrees with its parent, its
        whole subtree is orthogonal too, since direction never changes
        again below that point.
        """
        orthogonal: set = set()
        anchor = self.nodes[self.anchor_id]
        stack = list(anchor.children_ids)
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                if node_id in orthogonal or child.direction != node.direction:
                    orthogonal.add(child_id)
                stack.append(child_id)
        return orthogonal

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
            if node.burned:
                continue
            if node_id == self.anchor_id and not cfg.anchor_can_decay:
                continue
            neighbors = self._neighbors(node_id)
            if neighbors:
                inflow = sum(
                    self._edge_conductance(node_id, n) * previous.get(n, 0.0) for n in neighbors
                ) / (1 + len(neighbors))
            else:
                inflow = 0.0
            intrinsic = node.local_value + cfg.found_bonus
            head_cost = cfg.head_pressure_coefficient * abs(node.height)
            new_pressure = max(0.0, intrinsic + cfg.damping * inflow - head_cost)
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

    def _diffuse_auxin(self) -> None:
        """Apical-dominance-style suppression: a strong tip dampens branching elsewhere.

        Two passes, same shape as _digest but propagating a different
        quantity. Bottom-up: each node's subtree_auxin is the strongest
        quality-weighted source anywhere within its own subtree (a live
        leaf's own path_mean is its source; internal nodes only relay,
        decayed per hop). Top-down: each node's auxin_level is the ambient
        field it feels from *outside* its own subtree -- the strongest of
        its live siblings' subtree_auxin, or whatever's already flowing
        down into its parent, decayed once more. A node never includes its
        own subtree's contribution in its own auxin_level, so a growing tip
        never suppresses itself, only competing branches elsewhere -- real
        apical dominance doesn't stunt the leader shoot, only lateral buds.
        """
        cfg = self.config
        if cfg.auxin_suppression <= 0:
            return
        live_nodes = [n for n in self.nodes.values() if not n.burned and n.id != self.anchor_id]

        for node in sorted(live_nodes, key=lambda n: n.depth, reverse=True):
            live_children = self._live_children(node.id)
            own_source = math.exp(node.path_mean) if not live_children else 0.0
            child_best = max((self.nodes[c].subtree_auxin for c in live_children), default=0.0)
            node.subtree_auxin = max(own_source, child_best * cfg.auxin_decay)

        for node in sorted(live_nodes, key=lambda n: n.depth):
            parent_id = node.parent_id
            siblings = [
                c for c in self.nodes[parent_id].children_ids
                if c != node.id and not self.nodes[c].burned
            ]
            siblings_best = max((self.nodes[s].subtree_auxin for s in siblings), default=0.0)
            parent_ambient = 0.0 if parent_id == self.anchor_id else self.nodes[parent_id].auxin_level
            node.auxin_level = cfg.auxin_decay * max(siblings_best, parent_ambient)

    def _effective_branch_factor(self, node: FluxNode) -> int:
        """branch_factor, suppressed by ambient auxin felt at this node.

        Only the *final* number of children a node commits to shrinks --
        internal search width (round-0 shortlists, word-growth's own beam)
        stays at the configured branch_factor regardless, so a suppressed
        node still searches broadly before narrowing down to fewer winners.
        """
        cfg = self.config
        if cfg.auxin_suppression <= 0:
            return cfg.branch_factor
        suppression = 1.0 / (1.0 + cfg.auxin_suppression * node.auxin_level)
        return max(1, round(cfg.branch_factor * suppression))

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
        selected = candidates[: self.config.compute_budget_per_tick]
        if selected:
            self._expand_batch(selected)
        for node in selected:
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
            print(f"  [burn] node {node_id} (tokens={node.tokens}, dir={node.direction}) starved out")
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

    def _repeated_ngram_tokens(self, context: List[int], side: str) -> set:
        """Token ids that would recreate an n-gram already present in ``context``.

        ``side`` is "append" (forward: the candidate becomes the *last*
        token of the new n-gram) or "prepend" (backward: the candidate
        becomes the *first* token, since prepending grows the sequence
        leftward). Standard no-repeat-ngram check: an n-gram is blocked only
        if it exactly matches one already seen, so this stops loops without
        forbidding merely-similar phrasing.
        """
        n = self.config.no_repeat_ngram_size
        if not n or n < 2 or len(context) < n - 1:
            return set()
        blocked: set = set()
        if side == "append":
            prefix = tuple(context[-(n - 1):])
            for i in range(len(context) - n + 1):
                window = context[i:i + n]
                if tuple(window[:-1]) == prefix:
                    blocked.add(window[-1])
        else:
            tail = tuple(context[: n - 1])
            for i in range(len(context) - n + 1):
                window = context[i:i + n]
                if tuple(window[1:]) == tail:
                    blocked.add(window[0])
        return blocked

    def _plan_expand_chunks(self, row_lens: List[int], vocab_size: int) -> List[Tuple[int, int]]:
        """Split row indices into chunk (start, end) boundaries, one model call each.

        Greedy left-to-right: grow the current chunk while adding the next
        row keeps it under both expand_batch_chunk_size (row-count cap)
        and, if config.max_expand_elements is set, that element budget --
        a chunk's real memory cost is set by its *longest* row, since
        every row in a chunk gets padded to match, so the check uses the
        running max row_len seen so far in the chunk, not each row's own
        length. A single row that alone exceeds the element budget still
        gets its own one-row chunk rather than being dropped or raising --
        there's no correct way to serve it smaller than one row, and
        refusing to make progress at all would be worse than the request
        it's trying to bound.
        """
        cfg = self.config
        n = len(row_lens)
        if n == 0:
            return []
        if cfg.max_expand_elements is None:
            chunk_size = cfg.expand_batch_chunk_size
            return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]

        chunks: List[Tuple[int, int]] = []
        start = 0
        chunk_max_len = 0
        for i in range(n):
            candidate_len = max(chunk_max_len, row_lens[i])
            candidate_count = i - start + 1
            over_row_cap = candidate_count > cfg.expand_batch_chunk_size
            over_element_budget = candidate_count * candidate_len * vocab_size > cfg.max_expand_elements
            if (over_row_cap or over_element_budget) and candidate_count > 1:
                chunks.append((start, i))
                start = i
                chunk_max_len = row_lens[i]
            else:
                chunk_max_len = candidate_len
        chunks.append((start, n))
        return chunks

    def _expand_batch(self, nodes: List[FluxNode]) -> None:
        """Expand every node selected this tick in one shared, padded batch.

        Calling ``_expand_forward``/``_expand_backward`` once per node (the
        old path, still kept below for ``spawn_first_children`` and direct
        testing) means every node pays for its own separate model call --
        observably, on a real GPT-2 run, multiple full-vocab backward scoring
        passes back to back in the same tick. There is no reason forward and
        backward rows can't be padded to a shared length and scored in the
        same chunked pass: the model doesn't care what a row is *for*, only
        that every row in a call is the same length. This collects every
        row every selected node needs (one row per forward node, one row per
        backward candidate), pads and chunks them together, and only after
        every chunk is scored does it interpret each row back into its
        node's terms and attach children.
        """
        ops = self.tensor_ops
        backend_cls = type(ops)
        long_dtype = ops.long_dtype
        device = self.device

        fwd_leaf = self.best_leaf(Direction.FORWARD)
        fwd_tokens_cache = self.path_tokens(fwd_leaf)[0] if fwd_leaf is not None else []
        left_context = list(self.config.backward_left_context) if self.config.backward_left_context else []
        offset = len(left_context)

        rows: List[List[int]] = []
        row_kind: List[str] = []
        row_node: List[int] = []
        row_candidate: List[Optional[int]] = []
        row_suffix: List[List[int]] = []

        # Forward rows first, then backward -- keeps the only cross-kind
        # padding-waste boundary in the whole batch to at most one chunk,
        # instead of one per node interleaving. Within backward, each
        # node's candidates stay contiguous (built in one pass per node),
        # which is what lets the post-scoring gather below batch an entire
        # node's candidates as a single tensor op.
        for node in nodes:
            if node.direction is not Direction.FORWARD:
                continue
            prefix_tokens, _ = self.path_tokens(node.id)
            full = self._context_window(self.anchor_tokens + prefix_tokens)
            rows.append(full)
            row_kind.append("forward")
            row_node.append(node.id)
            row_candidate.append(None)
            row_suffix.append([])

        for node in nodes:
            if node.direction is Direction.FORWARD:
                continue
            if node.direction is not Direction.BACKWARD:
                raise ValueError(f"node {node.id} has no direction")
            prefix_tokens, _ = self.path_tokens(node.id)
            suffix = self._context_window(prefix_tokens + self.anchor_tokens + fwd_tokens_cache)
            pool = self.backward_scorer.candidate_pool(
                ops, vocab_size=self.backward_scorer.tokenizer.vocab_size, device=device
            )
            pool_list = pool.tolist()
            blocked = self._repeated_ngram_tokens(suffix, "prepend")
            if blocked:
                pool_list = [t for t in pool_list if t not in blocked]
            if self.config.verbose:
                print(f"  [expand-backward] node {node.id}: scoring {len(pool_list)} candidates ...")
            for cand in pool_list:
                rows.append(left_context + [cand] + suffix)
                row_kind.append("backward")
                row_node.append(node.id)
                row_candidate.append(cand)
                row_suffix.append(suffix)

        if not rows:
            return

        forward_logits: Dict[int, AbstractTensor] = {}
        backward_scores: Dict[int, List[Tuple[int, float]]] = {}

        vocab_size = self.backward_scorer.tokenizer.vocab_size
        all_row_lens = [len(r) for r in rows]
        for start, end in self._plan_expand_chunks(all_row_lens, vocab_size):
            chunk_rows = rows[start:end]
            chunk_kind = row_kind[start:end]
            chunk_node = row_node[start:end]
            chunk_candidate = row_candidate[start:end]
            chunk_suffix = row_suffix[start:end]
            n_chunk = len(chunk_rows)

            row_lens = [len(r) for r in chunk_rows]
            pad_len = max(row_lens)
            padded = [r + [0] * (pad_len - len(r)) for r in chunk_rows]
            mask = [[1] * l + [0] * (pad_len - l) for l in row_lens]

            batch_tokens = backend_cls.tensor(padded, dtype=long_dtype, device=device)
            attention_mask = backend_cls.tensor(mask, dtype=long_dtype, device=device)
            outputs = self.model_wrapper.forward(
                input_ids=batch_tokens.data, attention_mask=attention_mask.data
            )
            logits = batch_tokens.ensure_tensor(outputs["logits"])
            log_probs = logits.log_softmax(dim=-1)

            # Forward rows: one row per node, but each row's "next token"
            # lives at a different position (row_lens[i] - 1) -- gather all
            # of them in a single fancy-indexed pass instead of one slice
            # per row.
            fwd_idx = [i for i in range(n_chunk) if chunk_kind[i] == "forward"]
            if fwd_idx:
                row_index = backend_cls.tensor(fwd_idx, dtype=long_dtype, device=device)
                last_pos = backend_cls.tensor(
                    [row_lens[i] - 1 for i in fwd_idx], dtype=long_dtype, device=device
                )
                gathered = log_probs[row_index, last_pos, :]  # [len(fwd_idx), vocab]
                for j, i in enumerate(fwd_idx):
                    forward_logits[chunk_node[i]] = gathered[j]

            # Backward rows: every candidate belonging to the same node
            # shares the exact same suffix (same positions, same targets)
            # -- only the prepended candidate differs -- so a contiguous
            # run of one node's rows scores as a single [run_len, L]
            # gather+sum instead of run_len separate ones. Rows are built
            # node-by-node above, so runs are already contiguous; this
            # walk just finds their boundaries within the chunk.
            i = 0
            while i < n_chunk:
                if chunk_kind[i] != "backward":
                    i += 1
                    continue
                node_id = chunk_node[i]
                j = i + 1
                while j < n_chunk and chunk_kind[j] == "backward" and chunk_node[j] == node_id:
                    j += 1
                suffix = chunk_suffix[i]
                L = len(suffix)
                positions = AbstractTensor.arange(
                    offset, offset + L, device=device, dtype=long_dtype, cls=backend_cls
                )
                targets = backend_cls.tensor(suffix, dtype=long_dtype, device=device)
                block = log_probs[i:j, positions, targets]  # [j - i, L]
                totals = block.sum(dim=1).tolist()
                bucket = backward_scores.setdefault(node_id, [])
                for k, row_i in enumerate(range(i, j)):
                    bucket.append((chunk_candidate[row_i], totals[k] / max(L, 1)))
                i = j

            # See implicit_backpath.py's _score_batch for why this drops
            # references but deliberately doesn't call
            # torch.cuda.empty_cache() -- an A/B test against the
            # unmodified original showed that call makes no measurable
            # difference to a real OOM this workload can hit at large
            # candidate-pool/context sizes.
            del logits, log_probs, outputs, batch_tokens, attention_mask

        poetic = self.config.poetic_attractor
        word_trie = self.config.word_trie
        # A wider shortlist is only worth the extra (comparatively cheap:
        # decode text + heuristics, or a bounded subword beam search) work
        # when something downstream actually wants more than branch_factor
        # round-0 candidates to choose from.
        needs_wide_shortlist = poetic is not None or word_trie is not None
        for node in nodes:
            if node.direction is Direction.FORWARD:
                last_logits = forward_logits[node.id].unsqueeze(0)
                shortlist_k = self.config.branch_factor
                if needs_wide_shortlist:
                    shortlist_k = min(
                        max(shortlist_k, self.config.poetic_shortlist_k),
                        forward_logits[node.id].shape[0],
                    )
                scores, indices = self.choice_policy.choose(last_logits, k=shortlist_k)
                round0 = list(zip(indices.tolist()[0], scores.tolist()[0]))
                if word_trie is not None:
                    spans = self._grow_forward_word(node, round0)
                else:
                    spans = [([tid], sc) for tid, sc in round0]
                spans = self._select_final_candidates(
                    node, "forward", spans, self._effective_branch_factor(node)
                )
                if spans:
                    token_spans, scs = zip(*spans)
                    self._attach_children(node.id, Direction.FORWARD, list(scs), list(token_spans))
            else:
                scored = sorted(
                    backward_scores.get(node.id, []), key=lambda pair: pair[1], reverse=True
                )
                shortlist_k = self.config.branch_factor
                if needs_wide_shortlist:
                    shortlist_k = max(shortlist_k, self.config.poetic_shortlist_k)
                round0 = scored[:shortlist_k]
                if word_trie is not None:
                    spans = self._grow_backward_word(node, round0)
                else:
                    spans = [([tid], sc) for tid, sc in round0]
                spans = self._select_final_candidates(
                    node, "backward", spans, self._effective_branch_factor(node)
                )
                if spans:
                    token_spans, scs = zip(*spans)
                    self._attach_children(node.id, Direction.BACKWARD, list(scs), list(token_spans))

    def _grow_forward_word(
        self, node: FluxNode, round0: List[Tuple[int, float]]
    ) -> List[Tuple[List[int], float]]:
        """Grow each round-0 candidate (one subtoken) into a complete word.

        Boundary detection is unchanged from the original design: each
        step, the model's own top branch_factor next-token picks (over the
        *full* vocab -- this step's forward pass already produces that
        distribution for free) are checked with starts_new_word; the
        moment one carries GPT-2's fresh-word marker, the beam finalizes
        with what it already has (that candidate belongs to the *next*
        word, not this one) -- multiple tied boundary candidates in the
        same round still only finalize once.

        What changed is how *continuation* candidates are chosen. The
        original design took the model's top branch_factor picks and
        discarded whichever didn't extend a valid trie prefix -- which
        meant a beam could starve outright if none of the model's top
        picks happened to fit the dictionary, with no way to recover a
        lower-ranked-but-trie-valid candidate the model just didn't rank
        in the top few. Now, ``word_trie``'s own children at the beam's
        current trie state are looked up first (TrieGate, cached by node
        -- no extra model call, since the full-vocab distribution is
        already in hand), and choice_policy only ranks *among* those
        already-guaranteed-valid candidates, gathered from the same
        logits. A completed word can now only ever be a real trie member
        (see TrieGate/WordTrie), and the candidate count considered for
        growth shrinks as the word gets longer instead of staying flat at
        the full vocabulary. Scores for growth candidates are therefore
        relative to the trie-narrowed set, not the full vocabulary --
        renormalized log-probability over the candidates actually being
        chosen among, which is the semantically correct quantity once
        growth is constrained rather than merely filtered after the fact.

        ``word_trie=None`` (round0 seeded via _expand_batch never even
        calls this function in that case, but direct callers/tests can)
        falls back to the original top-k-then-unfiltered behavior exactly,
        so this stays a true no-op when word growth's trie isn't set.
        """
        trie = self.config.word_trie
        tokenizer = self.backward_scorer.tokenizer
        branch_factor = self.config.branch_factor
        prefix_tokens, _ = self.path_tokens(node.id)
        base_context = self.anchor_tokens + prefix_tokens

        trie_gate = self._get_word_trie_gate(trie) if trie is not None else None

        beams = []
        for token_id, score in round0:
            text = tokenizer.decode([token_id]).strip()
            node_state = trie.walk_from(trie.root_node(), text) if trie is not None else None
            if trie is not None and node_state is None:
                # This round-0 seed's own text isn't even a valid trie
                # prefix -- it can never grow into a real dictionary word,
                # so don't waste a beam slot on it.
                continue
            beams.append({
                "tokens": [token_id],
                "sum": float(score),
                "text": text,
                "node": node_state,
            })

        finalized: List[Tuple[List[int], float]] = []
        ops = self.tensor_ops
        backend_cls = type(ops)
        long_dtype = ops.long_dtype

        for _step in range(1, self.config.max_subword_steps):
            if not beams:
                break
            rows = [self._context_window(base_context + b["tokens"]) for b in beams]
            row_lens = [len(r) for r in rows]
            pad_len = max(row_lens)
            padded = [r + [0] * (pad_len - len(r)) for r in rows]
            mask = [[1] * l + [0] * (pad_len - l) for l in row_lens]
            batch_tokens = backend_cls.tensor(padded, dtype=long_dtype, device=self.device)
            attention_mask = backend_cls.tensor(mask, dtype=long_dtype, device=self.device)
            outputs = self.model_wrapper.forward(
                input_ids=batch_tokens.data, attention_mask=attention_mask.data
            )
            logits = batch_tokens.ensure_tensor(outputs["logits"])
            log_probs = logits.log_softmax(dim=-1)

            new_beams = []
            for bi, beam in enumerate(beams):
                row_logits = log_probs[bi, row_lens[bi] - 1, :].unsqueeze(0)

                # Boundary detection: same mechanism as before this change,
                # the model's own top-k over the full vocab.
                scores, indices = self.choice_policy.choose(row_logits, k=branch_factor)
                already_finalized_this_beam = False
                fallback_growth: List[Tuple[int, float, str]] = []
                for cand_id, cand_score in zip(indices.tolist()[0], scores.tolist()[0]):
                    cand_text = tokenizer.decode([cand_id])
                    if starts_new_word(cand_text):
                        if not already_finalized_this_beam:
                            finalized.append((list(beam["tokens"]), beam["sum"] / len(beam["tokens"])))
                            already_finalized_this_beam = True
                        continue
                    if trie_gate is None:
                        # No trie configured -- fall back to the original
                        # unfiltered behavior exactly (matches direct
                        # word_trie=None callers/tests).
                        fallback_growth.append((cand_id, cand_score, cand_text.strip()))

                produced_growth = False
                if trie_gate is not None:
                    if beam["node"] is not None:
                        valid = trie_gate.continuations(beam["node"])
                        if valid:
                            valid_ids = [tid for tid, _ in valid]
                            id_tensor = backend_cls.tensor(valid_ids, dtype=long_dtype, device=self.device)
                            narrowed_logits = row_logits[:, id_tensor]
                            k = min(branch_factor, len(valid_ids))
                            g_scores, g_indices = self.choice_policy.choose(narrowed_logits, k=k)
                            for local_idx, g_score in zip(g_indices.tolist()[0], g_scores.tolist()[0]):
                                cand_id, next_node = valid[local_idx]
                                new_beams.append({
                                    "tokens": beam["tokens"] + [cand_id],
                                    "sum": beam["sum"] + float(g_score),
                                    "text": beam["text"] + tokenizer.decode([cand_id]).strip(),
                                    "node": next_node,
                                })
                                produced_growth = True
                else:
                    for cand_id, cand_score, stripped_text in fallback_growth:
                        new_beams.append({
                            "tokens": beam["tokens"] + [cand_id],
                            "sum": beam["sum"] + float(cand_score),
                            "text": beam["text"] + stripped_text,
                            "node": None,
                        })
                        produced_growth = True

                if not produced_growth and not already_finalized_this_beam:
                    # Trie dead-end (or, with no trie, no candidate at all)
                    # and no boundary signal either -- keep the word as-is
                    # rather than losing it outright.
                    finalized.append((list(beam["tokens"]), beam["sum"] / len(beam["tokens"])))

            # See implicit_backpath.py's _score_batch for why this
            # deliberately skips torch.cuda.empty_cache().
            del logits, log_probs, outputs, batch_tokens, attention_mask

            new_beams.sort(key=lambda b: b["sum"] / len(b["tokens"]), reverse=True)
            beams = new_beams[:branch_factor]

        for beam in beams:
            finalized.append((list(beam["tokens"]), beam["sum"] / len(beam["tokens"])))

        finalized.sort(key=lambda pair: pair[1], reverse=True)
        return finalized

    def _grow_backward_word(
        self, node: FluxNode, round0: List[Tuple[int, float]]
    ) -> List[Tuple[List[int], float]]:
        """Backward mirror of _grow_forward_word: candidates get *prepended*.

        Because growth always prepends (each new token goes to the front of
        the span, which is already accumulating in correct left-to-right
        reading order), the completion signal is simpler than forward's
        look-ahead rule: a beam finalizes the instant the token it just
        prepended itself starts a new word -- that token *is* the word's
        first token, nothing earlier belongs to this word.

        Uses ``FluxGraphConfig.backward_word_trie`` -- a *reversed* WordTrie
        (see WordTrie's ``reverse`` parameter), since backward growth
        discovers a word from its end toward its start; validating against
        the forward word_trie would ask a prefix question about what's
        actually a suffix-in-progress. Growth candidates are that trie's
        children at the beam's current state (TrieGate, cached by node),
        scored via a single ``score_candidates`` call over just that
        narrowed set -- a real, usually dramatic reduction versus scoring
        the *entire* dictionary-filtered pool (tens of thousands of ids)
        on every single step, and it structurally guarantees a completed
        word is a real trie member, closing a real, previously-documented
        gap where backward-grown words were never validated against the
        dictionary at all (repeatedly prepending a token whose *own* text
        happens to be a short dictionary word, e.g. "ab", produced
        garbage like "AbAbAbAballah" -- each token passed a flat,
        no-memory per-token filter individually, nothing ever checked
        whether the *accumulating* span stayed on a path toward a real
        word). starts_new_word is still checked on whatever gets scored,
        as a defensive backstop -- in practice a reversed-trie child is
        always a plain lowercase letter continuation and essentially never
        trips it; a trie dead end (no valid continuations left) is the
        primary, expected way growth stops now.

        ``backward_word_trie=None`` falls back to the original
        score-the-whole-pool-then-topk behavior exactly, so this stays a
        true no-op when it isn't set (e.g. direct callers/tests that only
        configure the forward word_trie).
        """
        tokenizer = self.backward_scorer.tokenizer
        trie = self.config.backward_word_trie
        branch_factor = self.config.branch_factor
        ops = self.tensor_ops
        backend_cls = type(ops)

        trie_gate = self._get_word_trie_gate(trie) if trie is not None else None

        beams = []
        finalized: List[Tuple[List[int], float]] = []
        for token_id, score in round0:
            text = tokenizer.decode([token_id])
            if starts_new_word(text):
                finalized.append(([token_id], float(score)))
                continue
            stripped = text.strip()
            node_state = trie.walk_from(trie.root_node(), stripped[::-1]) if trie is not None else None
            if trie is not None and node_state is None:
                # This round-0 seed's own text isn't even a valid
                # (reversed) trie prefix -- it can never grow into a real
                # dictionary word.
                continue
            beams.append({"tokens": [token_id], "mean": float(score), "node": node_state})

        for _step in range(1, self.config.max_subword_steps):
            if not beams:
                break
            new_beams = []
            for beam in beams:
                if trie_gate is not None:
                    if beam["node"] is None:
                        continue
                    valid = trie_gate.continuations(beam["node"])
                    if not valid:
                        # Trie dead end: no candidate can extend this span
                        # any further toward a real word -- keep it as-is.
                        finalized.append((beam["tokens"], beam["mean"]))
                        continue
                    candidate_ids = [tid for tid, _ in valid]
                    node_by_id = dict(valid)
                else:
                    pool = self.backward_scorer.candidate_pool(
                        ops, vocab_size=self.backward_scorer.tokenizer.vocab_size, device=self.device
                    )
                    candidate_ids = pool.tolist()
                    node_by_id = {}

                suffix_t = backend_cls.tensor(beam["tokens"], dtype=ops.long_dtype, device=self.device)
                cand_t = backend_cls.tensor(candidate_ids, dtype=ops.long_dtype, device=self.device)
                # See FluxGraphConfig.max_expand_elements: the same element
                # budget that bounds _expand_batch's round-0 chunking also
                # bounds this call -- matters mainly for the no-trie
                # fallback path above, where candidate_ids can still be the
                # full multi-ten-thousand-token pool. Leaving max_batch_size
                # at score_candidates's own default (2048) when no budget is
                # configured -- passing None explicitly here would instead
                # *disable* chunking entirely, the opposite of preserving
                # original behavior.
                score_kwargs = {}
                if self.config.max_expand_elements is not None:
                    row_len = 1 + len(beam["tokens"])
                    vocab_size = self.backward_scorer.tokenizer.vocab_size
                    score_kwargs["max_batch_size"] = max(
                        1, self.config.max_expand_elements // max(row_len * vocab_size, 1)
                    )
                raw = self.backward_scorer.score_candidates(suffix_t, cand_t, **score_kwargs)
                raw = raw / max(len(beam["tokens"]), 1)
                k = min(branch_factor, raw.shape[0])
                top_scores, top_idx = AbstractTensor.topk(raw, k=k, dim=0)
                for cand_mean, idx in zip(top_scores.tolist(), top_idx.tolist()):
                    cand_id = candidate_ids[idx]
                    text = tokenizer.decode([cand_id])
                    new_tokens = [cand_id] + beam["tokens"]
                    if starts_new_word(text):
                        finalized.append((new_tokens, cand_mean))
                    else:
                        new_beams.append({
                            "tokens": new_tokens,
                            "mean": cand_mean,
                            "node": node_by_id.get(cand_id),
                        })
            new_beams.sort(key=lambda b: b["mean"], reverse=True)
            beams = new_beams[:branch_factor]

        for beam in beams:
            finalized.append((beam["tokens"], beam["mean"]))

        finalized.sort(key=lambda pair: pair[1], reverse=True)
        return finalized

    def _context_words_for_node(self, node: FluxNode, side: str) -> List[str]:
        """Decoded, whitespace-split words of ``node``'s currently-known context.

        The last entry is the current line-end word on that side; the rest
        are available for internal-rhyme sweeps.
        """
        prefix_tokens, _ = self.path_tokens(node.id)
        if side == "forward":
            tokens = self.anchor_tokens + prefix_tokens
        else:
            fwd_leaf = self.best_leaf(Direction.FORWARD)
            fwd_tokens = self.path_tokens(fwd_leaf)[0] if fwd_leaf is not None else []
            tokens = prefix_tokens + self.anchor_tokens + fwd_tokens
        tokens = self._context_window(tokens)
        text = self.backward_scorer.tokenizer.decode(tokens)
        return [w for w in text.split() if w]

    def _select_final_candidates(
        self, node: FluxNode, side: str, candidates: List[Tuple[List[int], float]], keep: int
    ) -> List[Tuple[List[int], float]]:
        """Pick the final ``keep`` candidates, honest scores intact.

        ``candidates`` is a list of (token_span, true_score) pairs, already
        sorted best-first by true model evidence -- a span is more than one
        token only when word growth (FluxGraphConfig.word_trie) produced it;
        otherwise every span is length 1, same as before word growth existed.
        ``keep`` is the caller's final child count for this node -- ordinarily
        branch_factor, but auxin suppression (_effective_branch_factor) can
        shrink it per node; the internal reranking shortlist width below is
        intentionally independent of that; a suppressed node still reranks
        over a full-width shortlist, it just keeps fewer winners.

        With no poetic attractor configured, this is just a truncation to
        ``keep`` -- identical to plain top-k behavior. With one configured,
        a bounded shortlist (never the full candidate pool) is re-ranked by
        ``true_score + poetic_scale * bonus`` and truncated -- but the
        *returned* score for every surviving candidate is always its real,
        unmodified evidence. The poetic bonus decides who gets picked, never
        what a picked node's local_evidence/pressure claims to be -- the
        same honesty local_evidence itself was just fixed to have for the
        forward/backward asymmetry.

        Vocabulary filtering (which words are even eligible at all) is a
        separate, earlier concern -- see WordTrie/DictionaryTokenFilter,
        built once from a real dictionary, not a per-candidate rerank here.
        """
        poetic = self.config.poetic_attractor
        if poetic is None or not candidates:
            return candidates[:keep]

        shortlist = candidates[: max(self.config.branch_factor, self.config.poetic_shortlist_k)]
        context_words = self._context_words_for_node(node, side)
        tokenizer = self.backward_scorer.tokenizer
        reranked = []
        for tokens, true_score in shortlist:
            word = tokenizer.decode(tokens).strip()
            bonus = self.config.poetic_scale * poetic.score_word(word, context_words)
            reranked.append((tokens, true_score, true_score + bonus))
        reranked.sort(key=lambda triple: triple[2], reverse=True)
        return [(tks, ts) for tks, ts, _ in reranked[:keep]]

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
        self._publish_snapshot()

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

        blocked = self._repeated_ngram_tokens(full, "append")
        if blocked:
            last_logits[0, list(blocked)] = float("-inf")

        scores, indices = self.choice_policy.choose(last_logits, k=self.config.branch_factor)
        spans = [[i] for i in indices.tolist()[0]]
        self._attach_children(node_id, Direction.FORWARD, scores.tolist()[0], spans)

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
        blocked = self._repeated_ngram_tokens(suffix, "prepend")
        if blocked:
            pool_list = pool.tolist()
            keep_idx = [i for i, tid in enumerate(pool_list) if tid not in blocked]
            if keep_idx:
                pool = pool[keep_idx]
        if self.config.verbose:
            print(f"  [expand-backward] node {node_id}: scoring {pool.shape[0]} candidates ...")
        raw_scores = self.backward_scorer.score_candidates(
            suffix_t, pool, left_context=self.config.backward_left_context
        )
        # score_candidates returns the *total* log-likelihood of the whole
        # suffix (summed over all len(suffix) positions), not one token's
        # log-prob -- a fundamentally different quantity from forward's
        # per-token local_evidence, and one whose magnitude grows with the
        # suffix (i.e. with how much the graph has already grown) regardless
        # of candidate quality. Dividing by the suffix length here converts
        # it to a per-token mean before it enters local_evidence/pressure,
        # so backward nodes are judged on the same scale as forward ones
        # instead of getting mechanically starved as the graph grows.
        raw_scores = raw_scores / max(len(suffix), 1)
        top_scores, top_idx = AbstractTensor.topk(raw_scores, k=self.config.branch_factor, dim=0)
        candidate_spans = [[int(pool[i].item())] for i in top_idx.tolist()]
        self._attach_children(node_id, Direction.BACKWARD, top_scores.tolist(), candidate_spans)

    def _attach_children(
        self, parent_id: int, direction: Direction, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        parent = self.nodes[parent_id]
        for score, tokens in zip(scores, token_spans):
            child_id = self._alloc_id()
            cumulative = parent.cumulative_evidence + float(score)
            new_depth = parent.depth + 1
            self.nodes[child_id] = FluxNode(
                id=child_id,
                tokens=[int(t) for t in tokens],
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

