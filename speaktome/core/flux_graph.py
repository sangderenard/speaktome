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
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensors import AbstractTensor
from .choice_policy import ChoicePolicy
from .implicit_backpath import ImplicitBackpathScorer
from .model_abstraction import AbstractModelWrapper
from .noodle_explorer import Direction
from .poetic_attractor import PoeticAttractor
from .word_trie import WordTrie, TrieGate
from .word_boundary import starts_new_word
# --- END HEADER ---


@dataclass(frozen=True)
class SolubleType:
    """One named kind of thing an Edge's channels can carry.

    value_kind distinguishes genuinely different arithmetic: "float" for
    continuous quantities like pressure (today's only real substance --
    see FluxNode.pressure), "int" for things that only make sense in
    whole units (a future soluble might count discrete events, not a
    continuous flow). Nothing currently reads value_kind to do anything
    -- see Edge/Channel's own docstrings for why: this whole registry is
    Phase 1, pure structure. It exists so a caller building richer
    edge-level metrics later has an actual place to declare a new
    soluble without inventing one.
    """
    name: str
    value_kind: str  # "float" or "int"


DEFAULT_SOLUBLE_TYPES: Dict[str, SolubleType] = {
    "pressure": SolubleType("pressure", "float"),
}


def uniform_field(_radius: float) -> float:
    """Default scalar gradient for a pie slice's region: the substance is
    uniformly present at strength 1.0 at every radius. See
    FluxGraphConfig.scalar_fields."""
    return 1.0


def _is_oom_error(exc: BaseException) -> bool:
    """True for a real GPU out-of-memory error, checked by message text
    rather than importing torch here -- flux_graph.py stays backend-
    agnostic (see AbstractTensor/tensor_ops) everywhere else. Covers both
    a plain RuntimeError ("CUDA out of memory...", older torch / some
    cuDNN paths) and torch.cuda.OutOfMemoryError, itself a RuntimeError
    subclass with the same wording."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass
class Channel:
    """One direction of one Edge -- see Edge's own docstring for the pair.

    conductance mirrors exactly what FluxGraph._edge_conductance/
    _directional_conductance already compute -- this just gives that
    number a home on a real, addressable object instead of only ever
    existing as a function's return value. style/whitelist/blacklist are
    declared now, in the shape a later gating pass can read, but nothing
    in _update_pressures consults them yet -- see Edge's docstring for
    why that's deliberate.
    """
    conductance: float = 0.0
    style: str = "bidirectional"  # "bidirectional" | "receive_only" | "give_only"
    whitelist: Optional["set[str]"] = None  # soluble type names allowed through; None = all allowed
    blacklist: "set[str]" = field(default_factory=set)  # soluble type names explicitly blocked


@dataclass
class SubEdge:
    """One direction of a Traversal's own connection through a real Edge.

    Every Traversal creates exactly two of these when it's recorded --
    one going one way, one going the other -- and each is genuinely
    one-way: "forward" only ever moves volume start_id -> end_id,
    "reverse" only ever moves it end_id -> start_id, independently
    constricted. The same two SubEdge objects are held by every real
    Edge that traversal's path crosses (see Edge.subedges): Edge is a
    hull, SubEdges are what it actually holds.

    constriction is this subedge's own valve declaration -- 1.0 is fully
    open (an open end is just a valve at its default/maximal setting,
    nothing special-cased), 0.0 blocks all transport through it. Gates
    both transport mechanisms in FluxGraph._transport_subedges uniformly:
    bulk flow from the pressure differential, and soluble flow from the
    volume/concentration differential (osmotic-style pull toward
    equalizing) -- a valve that resists or reverses that second pull
    (what "anti-osmotic" describes) isn't separately implemented yet;
    only the single constriction scalar is.

    A SubEdge is not, by default, open to a node's volume at all -- see
    is_open_at: the one default rule that *is* specified is that it's
    open at its own two endpoints (the owning Traversal's start_id and
    end_id) and closed everywhere else it merely passes through.
    """
    traversal_key: Tuple[int, int]  # (start_id, end_id) of the owning Traversal
    direction: str  # "forward" | "reverse"
    constriction: float = 1.0

    def is_open_at(self, node_id: int) -> bool:
        """Whether this subedge currently allows volume transfer at node_id.

        Closed everywhere except its own two endpoints.
        """
        return node_id in self.traversal_key


@dataclass
class Edge:
    """One connection between two nodes, as a real object -- not just a
    parent_id/children_ids pointer pair with conductance recomputed from
    scratch on demand.

    forward/reverse are two independent Channels (see Channel) -- this is
    the "delivery and return can carry different things" idea, given
    actual structure instead of only the single return_conductance_scale
    dial that already exists on the pressure equation itself.

    subedges is this Edge's role as a hull: every Traversal whose path
    crosses this Edge contributes its own two SubEdges (see SubEdge) here.

    formation/seed_id_at_formation are a *permanent* record of how this
    edge was actually grown, captured once at creation and never touched
    again -- deliberately independent of FluxNode.direction/depth/height,
    which are anchor-relative and do change when re-rooting moves the
    anchor (see FluxGraph._reroot). Without a stable record like this,
    there is no way to later ask "was this genuinely grown as a prefix or
    a postfix" once enough re-roots have happened that the live direction
    fields no longer agree with formation history. "prefix_beam" (backward
    growth -- each new token is prepended, building toward the start of
    the eventual text) and "postfix_beam" (forward growth -- appended,
    building toward the end) are the only two cases, matching Direction's
    own two values -- an Edge only ever exists for a real, directly-grown
    parent/child connection; see Traversal for the (much larger) space of
    causal paths that don't necessarily correspond to one direct edge.

    Phase 1 only: this object is populated by _attach_children the
    instant a node is created, and is otherwise inert -- nothing in
    _update_pressures reads it, and creating it changes no existing
    behavior. It exists to be queried/mapped by arbitrary later code
    (metrics, tooling, a future gating or multi-soluble pass), not to
    replace the pressure math that already runs today.
    """
    from_id: int
    to_id: int
    forward: Channel
    reverse: Channel
    formation: str  # "prefix_beam" | "postfix_beam"
    seed_id_at_formation: int
    created_tick: int
    subedges: List[SubEdge] = field(default_factory=list)


@dataclass
class Traversal:
    """Any causal path in the network: start_id and end_id are any two
    nodes where end_id is a genuine descendant of start_id along the
    tree's real parent chain -- NOT necessarily the seed, NOT necessarily
    a leaf. A short internal sub-phrase, anywhere in the graph, that has
    its own real path and its own real score, independent of whatever the
    seed happens to be doing right now.

    node_ids is start-to-end, tree order, along the real chain between
    them. mean_score is (end.cumulative_evidence - start.cumulative_
    evidence) / hop_count -- genuinely derivable from the two endpoints'
    already-known evidence sums, not something that needs a fresh model
    call: cumulative_evidence already sums each token's own independently
    -scored local_evidence along its real path, so the difference between
    any two points on the same lineage is exactly that sub-span's own
    total evidence.

    Recorded by FluxGraph._run_graph_auditor, which enumerates every
    (ancestor, descendant) pair among currently-live nodes -- keyed by
    (start_id, end_id) in FluxGraph.traversals, so "has this specific
    causal path already been run" is a direct lookup, not a search.

    subedges holds this traversal's own two SubEdges (see SubEdge) --
    the canonical, unique-per-traversal home for them. The same two
    objects are also referenced from every real Edge's own subedges list
    along node_ids (Edge as hull, for querying "what crosses this real
    connection"), but FluxGraph._transport_subedges iterates traversals,
    not edges, so each pair is processed exactly once per tick.
    """
    start_id: int
    end_id: int
    node_ids: List[int]
    mean_score: float
    closed_tick: int
    subedges: List[SubEdge] = field(default_factory=list)


@dataclass
class MetaEdge:
    """A hyperedge bundling every individual Edge between two regions of
    the graph into one aggregate connection with its own two-way pressure
    conduit -- sitting above Edge the way Edge sits above the old
    function-only conductance. "Region" is FluxGraph._edge_region: "main"
    for the anchor's own bowtie, or an orthogonal network's own root id
    (see orthogonal_network_roots) for a cousin network -- the same
    grouping the radar already uses for its own pie wedges, reused here
    instead of inventing a second regioning concept.

    member_edge_keys are the individual Edge keys this bundles; forward/
    reverse are the aggregate Channels (mean conductance of the bundled
    edges' own forward/reverse, in each direction) so the meta-connection
    between two regions can be read as one number without losing the
    individual edges it's built from -- those stay exactly where they
    are, in FluxGraph.edges, untouched.
    """
    id: int
    from_region: str
    to_region: str
    member_edge_keys: "set[Tuple[int, int]]"
    forward: Channel
    reverse: Channel
    updated_tick: int


@dataclass
class HeartPhase:
    """One step of the heart's beat script.

    valves is the nexus state for this phase: the heart's chambers are
    the split-and-twisted orange -- every slice contributes an "in" half
    and an "out" half, and the twist puts every in-chamber in potential
    contact with every out-chamber at one central nexus. A valve entry
    (in_slice, out_slice) -> throttle 0..1 opens that contact; anything
    absent is sealed. Two string presets resolve against whichever
    slices are actually live at beat time (slices come and go with
    re-roots, so static dicts can't cover them): "crossover" opens each
    in-chamber only to its own group's opposite-direction out-chamber
    (the four-chamber crossover pump), "all" opens every pair (total
    mixing).

    contractions: which chambers squeeze this phase and how hard (0..1
    fraction of contents expelled). An in-chamber squeezes through its
    open nexus valves into out-chambers, split by throttle share; an
    out-chamber squeezes out to the network's outflow subedges. A
    chamber squeezing with every valve sealed simply resists -- nothing
    leaves. Presets: "all" (every chamber, full), "in" / "out" (one
    stage only, full), or an explicit chamber-key -> fraction dict.

    Solutes additionally move by osmotic rebalancing speed, not squeeze
    pressure: across every open valve, each soluble flows down its own
    concentration differential at a rate set by the valve throttle --
    the mixing throttle -- independent of any contraction. Solvent only
    moves by squeeze.

    exit_scope: where an out-chamber's squeeze delivers -- "own" (its
    own slice's outflow subedges only) or "all" (every outflow subedge
    regardless of slice, the total-supply exit).
    """
    name: str
    valves: Any = "crossover"  # "crossover" | "all" | Dict[(in_slice, out_slice), float]
    contractions: Any = "all"  # "all" | "in" | "out" | Dict[chamber_key, float]
    exit_scope: str = "own"  # "own" | "all"


@dataclass
class HeartHook:
    """A system attached to the heart that works on chamber fluid.

    when: "pre" runs after intake lands in the in-chambers but before
    any beat physics; "post" runs after the beat, before residuals sit
    for the next tick. scope: a chamber key ("main:forward|in") to
    receive just that chamber's mixture dict, or "total" -- the
    privileged option -- to receive the whole chambers dict at once
    (total-supply access: a cardiopulmonary link, forced whole-supply
    filtering, or whatever else isn't defined yet). fn mutates what it's
    handed in place.
    """
    name: str
    when: str  # "pre" | "post"
    scope: str  # chamber key | "total"
    fn: Callable[[Dict], None]


def _flip_slice(slice_name: str) -> str:
    """"main:forward" -> "main:backward" and vice versa -- same group,
    opposite direction."""
    if slice_name.endswith(":forward"):
        return slice_name[: -len("forward")] + "backward"
    return slice_name[: -len("backward")] + "forward"


HEART_SCRIPTS: Dict[str, List[HeartPhase]] = {
    # The four-chamber crossover pump: one beat, every chamber
    # contracts fully, each in-chamber's only open valve is its own
    # group's opposite-direction out-chamber.
    "crossover": [HeartPhase(name="beat", valves="crossover", contractions="all")],
    # Everything open, everything squeezing: one shared intake pool,
    # one shared output.
    "total_mix": [HeartPhase(name="mix", valves="all", contractions="all", exit_scope="all")],
}


class Heart:
    """The anchor's pump, made explicit: per-slice chambers, a nexus
    valve matrix, a scripted beat, and an attachment API.

    Chambers are keyed "<slice>|in" / "<slice>|out" and hold real
    mixture dicts (same "solvent"-plus-soluble-names form the rest of
    transport uses) between ticks -- the heart is the one place in the
    system allowed to hold volume that belongs to no node; the anchor
    node itself still never holds anything. Chambers appear lazily as
    slices appear and simply sit empty when their slice dies.

    The script is a list of HeartPhases advanced one per tick (beat),
    wrapping -- rhythm is data, not code. set_script installs a named
    preset from HEART_SCRIPTS or a custom phase list. attach/detach
    manage HeartHooks (see that docstring for pre/post and the
    privileged "total" scope).
    """

    def __init__(self) -> None:
        self.chambers: Dict[str, Dict[str, float]] = {}
        self.script_name = "crossover"
        self.script: List[HeartPhase] = list(HEART_SCRIPTS["crossover"])
        self.phase_index = 0
        self.hooks: List[HeartHook] = []

    # -- configuration API ------------------------------------------------
    def set_script(self, script: Any) -> None:
        """Install a named preset (see HEART_SCRIPTS) or a custom list of
        HeartPhases. Resets the beat to the top of the new script."""
        if isinstance(script, str):
            if script not in HEART_SCRIPTS:
                raise ValueError(f"unknown heart script {script!r}; presets: {sorted(HEART_SCRIPTS)}")
            self.script_name = script
            self.script = list(HEART_SCRIPTS[script])
        else:
            self.script_name = "custom"
            self.script = list(script)
        self.phase_index = 0

    def attach(self, name: str, when: str, scope: str, fn: Callable[[Dict], None]) -> None:
        """Attach a system that works on chamber fluid -- see HeartHook."""
        self.detach(name)
        self.hooks.append(HeartHook(name=name, when=when, scope=scope, fn=fn))

    def detach(self, name: str) -> None:
        self.hooks = [h for h in self.hooks if h.name != name]

    def state(self) -> Dict[str, Dict[str, float]]:
        """Copy of every chamber's current contents, for inspection."""
        return {key: dict(mix) for key, mix in self.chambers.items()}

    # -- internals --------------------------------------------------------
    def chamber(self, slice_name: str, stage: str) -> Dict[str, float]:
        return self.chambers.setdefault(f"{slice_name}|{stage}", {})

    @staticmethod
    def _volume(mix: Dict[str, float]) -> float:
        return sum(mix.values())

    def _run_hooks(self, when: str) -> None:
        for hook in self.hooks:
            if hook.when != when:
                continue
            if hook.scope == "total":
                hook.fn(self.chambers)
            elif hook.scope in self.chambers:
                hook.fn(self.chambers[hook.scope])

    def _resolve_valves(self, phase: HeartPhase) -> Dict[Tuple[str, str], float]:
        in_slices = [k[: -len("|in")] for k in self.chambers if k.endswith("|in")]
        out_slices = [k[: -len("|out")] for k in self.chambers if k.endswith("|out")]
        if phase.valves == "crossover":
            return {(s, _flip_slice(s)): 1.0 for s in in_slices if _flip_slice(s) in out_slices}
        if phase.valves == "all":
            return {(i, o): 1.0 for i in in_slices for o in out_slices}
        return dict(phase.valves)

    def _resolve_contractions(self, phase: HeartPhase) -> Dict[str, float]:
        if phase.contractions == "all":
            return {key: 1.0 for key in self.chambers}
        if phase.contractions == "in":
            return {key: 1.0 for key in self.chambers if key.endswith("|in")}
        if phase.contractions == "out":
            return {key: 1.0 for key in self.chambers if key.endswith("|out")}
        return dict(phase.contractions)

    def _osmotic_rebalance(self, valves: Dict[Tuple[str, str], float]) -> None:
        """Solutes cross every open valve at rebalancing speed -- down
        their own concentration differential, scaled by the valve
        throttle and bounded by the smaller chamber's volume -- with no
        contraction needed. Solvent doesn't move here; bulk is squeeze's
        job."""
        for (in_slice, out_slice), throttle in valves.items():
            if throttle <= 0.0:
                continue
            a = self.chamber(in_slice, "in")
            b = self.chamber(out_slice, "out")
            vol_a, vol_b = self._volume(a), self._volume(b)
            if vol_a <= 0.0 and vol_b <= 0.0:
                continue
            scale = min(vol_a, vol_b) if vol_a > 0.0 and vol_b > 0.0 else max(vol_a, vol_b)
            for name in set(a) | set(b):
                if name == "solvent":
                    continue
                conc_a = a.get(name, 0.0) / vol_a if vol_a > 0.0 else 0.0
                conc_b = b.get(name, 0.0) / vol_b if vol_b > 0.0 else 0.0
                delta = throttle * (conc_a - conc_b) * scale
                if delta > 0.0:
                    delta = min(delta, a.get(name, 0.0))
                    a[name] = a.get(name, 0.0) - delta
                    b[name] = b.get(name, 0.0) + delta
                elif delta < 0.0:
                    delta = min(-delta, b.get(name, 0.0))
                    b[name] = b.get(name, 0.0) - delta
                    a[name] = a.get(name, 0.0) + delta

    def _squeeze_in_chambers(self, valves: Dict[Tuple[str, str], float], contractions: Dict[str, float]) -> None:
        """Contracting in-chambers expel through their open nexus valves
        into out-chambers, split by throttle share. Sealed = resists:
        contents stay put."""
        for key, fraction in contractions.items():
            if not key.endswith("|in") or fraction <= 0.0:
                continue
            in_slice = key[: -len("|in")]
            mix = self.chambers.get(key)
            if not mix:
                continue
            open_valves = [(o, t) for (i, o), t in valves.items() if i == in_slice and t > 0.0]
            total_throttle = sum(t for _, t in open_valves)
            if total_throttle <= 0.0:
                continue
            for name, amount in list(mix.items()):
                moved = amount * min(1.0, fraction)
                if moved == 0.0:
                    continue
                mix[name] = amount - moved
                for out_slice, throttle in open_valves:
                    out_mix = self.chamber(out_slice, "out")
                    out_mix[name] = out_mix.get(name, 0.0) + moved * (throttle / total_throttle)


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
    # What SubEdges actually move in and out of -- separate from pressure.
    # Not a single scalar: an inventory of individually-named solubles
    # (e.g. one per pie slice, once ring-dispensing is built), each
    # tracked separately so transport can debit/credit specific
    # substances rather than one undifferentiated amount.
    solubles: Dict[str, float] = field(default_factory=dict)
    # The liquid the solubles are dissolved in. Sourced from the slice's
    # ambient humidity field (see FluxGraph._exchange_humidity): humidity
    # outside the node, solvent once transformed inside it.
    solvent: float = 0.0
    # Openness of this node's humidity exchange -- present by default
    # (1.0), configurable per node, 0.0 shuts it off entirely.
    humidity_exchange: float = 1.0
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
    def volume(self) -> float:
        """Total volume of solution: solvent plus every dissolved soluble
        -- always in sync, since it's derived rather than tracked
        separately."""
        return self.solvent + sum(self.solubles.values())

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
    # The heart-overpressure ceiling for the humidity system (see
    # _exchange_humidity): anchor pressure above this biases every pore
    # to expel-only -- all water must go -- the exact mirror of the
    # starvation_floor intake-only bias. 0 = off, matching every other
    # 0-disables knob here; there's no natural pre-existing "high
    # pressure" quantity to inherit a default from the way the low side
    # inherits starvation_floor.
    overpressure_ceiling: float = 0.0
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
    # How many rounds of real, honest beam search each tick's compute-
    # budget picks run down in one hot loop (see FluxGraph._expand_batch_
    # hot_loop), before control returns to the tick loop and the pressure
    # system gets to reconsider who goes next. Every round is the exact
    # same expansion machinery as always -- branch_factor width, no_repeat_
    # ngram_size, word_trie, etc all still apply identically at every
    # round -- this only controls how many rounds happen back-to-back.
    # 1 (default) is exactly today's one-level-per-tick behavior; higher
    # values let a promising lineage run out several tokens/words deep in
    # one shot, at real, deliberate compute cost (branch_factor**depth
    # for the picked batch), instead of drip-feeding one level per tick.
    hot_loop_depth: int = 1
    # Per-direction overrides of branch_factor/hot_loop_depth/this tick's
    # share of compute_budget_per_tick -- None (default, both directions)
    # falls back to the shared values above exactly, reproducing today's
    # symmetric behavior until you actually diverge them. Forward and
    # backward are genuinely different problems (different scoring
    # regimes, different growth pressure -- see balance_weight/return_
    # conductance_scale, both born from that asymmetry), so letting them
    # run at different widths/depths/rates independently is often more
    # honest than forcing one shared knob to serve both. The budget
    # overrides are hard per-direction caps when set (not a floor with
    # leftover redistribution the way the unset default is -- see
    # FluxGraph._expand_top_pressure_nodes).
    forward_branch_factor: Optional[int] = None
    backward_branch_factor: Optional[int] = None
    forward_hot_loop_depth: Optional[int] = None
    backward_hot_loop_depth: Optional[int] = None
    forward_budget_per_tick: Optional[int] = None
    backward_budget_per_tick: Optional[int] = None
    # Per-direction candidate selection. "topk" (default): exactly
    # branch_factor (or its per-direction override) highest-probability
    # candidates, a fixed count regardless of how the distribution is
    # actually shaped. "topp": nucleus sampling -- keep however many of
    # the model's own top candidates are needed for their cumulative
    # probability to reach top_p, so width tracks the model's real
    # uncertainty (narrow when confident, wide when it's not) instead of
    # an arbitrary fixed count. branch_factor is ignored in "topp" mode --
    # there's no fixed-K concept once selection is coverage-based --
    # except as an upper cap when auxin_suppression is also on, so apical
    # dominance still means something in topp mode too.
    forward_selection_mode: str = "topk"
    backward_selection_mode: str = "topk"
    forward_top_p: float = 0.9
    backward_top_p: float = 0.9
    # How many of the model's own top candidates are even fetched when a
    # direction is in "topp" mode -- a safety ceiling, since a genuinely
    # flat distribution could otherwise need the entire vocabulary before
    # cumulative probability ever reaches top_p.
    top_p_shortlist_ceiling: int = 40
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
    # Asymmetry correction: if one direction's frontier (max abs(height)
    # among its live nodes) has pulled ahead of the other's, the *lagging*
    # side's candidates get a priority boost proportional to the gap --
    # the leading side gets none, since it's already winning the
    # exploit/explore race on its own. Without this, a direction that
    # starts even slightly ahead keeps compounding that lead every tick
    # (more depth -> more candidates -> more chances to look good -> more
    # budget), and the other side stalls out permanently. 0 (default)
    # disables it, matching pre-balance behavior.
    balance_weight: float = 0.0
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
    # Per-tick fractional pressure loss (a node with no support at all
    # still loses decay_rate * its own pressure every tick), on top of
    # whatever the resistor network computes -- 0 (default) disables it,
    # matching pre-decay behavior. Proportional to the node's own
    # pressure, deliberately gentle: at equilibrium this only ever divides
    # a node's settled pressure by (1 + decay_rate), so even a large value
    # here barely touches an already-strong node -- a slow, always-on
    # leak, not a real cap. population_target (below) is the real cap,
    # and is intentionally NOT built the same way.
    decay_rate: float = 0.0
    # Homeostatic control: once the live (non-burned) node count exceeds
    # this target, overshoot -- how many multiples over target the graph
    # currently is (e.g. 1.0 at exactly double) -- is subtracted as its
    # own *flat* cost, the same shape head_pressure_coefficient already
    # uses (not multiplied by the node's own pressure the way decay_rate
    # is) -- a flat cost can actually threaten starvation_floor for an
    # ordinary-strength node once overshoot is large enough, where a
    # proportional-to-self cost only ever asymptotically approaches a
    # nonzero floor no matter how large the overshoot gets. Independent
    # of decay_rate: population_target drives real decay on its own even
    # with decay_rate left at its default of 0. None (default): no
    # target, population never costs anything regardless of live count.
    population_target: Optional[int] = None
    # Nodes across the whole graph that share the exact same token span
    # (tuple(node.tokens)) divide their intrinsic claim (found_bonus +
    # local_value) by the number of live instances of that span, once per
    # tick, before the exterior relaxation begins (see
    # FluxGraph._shared_token_intrinsic). Because this feeds the actual
    # intrinsic term _update_pressures uses, not just a starting value,
    # it's a real, standing feature of the equilibrium the graph settles
    # to -- not a post-hoc average layered on top of it (which a later
    # tick's from-scratch re-settle would just erase). Growing redundancy
    # (more live instances of the same token/word) costs every member,
    # including the strongest, for as long as the redundancy persists,
    # not just the newly-formed one -- see _attach_children for the
    # spawn-time half of this: a brand new duplicate starts at zero
    # rather than waiting for the next tick's division to catch up. On by
    # default; off reproduces pre-sharing behavior exactly (every node's
    # intrinsic claim is purely its own).
    shared_token_pressure_enabled: bool = True
    # First step toward a real circulatory system: every edge's single,
    # shared conductance (see FluxGraph._edge_conductance) still sets the
    # base "how much" for both directions, but this scales it down for
    # the *return* leg specifically -- child pressure flowing back toward
    # its parent -- while the *delivery* leg (parent flowing out toward
    # its child) stays at the unscaled base value. 1.0 (default) makes
    # every edge symmetric, identical to pre-circulatory behavior; below
    # 1.0, growth still gets full support flowing outward but a child's
    # own success reports back to its ancestors more weakly, so a strong
    # discovery no longer automatically inflates its whole lineage's
    # pressure just by existing. This is deliberately just the direction
    # split on its own -- no gates, no allow/deny lists, no altering what
    # flows, those are later, separate layers on top of this one.
    return_conductance_scale: float = 1.0
    # The graph auditor (see FluxGraph._run_graph_auditor), run once per
    # tick if on. Combinatorial by design, not an approximation of it:
    # enumerates every (ancestor, descendant) pair among currently-live
    # nodes -- every causal path in the graph, not just direct parent/
    # child edges -- and records whichever ones aren't in self.traversals
    # yet. Pure bookkeeping, not new scoring: each pair's score comes
    # directly from its two endpoints' already-known cumulative_evidence.
    # Off by default since it's still a real, uncapped per-tick cost
    # proportional to graph size -- on, it's deliberately not scaled back
    # regardless of how large the graph gets.
    graph_auditor_enabled: bool = False
    # Scalar gradient layers for every pie slice's region: layer name ->
    # function of radial position (ring index, 0 at the slice's own
    # center) -> substance presence at that radius. As many layers as
    # needed; "humidity" is the one layer present by default (uniform_
    # field, 1.0 everywhere). slice_scalar_fields overrides individual
    # layers for individual slices, keyed by slice name ("main:forward",
    # "net:7:backward", ...) -- a slice not listed just uses the shared
    # defaults.
    scalar_fields: Dict[str, Callable[[float], float]] = field(default_factory=lambda: {"humidity": uniform_field})
    slice_scalar_fields: Dict[str, Dict[str, Callable[[float], float]]] = field(default_factory=dict)


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
        # Set once by _fallback_to_cpu the first time a real GPU
        # out-of-memory error is caught (see _run_with_oom_fallback) --
        # after that, self.device is "cpu" and every subsequent model
        # call already reads it fresh, so this flag only prevents
        # redundant fallback attempts, not anything functional.
        self._cpu_fallback_active = False
        # The anchor's pump, explicit and scriptable -- see Heart. Owns
        # per-slice chamber contents between ticks; _pump_anchor feeds it
        # intake and outflow routes each beat.
        self.heart = Heart()

        self.nodes: Dict[int, FluxNode] = {}
        # Phase 1 edge scaffold (see Edge's own docstring) -- keyed by
        # (parent_id, child_id) at creation time, which never changes even
        # if a later re-root flips which end reads as "parent" in the live
        # tree; populated once per node in _attach_children, and never
        # read by any pressure/expansion/starvation code today.
        self.edges: Dict[Tuple[int, int], Edge] = {}
        # Every causal path (ancestor, descendant) that's been enumerated
        # and scored so far -- see Traversal's own docstring and
        # FluxGraph._run_graph_auditor, the only thing that populates
        # this. Keyed by (start_id, end_id) exactly like self.edges, so
        # "has this specific path already been run" is a direct lookup.
        self.traversals: Dict[Tuple[int, int], Traversal] = {}
        # Hyperedges bundling self.edges by region pair -- see MetaEdge's
        # own docstring. Rebuilt on demand by meta_edges(), not kept
        # continuously current -- there's no tick-loop consumer of these
        # yet, so eagerly recomputing them every tick would be pure waste.
        # Latest state published by each *other* physics domain, keyed by
        # domain name ("client" = the browser's gamified interface sim
        # today; a compute shader later would publish through the same
        # door). Written wholesale by absorb_external_physics (one atomic
        # reference swap, same GIL reasoning as published_snapshot -- no
        # lock needed), read by whichever backend passes consume that
        # domain's observations (see _ingest_from_rings). Latest payload
        # per domain wins; stale observations are simply reused until the
        # next one arrives.
        self.external_physics: Dict[str, Dict[str, Any]] = {}
        self.anchor_id: Optional[int] = None
        self.anchor_tokens: List[int] = []
        # The current anchor's own real local_evidence, captured at
        # promotion time and restored on demotion -- see
        # _reset_to_anchor_invariants/_reroot. The anchor's own node
        # always reads local_evidence=0.0 while it holds that role (an
        # anchor is treated as certain/free), but that's not the same
        # thing as its real value being zero.
        self.anchor_local_evidence: float = 0.0
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
        self.anchor_local_evidence = 0.0  # the original seed is fixed, not generated -- no real cost to restore later
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

    def restore_state(
        self,
        nodes: Dict[int, FluxNode],
        anchor_id: int,
        anchor_tokens: List[int],
        tick_count: int,
        anchor_local_evidence: float = 0.0,
    ) -> None:
        """Replace this (freshly-constructed, never-seeded) graph's state wholesale.

        The persistence counterpart to seed(): for resuming a graph a
        caller reconstructed from saved state (see flux_radar_server.py's
        autosave/resume) rather than growing fresh from raw seed tokens.
        The caller owns building real FluxNode objects (deserializing
        whatever it saved) -- this just installs them as this graph's own
        and derives the one thing a fresh seed() would otherwise set up
        itself (_next_id, from the live max id rather than a separately
        persisted counter that could drift out of sync with the nodes
        actually present). anchor_local_evidence defaults to 0.0 for
        state saved before this field existed -- harmless unless the
        live anchor at save time was itself a promoted (not original
        seed) node, the same one-tick imprecision as any other field a
        pre-existing save simply didn't have yet.
        """
        self.nodes = nodes
        self.anchor_id = anchor_id
        self.anchor_tokens = list(anchor_tokens)
        self.anchor_local_evidence = anchor_local_evidence
        self.tick_count = tick_count
        self._next_id = max(nodes.keys()) + 1 if nodes else 0
        self._publish_snapshot()

    def absorb_external_physics(self, domain: str, payload: Dict[str, Any]) -> None:
        """The one door another physics domain publishes through.

        The tick loop stays the backend's own; a foreign domain (the
        client's interface sim now, a shader later) runs its physics on
        its own clock and drops its latest observations here whenever it
        has them. One atomic reference swap per domain (GIL -- same
        reasoning as published_snapshot), so no locking against a tick in
        progress: a consuming pass either sees the previous complete
        payload or the new complete one, never a torn mix.
        """
        self.external_physics[domain] = payload

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
        """Tokens hung between the anchor and ``node_id``, in left-to-right reading order, plus that side's direction.

        Walks child-to-parent (leaf up to the anchor), so the per-node
        spans are collected leaf-first: for FORWARD (append -- each new
        child extends further right), that's the reverse of reading
        order, so it needs flipping before flattening. For BACKWARD
        (prepend -- each new child extends further left), leaf-first
        *is* reading order already: the deepest node is the furthest
        left, and flipping it would scramble the read exactly backward
        (this was a real bug: multi-hop backward growth read in creation
        order, e.g. "near situated the ocean" instead of "situated near
        the ocean" -- not just a display issue, since _expand_backward/
        _expand_batch/the poetic context builder all feed this same
        text to the model as real scoring context). Either way, only the
        *order of spans* flips, never a span's own internal token order,
        or a multi-token word would get its own tokens scrambled.
        """
        spans: List[List[int]] = []
        direction = self.nodes[node_id].direction
        cur = node_id
        while cur != self.anchor_id:
            node = self.nodes[cur]
            spans.append(node.tokens)
            cur = node.parent_id
        if direction is not Direction.BACKWARD:
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
        if self.config.graph_auditor_enabled:
            self._run_graph_auditor()
            self._exchange_humidity()
            self._ingest_from_rings()
            self._transport_subedges()
            self._pump_anchor()
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
        old_anchor_local_evidence = self.anchor_local_evidence

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
        self.nodes[old_anchor_id].local_evidence = old_anchor_local_evidence
        self.anchor_tokens, self.anchor_local_evidence = self._reset_to_anchor_invariants(new_root)
        self.anchor_id = new_root_id
        self._recompute_depths_from(new_root_id)

    def _reset_to_anchor_invariants(self, node: FluxNode) -> Tuple[List[int], float]:
        """Match seed()'s anchor construction, in place, for a node that's becoming the anchor.

        Returns (captured_tokens, captured_local_evidence) as they were
        just before this reset -- the caller stashes both as the new
        anchor_tokens/anchor_local_evidence, exactly the round-trip
        tokens already got: local_evidence is a permanent fact about this
        node's own token (what the model actually thought of it, fixed at
        creation), not something anchor status should get to overwrite.
        Zeroing it here is still correct *while this node serves as
        anchor* (an anchor is treated as certain/free, same as tokens
        being cleared into anchor_tokens -- see _edge_conductance's own
        "child side" framing), but it must come back when this node is
        later demoted, or every cumulative_evidence sum through it from
        then on silently treats its real cost as zero forever. Deliberately
        leaves pressure, children_ids, and created_tick alone -- pressure
        is why this node was promoted in the first place (no reason to
        discard that live signal), children_ids/created_tick are real
        facts about this node's own history that promotion doesn't change.
        """
        captured_tokens = node.tokens
        captured_local_evidence = node.local_evidence
        node.tokens = []
        node.direction = None
        node.parent_id = None
        node.depth = 0
        node.local_evidence = 0.0
        node.cumulative_evidence = 0.0
        node.low_pressure_ticks = 0
        node.expanded = True
        return captured_tokens, captured_local_evidence

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

    def orthogonal_network_roots(self) -> Dict[int, int]:
        """Maps every orthogonal node to the id of the node its "network" is rooted at.

        A network's root is the shallowest node in one contiguous
        orthogonal subtree -- the exact node whose direction first broke
        from its parent's (see orthogonal_node_ids). Two orthogonal nodes
        share a root iff they're part of the same disconnected-feeling
        branch; unrelated orthogonal branches elsewhere in the tree get
        distinct roots. Existing display consumers (the radar's per-network
        pie wedges) use this to group a whole branch as one visual unit
        instead of scattering it across unrelated groups. Processed in
        depth order (shallowest first) so a child always resolves after
        its parent -- self.nodes[node_id].depth is already well-defined
        and stable for this since it's only recomputed on re-root, never
        mid-scan.
        """
        orthogonal = self.orthogonal_node_ids()
        roots: Dict[int, int] = {}
        for node_id in sorted(orthogonal, key=lambda nid: self.nodes[nid].depth):
            parent_id = self.nodes[node_id].parent_id
            roots[node_id] = roots.get(parent_id, node_id)
        return roots

    def _edge_region(self, node_id: int, network_roots: Dict[int, int]) -> str:
        """Which region node_id belongs to for MetaEdge bundling -- "main"
        for the anchor's own bowtie, or "net:<root id>" for an orthogonal
        cousin network, using the exact same grouping the radar's own pie
        wedges already use (see orthogonal_network_roots), not a second
        regioning concept invented just for this.
        """
        root_id = network_roots.get(node_id)
        return "main" if root_id is None else f"net:{root_id}"

    def meta_edges(self) -> Dict[Tuple[str, str], MetaEdge]:
        """Bundle every Edge in self.edges into hyperedges by (from_region,
        to_region) pair -- see MetaEdge's own docstring. Rebuilt fresh
        from self.edges every call, not cached: there's no tick-loop
        consumer keeping a cached version current, and self.edges only
        ever grows (nothing is pruned from it), so this is proportional
        to edge count, not something that needs incremental maintenance
        yet.
        """
        network_roots = self.orthogonal_network_roots()
        groups: Dict[Tuple[str, str], List[Tuple[Tuple[int, int], Edge]]] = {}
        for key, edge in self.edges.items():
            from_region = self._edge_region(edge.from_id, network_roots)
            to_region = self._edge_region(edge.to_id, network_roots)
            groups.setdefault((from_region, to_region), []).append((key, edge))

        result: Dict[Tuple[str, str], MetaEdge] = {}
        for meta_id, (region_pair, members) in enumerate(groups.items()):
            from_region, to_region = region_pair
            forward_mean = sum(e.forward.conductance for _, e in members) / len(members)
            reverse_mean = sum(e.reverse.conductance for _, e in members) / len(members)
            result[region_pair] = MetaEdge(
                id=meta_id,
                from_region=from_region,
                to_region=to_region,
                member_edge_keys={key for key, _ in members},
                forward=Channel(conductance=forward_mean),
                reverse=Channel(conductance=reverse_mean),
                updated_tick=self.tick_count,
            )
        return result

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

    def _directional_conductance(self, node_id: int, neighbor_id: int) -> float:
        """_edge_conductance's base value, scaled for which leg this flow is.

        neighbor_id is node_id's *parent* -> this is the delivery leg
        (support flowing out from parent to child, away from the anchor):
        unscaled, exactly _edge_conductance's own value. neighbor_id is
        node_id's *child* -> this is the return leg (a child's own
        pressure flowing back toward its parent): scaled by
        config.return_conductance_scale. 1.0 (default) makes both legs of
        every edge identical, matching pre-circulatory behavior exactly.
        """
        base = self._edge_conductance(node_id, neighbor_id)
        if neighbor_id == self.nodes[node_id].parent_id:
            return base
        return base * self.config.return_conductance_scale

    def _shared_token_intrinsic(self) -> Dict[int, float]:
        """Nodes sharing the exact same token span divide their intrinsic claim
        by how many live instances of that span exist in the graph.

        Computed once per tick, structurally (from tokens/burned, which
        never change mid-relaxation), *before* _settle_circuit begins --
        not folded into the per-sweep loop, and not a post-hoc average
        applied after settling finishes. Each member's ordinary intrinsic
        claim (found_bonus + local_value) is divided by the group's live
        member count, so growing redundancy costs every instance,
        including the strongest, for as long as the redundancy persists.
        A lone instance (no live sibling sharing its token) is untouched
        -- there's nothing to divide against.

        Returns a node_id -> intrinsic override map used in place of the
        node's own found_bonus + local_value for every node that has one;
        every other node keeps computing its intrinsic term the ordinary
        way. Because this is computed once and then used as the actual
        intrinsic term for the whole relaxation (not just an initial
        value), it's a real, standing feature of the equilibrium
        _update_pressures converges to -- not something the next
        iteration's from-scratch fixed point would silently erase.
        """
        overrides: Dict[int, float] = {}
        if not self.config.shared_token_pressure_enabled:
            return overrides
        groups: Dict[Tuple[int, ...], List[FluxNode]] = {}
        for node in self.nodes.values():
            if node.burned or not node.tokens:
                continue
            groups.setdefault(tuple(node.tokens), []).append(node)
        for members in groups.values():
            if len(members) < 2:
                continue
            count = len(members)
            for n in members:
                overrides[n.id] = (self.config.found_bonus + n.local_value) / count
        return overrides

    def _update_pressures(
        self,
        shared_intrinsic: Optional[Dict[int, float]] = None,
        forward_reach: float = 0.0,
        backward_reach: float = 0.0,
    ) -> float:
        """One relaxation sweep. Returns the largest pressure change seen.

        inflow is a conductance-weighted average of neighbor pressure,
        normalized by *total conductance* -- sum(conductance_i * pressure_i)
        / (1 + sum(conductance_i)) -- the standard Gauss-Seidel update rule
        for solving nodal voltages in a real resistor network. Normalizing
        by raw neighbor *count* instead (an earlier version of this) is not
        that: it dilutes a node's inflow by how many neighbors it happens
        to have, independent of how strong those connections actually are,
        so a well-connected hub with many good edges got no more support
        than a node with one mediocre one -- exactly backwards from how a
        real circuit (or a real vascular system) works, where more/thicker
        vessels into a junction mean more flow reaches it, not the same
        flow diluted further. This is why a heavily-relied-on node (like
        the anchor once anchor_can_decay is on) could starve as easily as
        a peripheral leaf despite being structurally load-bearing for much
        of the graph: connectivity conferred no advantage at all under the
        count-normalized version.

        decay_rate (if set) is a self-proportional loss term
        (decay_cost = decay_rate * node.pressure), same shape head_cost
        uses. Proportional-to-self terms are deceptively weak at the
        actual fixed point, though: solving the equilibrium algebraically
        (ignoring inflow/other terms for a moment), P = intrinsic -
        decay_rate*P implies P = intrinsic / (1 + decay_rate) -- decay_rate
        only ever divides the equilibrium down by that ratio, so even a
        large decay_rate barely dents an already-strong node (decay_rate=1
        only halves it), and this is a *good* property for the base knob:
        a gentle, always-on leak that never disproportionately punishes a
        node just for being strong.

        population_target (if set) is deliberately NOT built the same way.
        A homeostatic population cap needs to be able to actually threaten
        starvation_floor for an ordinary-strength node once the graph is
        far enough over target, not just asymptotically approach a nonzero
        floor no matter how large the overshoot gets -- so once live count
        exceeds target, overshoot (how many multiples over target the
        graph currently is) is subtracted as its own *flat* cost
        (population_cost), the same shape head_cost already uses, not
        multiplied by the node's own pressure. This is what actually lets
        population_target win against a healthy node's baseline once
        overshoot is large enough, the way head_pressure_coefficient
        already reliably does against distance -- an earlier version
        folded overshoot into a shared "effective_decay_rate" multiplied
        by pressure, which is exactly the weak-at-equilibrium shape above,
        and could leave population growing indefinitely with no decay
        knob actually capable of stopping it.

        balance_weight (if set) adds its own gain term, mirroring
        head_pressure_coefficient's shape but as a benefit instead of a
        cost: every live node on whichever direction is currently behind
        (forward_reach vs backward_reach -- see FluxGraph._direction_reach)
        gets cfg.balance_weight * the reach gap added straight into its
        settled pressure, for as long as the gap persists. This is the
        same asymmetry signal _expansion_priority uses to bias *which*
        node gets picked for compute, applied here to *how much pressure
        that side is allowed to keep* once it exists -- without this,
        boosting a lagging node's odds of being picked doesn't stop it
        starving right back out the moment it's expanded, since
        starvation reads settled pressure, not expansion priority.

        shared_intrinsic (see _shared_token_intrinsic), if given,
        replaces this sweep's intrinsic term for whichever nodes it
        covers. Defaults to empty (ordinary per-node intrinsic
        everywhere) so this stays directly callable on its own, the way
        existing tests and _settle_circuit's docstring both rely on.
        """
        cfg = self.config
        shared_intrinsic = shared_intrinsic or {}
        previous = {nid: n.pressure for nid, n in self.nodes.items() if not n.burned}

        population_cost = 0.0
        if cfg.population_target:
            live_count = len(previous)
            if live_count > cfg.population_target:
                overshoot = (live_count - cfg.population_target) / cfg.population_target
                population_cost = overshoot

        max_delta = 0.0
        for node_id, node in self.nodes.items():
            if node.burned:
                continue
            if node_id == self.anchor_id and not cfg.anchor_can_decay:
                continue
            neighbors = self._neighbors(node_id)
            if neighbors:
                conductances = [self._directional_conductance(node_id, n) for n in neighbors]
                weighted_sum = sum(c * previous.get(n, 0.0) for c, n in zip(conductances, neighbors))
                inflow = weighted_sum / (1 + sum(conductances))
            else:
                inflow = 0.0
            intrinsic = shared_intrinsic.get(node_id, node.local_value + cfg.found_bonus)
            head_cost = cfg.head_pressure_coefficient * abs(node.height)
            decay_cost = cfg.decay_rate * node.pressure
            balance_gain = 0.0
            if cfg.balance_weight and node.direction is not None:
                own_reach = forward_reach if node.direction is Direction.FORWARD else backward_reach
                opposite_reach = backward_reach if node.direction is Direction.FORWARD else forward_reach
                gap = opposite_reach - own_reach
                if gap > 0:
                    balance_gain = cfg.balance_weight * gap
            new_pressure = max(
                0.0, intrinsic + cfg.damping * inflow + balance_gain - head_cost - decay_cost - population_cost
            )
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

        Shared-token trading (_shared_token_intrinsic) and the forward/
        backward reach gap (for balance_weight) both happen once here,
        before the very first sweep -- structural facts about the tree
        that the whole exterior relaxation then treats as fixed input for
        this tick, not something recomputed sweep to sweep.
        """
        cfg = self.config
        shared_intrinsic = self._shared_token_intrinsic()
        forward_reach = self._direction_reach(Direction.FORWARD)
        backward_reach = self._direction_reach(Direction.BACKWARD)
        for i in range(cfg.max_relaxation_iterations):
            delta = self._update_pressures(shared_intrinsic, forward_reach, backward_reach)
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

    def _direction_branch_factor(self, direction: Optional[Direction]) -> int:
        cfg = self.config
        override = cfg.forward_branch_factor if direction is Direction.FORWARD else cfg.backward_branch_factor
        return override if override is not None else cfg.branch_factor

    def _direction_hot_loop_depth(self, direction: Optional[Direction]) -> int:
        cfg = self.config
        override = cfg.forward_hot_loop_depth if direction is Direction.FORWARD else cfg.backward_hot_loop_depth
        return override if override is not None else cfg.hot_loop_depth

    def _direction_selection_mode(self, direction: Optional[Direction]) -> str:
        cfg = self.config
        return cfg.forward_selection_mode if direction is Direction.FORWARD else cfg.backward_selection_mode

    def _direction_top_p(self, direction: Optional[Direction]) -> float:
        cfg = self.config
        return cfg.forward_top_p if direction is Direction.FORWARD else cfg.backward_top_p

    def _effective_branch_factor(self, node: FluxNode) -> int:
        """node's direction's branch_factor, suppressed by ambient auxin felt at this node.

        Only the *final* number of children a node commits to shrinks --
        internal search width (round-0 shortlists, word-growth's own beam)
        stays wide regardless, so a suppressed node still searches broadly
        before narrowing down to fewer winners.
        """
        cfg = self.config
        base = self._direction_branch_factor(node.direction)
        if cfg.auxin_suppression <= 0:
            return base
        suppression = 1.0 / (1.0 + cfg.auxin_suppression * node.auxin_level)
        return max(1, round(base * suppression))

    def _top_p_keep_count(self, sorted_scores: List[float], top_p: float) -> int:
        """How many of sorted_scores (true log-probabilities, descending) to
        keep so their cumulative probability reaches top_p.

        scores here are always real log_softmax values over the full
        vocabulary (see ChoicePolicy.choose's contract), not renormalized
        over just the shortlist -- so summing exp(score) directly gives a
        true lower-bound estimate of cumulative probability mass, exactly
        matching nucleus sampling's actual definition. Always keeps at
        least 1 candidate if any exist, even if its own probability
        already exceeds top_p.
        """
        if not sorted_scores:
            return 0
        cumulative = 0.0
        for i, score in enumerate(sorted_scores):
            cumulative += math.exp(score)
            if cumulative >= top_p:
                return i + 1
        return len(sorted_scores)

    def _resolve_keep_count(self, node: FluxNode, spans: List[Tuple[List[int], float]]) -> int:
        """How many of `spans` (token_span, score) this node actually keeps as children.

        "topk" mode: _effective_branch_factor, exactly today's fixed-count
        behavior. "topp" mode: nucleus selection over spans' own scores --
        width tracks the model's real uncertainty at this node instead of
        an arbitrary fixed count. auxin_suppression still caps the topp
        count when it's on, so apical dominance means something in either
        mode; unsuppressed topp mode is otherwise uncapped by branch_factor
        entirely, since there's no fixed-K concept to apply once selection
        is coverage-based.
        """
        if self._direction_selection_mode(node.direction) != "topp":
            return self._effective_branch_factor(node)
        if not spans:
            return 0
        sorted_scores = sorted((sc for _, sc in spans), reverse=True)
        keep = self._top_p_keep_count(sorted_scores, self._direction_top_p(node.direction))
        if self.config.auxin_suppression > 0:
            keep = min(keep, self._effective_branch_factor(node))
        return max(1, keep)

    def _direction_reach(self, direction: Direction) -> float:
        """How far a direction's frontier currently extends from the anchor.

        max(abs(height)) among that direction's own live nodes, 0 if it
        has none yet. Pure topology, recomputed fresh each call -- cheap
        enough to call once per tick, and always current.
        """
        reaches = [abs(n.height) for n in self.nodes.values() if not n.burned and n.direction is direction]
        return max(reaches) if reaches else 0.0

    def _expansion_priority(self, node: FluxNode, forward_reach: float = 0.0, backward_reach: float = 0.0) -> float:
        """Pressure (exploit) + neighborhood rollup (digest) + wait bonus (explore) + balance (symmetry).

        Pure top-pressure selection is greedy exploitation: whatever looks
        best right now always wins, forever. The neighborhood term uses
        the *parent's* rollup (not the candidate's own -- a fresh leaf's
        own rollup is trivially just itself) so siblings of a known-good
        discovery get a boost. The wait-time term grows with ticks spent
        eligible-but-unpicked, so a merely-ordinary node isn't starved of
        its turn forever just because something else looked better first.
        The balance term corrects a *directional* version of the same
        starvation: if forward has pulled ahead of backward (or vice
        versa), the trailing side's candidates get a boost proportional to
        the gap, so one side compounding its lead every tick doesn't run
        away while the other stalls out. forward_reach/backward_reach are
        graph-wide quantities -- callers compute them once per tick
        (_direction_reach) rather than per-candidate.
        """
        cfg = self.config
        parent = self.nodes[node.parent_id] if node.parent_id is not None else None
        neighborhood_bonus = cfg.rollup_weight * math.exp(parent.rollup_mean) if parent is not None else 0.0
        wait = max(0, self.tick_count - node.created_tick)
        exploration_bonus = cfg.exploration_constant * math.sqrt(wait)
        balance_bonus = 0.0
        if cfg.balance_weight and node.direction is not None:
            own_reach = forward_reach if node.direction is Direction.FORWARD else backward_reach
            opposite_reach = backward_reach if node.direction is Direction.FORWARD else forward_reach
            gap = opposite_reach - own_reach
            if gap > 0:
                balance_bonus = cfg.balance_weight * gap
        return node.pressure + neighborhood_bonus + exploration_bonus + balance_bonus

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
        """Pick this tick's compute-budget candidates, split fairly by direction.

        Pure global top-K-by-priority has no floor per direction: nothing
        stops one side from winning the *entire* budget on a given tick,
        and once it does, this is a real positive-feedback loop, not just
        a one-tick fluke -- more live leaves on that side next tick means
        more chances to top the priority sort again, which (especially
        with hot_loop_depth > 1, where a single winning tick can inject
        branch_factor**depth new nodes at once) can run away to near-total
        dominance of one direction within a handful of ticks. balance_
        weight pushes back on this too, but it's optional and off by
        default; this floor holds even at balance_weight=0.

        forward_budget_per_tick/backward_budget_per_tick, if either is
        set, are hard per-direction caps -- exactly that many (or fewer,
        if that side is thin) get expanded, with no redistribution to the
        other side even if it has room. Left unset (the default, both
        None), the shared compute_budget_per_tick is split as an even
        floor instead (forward gets half, backward the remainder), and
        whatever's left over if one side has fewer eligible candidates
        than its floor rolls over to the other side's next-best -- so
        budget is never wasted just because one direction is temporarily
        thin, the way a hard per-direction cap deliberately can be.

        Each direction's selected batch runs its own hot loop, at its own
        hot_loop_depth (see _direction_hot_loop_depth) -- they can't share
        one combined hot loop once depths can differ per direction, so
        forward and backward no longer share a single batched model call
        per hot-loop round the way a single mixed batch used to.

        If the anchor itself currently has zero live children -- every
        forward and backward node has starved away and burned, leaving
        only the anchor standing -- this is a hard stall, not just a slow
        tick: _expandable_nodes() never includes the anchor (it isn't a
        normal forward/backward node -- no direction, so _expand_batch has
        nothing to score it with), so with nothing else live either, the
        candidate pools below would both come back empty every single
        tick from here on, forever. The anchor doesn't get to just sit
        there once it's the only thing left -- it re-seeds both directions
        fresh, the same bootstrap spawn_first_children() used originally,
        so growth actually resumes instead of the graph going silently and
        permanently dead.
        """
        if not self._live_children(self.anchor_id):
            self._expand_forward(self.anchor_id)
            self._expand_backward(self.anchor_id)
            return

        forward_reach = self._direction_reach(Direction.FORWARD)
        backward_reach = self._direction_reach(Direction.BACKWARD)

        def priority(n: FluxNode) -> float:
            return self._expansion_priority(n, forward_reach, backward_reach)

        forward_pool = sorted(
            (n for n in self._expandable_nodes() if n.direction is Direction.FORWARD),
            key=priority, reverse=True,
        )
        backward_pool = sorted(
            (n for n in self._expandable_nodes() if n.direction is Direction.BACKWARD),
            key=priority, reverse=True,
        )

        cfg = self.config
        explicit_budgets = cfg.forward_budget_per_tick is not None or cfg.backward_budget_per_tick is not None
        if explicit_budgets:
            forward_floor = cfg.forward_budget_per_tick or 0
            backward_floor = cfg.backward_budget_per_tick or 0
        else:
            forward_floor = cfg.compute_budget_per_tick // 2
            backward_floor = cfg.compute_budget_per_tick - forward_floor

        forward_selected = forward_pool[:forward_floor]
        backward_selected = backward_pool[:backward_floor]

        if not explicit_budgets:
            leftover_budget = cfg.compute_budget_per_tick - len(forward_selected) - len(backward_selected)
            if leftover_budget > 0:
                remaining = sorted(
                    forward_pool[forward_floor:] + backward_pool[backward_floor:],
                    key=priority, reverse=True,
                )
                for n in remaining[:leftover_budget]:
                    if n.direction is Direction.FORWARD:
                        forward_selected.append(n)
                    else:
                        backward_selected.append(n)

        if forward_selected:
            self._expand_batch_hot_loop(forward_selected, self._direction_hot_loop_depth(Direction.FORWARD))
        if backward_selected:
            self._expand_batch_hot_loop(backward_selected, self._direction_hot_loop_depth(Direction.BACKWARD))

    def _expand_batch_hot_loop(self, nodes: List[FluxNode], depth: int) -> None:
        """Run _expand_batch repeatedly, feeding each round's brand-new
        children back in as the next round's targets -- a real, honest
        multi-level beam search down whichever lineage(s) this tick's pick
        started from, happening entirely within one tick instead of one
        level per tick with the pressure system re-deciding who goes next
        in between. Every round uses the *exact same* per-node machinery
        as an ordinary single-level expansion -- branch_factor, no_repeat_
        ngram_size, word_trie, the poetic reranker, everything -- so this
        knob only controls how many rounds happen back-to-back before
        control returns to the tick loop, never how any individual choice
        gets made. It's also not free: round 2 works from round 1's *own*
        children, so total nodes created can grow like branch_factor**depth
        for the batch that was picked -- depth=1 (the default) is exactly
        today's single-level-per-tick behavior, higher values trade real
        compute for letting a promising lineage run out to a full phrase
        or clause in one shot instead of drip-feeding one token per tick.

        population_target (if set) caps rounds here too, not just next
        tick's decay: decay_rate/population_target only ever throttle
        *pressure*, recomputed once per tick in _update_pressures --
        they can't see or react to growth from a hot loop still in
        progress, and burning is gated by burn_after_ticks consecutive
        low-pressure ticks on top of that. A hot loop left unchecked can
        trivially blow the graph past population_target in a single
        tick, well before either mechanism gets a chance to respond, so
        this stops taking further rounds the moment live population
        reaches the target -- the round that crossed it still completes
        (never interrupted mid-batch), it just doesn't start another one.
        """
        cfg = self.config
        frontier = nodes
        for _ in range(max(1, depth)):
            if not frontier:
                return
            for node in frontier:
                node.expanded = True
            before = set(self.nodes.keys())
            self._expand_batch(frontier)
            new_ids = set(self.nodes.keys()) - before
            frontier = [self.nodes[nid] for nid in new_ids]
            if cfg.population_target:
                live_count = sum(1 for n in self.nodes.values() if not n.burned)
                if live_count >= cfg.population_target:
                    return

    def _starve_and_burn(self) -> None:
        # Every live non-anchor node is a candidate now, not just current
        # leaves -- an internal (ancestor) node with a live child used to
        # be permanently exempt from burning no matter how low its own
        # pressure fell, which meant population_target/decay_rate could
        # only ever thin the current leaf fringe: the accumulated trunk
        # of ancestor nodes only ever grew, so total live population was
        # structurally incapable of actually shrinking, regardless of how
        # aggressively either knob was set. See _burn for what happens to
        # an internal node's own live descendants once it goes.
        cfg = self.config
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            if node.pressure < cfg.starvation_floor:
                node.low_pressure_ticks += 1
            else:
                node.low_pressure_ticks = 0
            if node.low_pressure_ticks >= cfg.burn_after_ticks:
                self._burn(node_id)

    def _burn(self, node_id: int) -> None:
        """Burn ``node_id`` and, if it had any, its whole live subtree.

        A burned internal node's live children can't be left dangling:
        _update_pressures excludes burned nodes from ``previous`` entirely,
        so a descendant whose parent just burned would lose its only
        conduit back to the anchor's pressure network on the very next
        sweep, and path_tokens/best_leaf would still be free to walk into
        a node no longer actually connected to anything live. Cascading
        the burn is the coherent version of "a starved branch withers" --
        the whole thing goes at once, not just its current tip -- rather
        than leaving orphaned nodes alive-but-disconnected until each one
        separately starves its own way down to a leaf over further ticks.
        """
        stack = [node_id]
        newly_burned: List[int] = []
        while stack:
            cur_id = stack.pop()
            cur = self.nodes[cur_id]
            if cur.burned:
                continue
            cur.burned = True
            newly_burned.append(cur_id)
            if self.config.verbose:
                print(f"  [burn] node {cur_id} (tokens={cur.tokens}, dir={cur.direction}) starved out")
            stack.extend(self._live_children(cur_id))
        node = self.nodes[node_id]
        if node.parent_id is not None:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        if newly_burned:
            self._prune_traversals_touching(newly_burned)

    def _prune_traversals_touching(self, burned_ids: List[int]) -> None:
        """Drop every cached Traversal whose start or end just burned.

        self.traversals otherwise only ever grows: a Traversal is
        recorded once by _run_graph_auditor and never re-derived, so
        across a long-running session it accumulates one entry per
        (ancestor, descendant) pair ever seen -- not just the ones still
        meaningful now. That's a real unbounded-memory risk on its own,
        and since audit_edge_influence (called every single tick, from
        snapshot_graph) walks every recorded traversal, it's an
        unboundedly growing per-tick cost too. A traversal referencing a
        burned node isn't just stale -- it's describing a causal path
        that no longer exists anywhere in the live graph, so there's
        nothing left worth keeping it for. This keeps self.traversals
        bounded by roughly "combinatorial pairs among currently-live
        nodes" instead of "cumulative pairs across the whole session."
        """
        burned_set = set(burned_ids)
        dead_keys = {key for key in self.traversals if key[0] in burned_set or key[1] in burned_set}
        for key in dead_keys:
            del self.traversals[key]
        # Each of those traversals' own two SubEdges (see _record_traversal)
        # are sitting in every real Edge's subedges list along its old
        # path -- same unbounded-growth risk as self.traversals itself,
        # so they go too.
        if dead_keys:
            for edge in self.edges.values():
                if edge.subedges:
                    edge.subedges = [s for s in edge.subedges if s.traversal_key not in dead_keys]

    def evaluator(self) -> Dict[Tuple[int, int], Traversal]:
        """Every causal path recorded so far -- see Traversal's own
        docstring for why this is just self.traversals itself: a
        traversal isn't a fixed forward/backward pair, or a single-
        direction record, or anything else with a shape to classify by,
        it's just "this causal path has been run." Exists as its own
        named accessor since "the evaluator" is the right name for "what
        holds the paths and scores of every scored phrase," even though,
        today, it's a direct view over already-public state.
        """
        return self.traversals

    # ------------------------------------------------------------------
    # Expansion: generate real children via the model
    # ------------------------------------------------------------------
    def _context_window(self, tokens: List[int]) -> List[int]:
        limit = self.config.max_context_tokens
        if len(tokens) <= limit:
            return tokens
        return tokens[-limit:]

    def _fallback_to_cpu(self) -> None:
        """Move the model -- and every future tensor this graph builds --
        to CPU, once. Triggered by _run_with_oom_fallback catching a real
        GPU out-of-memory error: rather than crash the whole run because
        something else on the machine is contending for GPU memory right
        now, finish out on CPU. Slower, but it keeps going. Idempotent:
        a second call (a later chunk hitting OOM after the first already
        switched) is a cheap no-op.
        """
        if self._cpu_fallback_active:
            return
        self._cpu_fallback_active = True
        model = getattr(self.model_wrapper, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
        self.device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- best-effort cache clearing, never worth failing over
            pass
        print("  [oom-fallback] GPU out of memory -- this graph is switching to CPU for the rest of the run")

    def _run_with_oom_fallback(self, fn: Callable[[], Any]) -> Any:
        """Run fn(); on a real GPU out-of-memory error, fall back to CPU
        once (see _fallback_to_cpu) and retry fn() exactly once more.

        fn must rebuild its own input tensors from self.device rather
        than a value captured before this call -- otherwise a retry
        would hand freshly-CPU-bound model weights the original GPU
        tensors right back. A second failure (a real CPU MemoryError, or
        a second OOM after the fallback already ran) is a genuine error
        and propagates.
        """
        try:
            return fn()
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            self._fallback_to_cpu()
            return fn()

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

            def _build_and_forward(padded=padded, mask=mask):
                # Rebuilt from self.device fresh, not the outer `device`
                # captured before this loop started -- a retry after
                # _fallback_to_cpu needs CPU tensors, not the original
                # GPU ones handed to a now-CPU model.
                bt = backend_cls.tensor(padded, dtype=long_dtype, device=self.device)
                am = backend_cls.tensor(mask, dtype=long_dtype, device=self.device)
                return bt, self.model_wrapper.forward(input_ids=bt.data, attention_mask=am.data)

            batch_tokens, outputs = self._run_with_oom_fallback(_build_and_forward)
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
        # A wider shortlist is worth the extra (comparatively cheap: decode
        # text + heuristics, or a bounded subword beam search) work when
        # something downstream wants more than the base round-0 candidate
        # count to choose from -- poetic/word_trie reranking, or "topp"
        # mode, which needs enough of the real distribution's tail visible
        # to find where cumulative probability actually crosses top_p
        # rather than being capped by a narrow topk-sized fetch.
        needs_wide_shortlist = poetic is not None or word_trie is not None
        for node in nodes:
            base_shortlist_k = (
                self.config.top_p_shortlist_ceiling
                if self._direction_selection_mode(node.direction) == "topp"
                else self._direction_branch_factor(node.direction)
            )
            if node.direction is Direction.FORWARD:
                last_logits = forward_logits[node.id].unsqueeze(0)
                shortlist_k = base_shortlist_k
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
                    node, "forward", spans, self._resolve_keep_count(node, spans)
                )
                if spans:
                    token_spans, scs = zip(*spans)
                    self._attach_children(node.id, Direction.FORWARD, list(scs), list(token_spans))
            else:
                scored = sorted(
                    backward_scores.get(node.id, []), key=lambda pair: pair[1], reverse=True
                )
                shortlist_k = base_shortlist_k
                if needs_wide_shortlist:
                    shortlist_k = max(shortlist_k, self.config.poetic_shortlist_k)
                round0 = scored[:shortlist_k]
                if word_trie is not None:
                    spans = self._grow_backward_word(node, round0)
                else:
                    spans = [([tid], sc) for tid, sc in round0]
                spans = self._select_final_candidates(
                    node, "backward", spans, self._resolve_keep_count(node, spans)
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
        branch_factor = self._direction_branch_factor(Direction.FORWARD)
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
            def _build_and_forward(padded=padded, mask=mask):
                bt = backend_cls.tensor(padded, dtype=long_dtype, device=self.device)
                am = backend_cls.tensor(mask, dtype=long_dtype, device=self.device)
                return bt, self.model_wrapper.forward(input_ids=bt.data, attention_mask=am.data)

            batch_tokens, outputs = self._run_with_oom_fallback(_build_and_forward)
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
        branch_factor = self._direction_branch_factor(Direction.BACKWARD)
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

                def _score(tokens=beam["tokens"], candidate_ids=candidate_ids, score_kwargs=score_kwargs):
                    suffix_t = backend_cls.tensor(tokens, dtype=ops.long_dtype, device=self.device)
                    cand_t = backend_cls.tensor(candidate_ids, dtype=ops.long_dtype, device=self.device)
                    return self.backward_scorer.score_candidates(suffix_t, cand_t, **score_kwargs)

                raw = self._run_with_oom_fallback(_score)
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

        shortlist = candidates[: max(self._direction_branch_factor(node.direction), self.config.poetic_shortlist_k)]
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

        def _build_and_forward():
            r = backend_cls.tensor([full], dtype=ops.long_dtype, device=self.device)
            m = backend_cls.tensor([[1] * len(full)], dtype=ops.long_dtype, device=self.device)
            return r, self.model_wrapper.forward(input_ids=r.data, attention_mask=m.data)

        row, outputs = self._run_with_oom_fallback(_build_and_forward)
        logits = row.ensure_tensor(outputs["logits"])
        last_logits = logits[0, -1, :].unsqueeze(0)

        blocked = self._repeated_ngram_tokens(full, "append")
        if blocked:
            last_logits[0, list(blocked)] = float("-inf")

        scores, indices = self.choice_policy.choose(last_logits, k=self._direction_branch_factor(Direction.FORWARD))
        spans = [[i] for i in indices.tolist()[0]]
        self._attach_children(node_id, Direction.FORWARD, scores.tolist()[0], spans)

    def _expand_backward(self, node_id: int) -> None:
        prefix_tokens, _ = self.path_tokens(node_id)  # tokens between anchor and node_id
        fwd_leaf = self.best_leaf(Direction.FORWARD)
        fwd_tokens = self.path_tokens(fwd_leaf)[0] if fwd_leaf is not None else []
        suffix = self._context_window(prefix_tokens + self.anchor_tokens + fwd_tokens)

        ops = self.tensor_ops
        blocked = self._repeated_ngram_tokens(suffix, "prepend")

        def _build_and_score():
            suffix_t = type(ops).tensor(suffix, dtype=ops.long_dtype, device=self.device)
            pool = self.backward_scorer.candidate_pool(
                ops, vocab_size=self.backward_scorer.tokenizer.vocab_size, device=self.device
            )
            if blocked:
                pool_list = pool.tolist()
                keep_idx = [i for i, tid in enumerate(pool_list) if tid not in blocked]
                if keep_idx:
                    pool = pool[keep_idx]
            if self.config.verbose:
                print(f"  [expand-backward] node {node_id}: scoring {pool.shape[0]} candidates ...")
            scores = self.backward_scorer.score_candidates(
                suffix_t, pool, left_context=self.config.backward_left_context
            )
            return pool, scores

        pool, raw_scores = self._run_with_oom_fallback(_build_and_score)
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
        top_scores, top_idx = AbstractTensor.topk(raw_scores, k=self._direction_branch_factor(Direction.BACKWARD), dim=0)
        candidate_spans = [[int(pool[i].item())] for i in top_idx.tolist()]
        self._attach_children(node_id, Direction.BACKWARD, top_scores.tolist(), candidate_spans)

    def _attach_children(
        self, parent_id: int, direction: Direction, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        parent = self.nodes[parent_id]
        cfg = self.config
        # Every node created here that shares a token span with one
        # already live elsewhere in the graph spawns with zero pressure
        # instead of the usual found_bonus+exp(score) claim -- its
        # "volume" is already occupied. Seeded from what's live *before*
        # this batch, then grown as siblings are added, so two duplicate
        # candidates attached in the very same call also see each other,
        # not just pre-existing nodes. Ongoing division by group size
        # (see _update_pressures) is what keeps weakening every instance
        # as redundancy grows beyond this one tick's formation.
        live_token_spans = None
        if cfg.shared_token_pressure_enabled:
            live_token_spans = {tuple(n.tokens) for n in self.nodes.values() if not n.burned and n.tokens}
        for score, tokens in zip(scores, token_spans):
            child_id = self._alloc_id()
            cumulative = parent.cumulative_evidence + float(score)
            new_depth = parent.depth + 1
            token_key = tuple(int(t) for t in tokens)
            if live_token_spans is not None and token_key in live_token_spans:
                pressure = 0.0
            else:
                pressure = cfg.found_bonus + math.exp(float(score))
            self.nodes[child_id] = FluxNode(
                id=child_id,
                tokens=list(token_key),
                direction=direction,
                parent_id=parent_id,
                depth=new_depth,
                local_evidence=float(score),
                pressure=pressure,
                created_tick=self.tick_count,
                cumulative_evidence=cumulative,
                rollup_mean=cumulative / new_depth,
            )
            parent.children_ids.append(child_id)
            if live_token_spans is not None:
                live_token_spans.add(token_key)
            # Phase 1 edge scaffold -- see Edge's own docstring. forward/
            # reverse conductance mirror _directional_conductance exactly
            # (so return_conductance_scale's real delivery/return
            # asymmetry shows up here too, not just the symmetric base
            # _edge_conductance value), captured once at creation as a
            # real object; formation/seed_id_at_formation are permanent,
            # independent of anything a later re-root does to this node's
            # own .direction/.depth.
            self.edges[(parent_id, child_id)] = Edge(
                from_id=parent_id,
                to_id=child_id,
                forward=Channel(conductance=self._directional_conductance(child_id, parent_id)),
                reverse=Channel(conductance=self._directional_conductance(parent_id, child_id)),
                formation="postfix_beam" if direction is Direction.FORWARD else "prefix_beam",
                seed_id_at_formation=self.anchor_id,
                created_tick=self.tick_count,
            )

    def _run_graph_auditor(self) -> None:
        """Enumerate every causal path among currently-live nodes -- every
        (ancestor, descendant) pair along a real parent chain -- and
        record whichever ones aren't in self.traversals yet.

        Combinatorial on purpose: for a tree of live nodes, the number of
        such pairs is the sum, over every node, of how many live ancestors
        it has -- finite, bounded by the graph's own actual size and its
        directional structure, not an artificial window. This does NOT
        create Edges -- Edge is just the plain container for "these two
        nodes are directly connected"; a Traversal is the thing that
        actually represents a scored causal path, short or long, anywhere
        in the graph.

        No new model call is needed here: a sub-path's score is exactly
        derivable from its two endpoints' already-known cumulative_
        evidence (see Traversal's own docstring for why), since that
        value already sums each token's own independently-scored local_
        evidence along its real path. Real new scoring only ever happens
        when a node is first created (_attach_children) -- this is pure
        bookkeeping on top of that, not a second scoring pass.
        """
        live_ids = [nid for nid, n in self.nodes.items() if not n.burned]
        live_set = set(live_ids)
        for end_id in live_ids:
            cur_id = self.nodes[end_id].parent_id
            while cur_id is not None and cur_id in live_set:
                key = (cur_id, end_id)
                if key not in self.traversals:
                    self._record_traversal(cur_id, end_id)
                cur_id = self.nodes[cur_id].parent_id

    def _record_traversal(self, start_id: int, end_id: int) -> None:
        node_ids: List[int] = []
        cur_id = end_id
        while cur_id != start_id:
            node_ids.append(cur_id)
            cur_id = self.nodes[cur_id].parent_id
        node_ids.append(start_id)
        node_ids.reverse()  # start -> end, tree order along the real chain

        start = self.nodes[start_id]
        end = self.nodes[end_id]
        hops = end.depth - start.depth
        mean_score = (end.cumulative_evidence - start.cumulative_evidence) / hops if hops else 0.0

        # Every Traversal adds two SubEdges -- one going one way, one
        # going the other -- held by every real Edge its path crosses.
        forward_sub = SubEdge(traversal_key=(start_id, end_id), direction="forward")
        reverse_sub = SubEdge(traversal_key=(start_id, end_id), direction="reverse")
        for a, b in zip(node_ids, node_ids[1:]):
            edge = self.edges.get((a, b))
            if edge is not None:
                edge.subedges.append(forward_sub)
                edge.subedges.append(reverse_sub)

        self.traversals[(start_id, end_id)] = Traversal(
            start_id=start_id,
            end_id=end_id,
            node_ids=node_ids,
            mean_score=mean_score,
            closed_tick=self.tick_count,
            subedges=[forward_sub, reverse_sub],
        )

    def audit_edge_influence(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Run the auditor, then integrate every traversal's own path
        quality onto each real edge (an adjacent pair in some traversal's
        node_ids) it passes through.

        Each traversal's mean_score is a length-normalized log-prob for
        its own particular causal path; exp(mean_score) turns that back
        into a (0, 1] path quality. Summing that quality, over every
        traversal that happens to cross a given real edge, is the
        discrete integral this is named for: how much cumulative,
        probability-weighted influence from the whole graph's causal-
        path manifold flows through that one direct connection. "count"
        is how many distinct traversals contributed, so a caller can
        also see the *average* quality of paths through an edge
        (total / count) separately from the *total* volume of influence
        running through it.
        """
        self._run_graph_auditor()
        influence: Dict[Tuple[int, int], Dict[str, float]] = {}
        for traversal in self.traversals.values():
            quality = math.exp(traversal.mean_score)
            node_ids = traversal.node_ids
            for a, b in zip(node_ids, node_ids[1:]):
                entry = influence.setdefault((a, b), {"total": 0.0, "count": 0})
                entry["total"] += quality
                entry["count"] += 1
        return influence

    def _subedge_flow_amount(self, sub: "SubEdge", src: FluxNode, dst: FluxNode) -> float:
        """How much sub would move from src to dst this tick, before
        applying it. Two differentials, added together then gated by this
        subedge's own constriction: the pressure differential (bulk
        transport) and the volume differential (soluble/concentration
        transport, pulling toward equalizing). Never negative -- a
        subedge only ever pushes in its own declared direction; that's
        what the *other* subedge (going the other way) is for.
        """
        pressure_diff = src.pressure - dst.pressure
        volume_diff = src.volume - dst.volume
        return sub.constriction * max(0.0, pressure_diff + volume_diff)

    def _drain_mixture(self, node: FluxNode, amount: float) -> Dict[str, float]:
        """Remove ``amount`` of solution from node and return the removed
        mix, keyed by component name ("solvent" plus each soluble).
        Everything leaves in proportion to its current share of the
        node's total volume -- a drawn-off sample of the solution, not a
        selective extraction.
        """
        total = node.volume
        if total <= 0.0 or amount <= 0.0:
            return {}
        frac = min(amount, total) / total
        mix: Dict[str, float] = {}
        if node.solvent > 0.0:
            moved = node.solvent * frac
            node.solvent -= moved
            mix["solvent"] = moved
        for name, held in list(node.solubles.items()):
            moved = held * frac
            if moved == 0.0:
                continue
            node.solubles[name] = held - moved
            mix[name] = mix.get(name, 0.0) + moved
        return mix

    def _deposit_mixture(self, node: FluxNode, mix: Dict[str, float], scale: float = 1.0) -> None:
        """Add a drained mix (see _drain_mixture) into node, optionally
        scaled -- the pump uses scale to split one pooled intake across
        several outflow subedges."""
        for name, amount in mix.items():
            amount *= scale
            if name == "solvent":
                node.solvent += amount
            else:
                node.solubles[name] = node.solubles.get(name, 0.0) + amount

    def _move_volume(self, src: FluxNode, dst: FluxNode, amount: float) -> None:
        """Move ``amount`` of solution from src to dst -- solvent and
        solubles together, in their current proportions; transport
        carries the mixture, it doesn't convert anything."""
        self._deposit_mixture(dst, self._drain_mixture(src, amount))

    def _node_slice(self, nid: int, node: FluxNode, network_roots: Dict[int, int]) -> Optional[Tuple[str, str]]:
        """(own slice, opposite slice) for a node -- "<group>:<direction>"
        naming, identical to the frontend's wedgeKey ("main:forward",
        "net:7:backward", ...). None for the anchor, which sits at the
        center and belongs to no slice."""
        if node.direction is None:
            return None
        root = network_roots.get(nid)
        group = "main" if root is None else f"net:{root}"
        own_dir = "forward" if node.direction is Direction.FORWARD else "backward"
        opp_dir = "backward" if own_dir == "forward" else "forward"
        return f"{group}:{own_dir}", f"{group}:{opp_dir}"

    def _field_value(self, slice_name: str, field_name: str, radius: float) -> float:
        """Evaluate one scalar gradient layer of one slice at a radius.
        Per-slice overrides (config.slice_scalar_fields) win over the
        shared defaults (config.scalar_fields); a layer defined in
        neither is simply absent (0.0 everywhere)."""
        per_slice = self.config.slice_scalar_fields.get(slice_name, {})
        fn = per_slice.get(field_name, self.config.scalar_fields.get(field_name))
        return float(fn(radius)) if fn is not None else 0.0

    def _exchange_humidity(self) -> None:
        """Every node's default-on exchange with its slice's ambient
        humidity field: humidity outside, solvent once transformed inside
        the node. Exchange follows the differential between the ambient
        humidity at the node's own radius and the solvent it already
        holds, scaled by the node's own humidity_exchange openness --
        intake when the air is wetter than the node, outward (back to the
        field, which is ambient and doesn't accumulate) when the node is
        wetter than the air.

        Dissolved solubles add real osmotic pull: concentration (the
        solutes' share of the node's total volume, 0..1 -- same scale as
        humidity, so no arbitrary coefficient) enters the differential on
        the intake side, exactly the way solutes lower water potential in
        the physical version. A solute-rich node draws harder on ambient
        humidity and equilibrates wetter than ambient; a solute-free node
        equalizes to ambient exactly as before.

        When the heart's pressure is low (anchor pressure below
        starvation_floor, the codebase's existing definition of "low
        pressure"), every node is biased to intake only: no outward
        exchange, the system holds onto its water. The mirror signal:
        when the heart is overpressured (anchor pressure above
        config.overpressure_ceiling, 0 = off), every pore is biased to
        expel only -- all water must go, no intake anywhere. If both
        somehow apply (a ceiling configured below the floor), low wins:
        survival over purge.
        """
        anchor_pressure = self.nodes[self.anchor_id].pressure
        heart_low = anchor_pressure < self.config.starvation_floor
        ceiling = self.config.overpressure_ceiling
        heart_high = ceiling > 0.0 and anchor_pressure > ceiling
        network_roots = self.orthogonal_network_roots()
        for nid, node in self.nodes.items():
            if node.burned or node.humidity_exchange <= 0.0:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            own_slice, _ = slices
            root = network_roots.get(nid)
            radius = node.depth if root is None else node.depth - self.nodes[root].depth
            ambient = self._field_value(own_slice, "humidity", float(radius))
            total = node.volume
            concentration = sum(node.solubles.values()) / total if total > 0.0 else 0.0
            flow = node.humidity_exchange * (ambient - node.solvent + concentration)
            if heart_low:
                flow = max(0.0, flow)
            elif heart_high:
                flow = min(0.0, flow)
            node.solvent += flow

    def _ingest_from_rings(self) -> None:
        """Ring dispensing: each pie slice's ring hands its own unique
        soluble to nearby nodes, at a rate set by how close each node
        currently sits to its ring *in the client's own interface sim*.
        That nearness doesn't exist back here at all -- the backend has
        no node positions -- so it arrives through the external-physics
        exchange (absorb_external_physics, domain "client", key
        "ring_proximity": node_id -> 0..1 with 1 = right on the ring).
        No client watching means no payload means no dispensing: ring
        ingestion is genuinely part of the gamified simulation, not a
        backend-only process wearing its name.

        One unique substance per pie slice (see _node_slice for naming).
        Ingesting your own slice's soluble consumes an equal amount of
        the *opposite* slice's soluble (same group, other direction)
        already held in the node: amount = proximity * opposite_held, so
        nearness and available payment are the only throttles -- no
        arbitrary rate constant.
        """
        client = self.external_physics.get("client", {})
        proximity: Dict[int, float] = client.get("ring_proximity") or {}
        if not proximity:
            return
        network_roots = self.orthogonal_network_roots()
        for nid, node in self.nodes.items():
            if node.burned:
                continue
            near = proximity.get(nid)
            if not near:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            own_name, opposite_name = slices
            opposite_held = node.solubles.get(opposite_name, 0.0)
            if opposite_held <= 0.0:
                continue
            amount = min(1.0, max(0.0, float(near))) * opposite_held
            node.solubles[opposite_name] = opposite_held - amount
            node.solubles[own_name] = node.solubles.get(own_name, 0.0) + amount

    def _transport_subedges(self) -> None:
        """One pass of volume transport through every traversal's subedges.

        Iterates traversals, not edges: a subedge is only open at its own
        traversal's start_id/end_id (see SubEdge.is_open_at), which for a
        multi-hop traversal usually aren't the same as any one real Edge's
        own from_id/to_id -- Edge.subedges is a query index (the "hull"),
        not the right iteration path, and walking it would see each
        subedge once per real edge it crosses instead of once overall.

        Skips any subedge open at the anchor -- those are the four-chamber
        pump's job (_pump_anchor), not ordinary node-to-node transport;
        the anchor's own volume is never touched by a regular subedge.
        """
        anchor_id = self.anchor_id
        for traversal in self.traversals.values():
            start_id, end_id = traversal.start_id, traversal.end_id
            if anchor_id in (start_id, end_id):
                continue
            start = self.nodes.get(start_id)
            end = self.nodes.get(end_id)
            if start is None or end is None or start.burned or end.burned:
                continue
            for sub in traversal.subedges:
                if not (sub.is_open_at(start_id) and sub.is_open_at(end_id)):
                    continue
                src, dst = (start, end) if sub.direction == "forward" else (end, start)
                flow = self._subedge_flow_amount(sub, src, dst)
                if flow > 0.0:
                    self._move_volume(src, dst, flow)

    def _pump_anchor(self) -> None:
        """One beat of the heart (see Heart): sort every anchor-open
        subedge into its slice's in- or out-chamber routes, measure all
        intake from the same pre-beat snapshot, land it in the
        in-chambers, then let the current script phase's valves,
        contractions, and osmotic rebalancing decide what actually moves
        and where it exits. The anchor node's own solubles are never
        touched -- the heart's chambers hold, the anchor doesn't.
        """
        anchor_id = self.anchor_id
        network_roots = self.orthogonal_network_roots()
        inflow_routes: Dict[str, List[Tuple[SubEdge, FluxNode]]] = {}
        outflow_routes: Dict[str, List[Tuple[SubEdge, FluxNode]]] = {}
        for traversal in self.traversals.values():
            start_id, end_id = traversal.start_id, traversal.end_id
            if anchor_id not in (start_id, end_id):
                continue
            for sub in traversal.subedges:
                if not sub.is_open_at(anchor_id):
                    continue
                src_id, dst_id = (start_id, end_id) if sub.direction == "forward" else (end_id, start_id)
                is_inflow = dst_id == anchor_id
                far_id = src_id if is_inflow else dst_id
                far = self.nodes.get(far_id)
                if far is None or far.burned:
                    continue
                slices = self._node_slice(far_id, far, network_roots)
                if slices is None:
                    continue
                slice_name = slices[0]
                routes = inflow_routes if is_inflow else outflow_routes
                routes.setdefault(slice_name, []).append((sub, far))

        anchor = self.nodes[anchor_id]
        heart = self.heart

        # All intake measured from the same pre-beat snapshot before any
        # of it is drained -- the chambers act in one beat, not as
        # sequential transfers that would see each other's deliveries as
        # fresh differentials and immediately pump them straight back.
        intake_plans = {
            slice_name: self._chamber_intake_plan(routes, anchor)
            for slice_name, routes in inflow_routes.items()
        }
        for slice_name, plan in intake_plans.items():
            pooled = self._drain_plan(plan)
            chamber = heart.chamber(slice_name, "in")
            for name, amount in pooled.items():
                chamber[name] = chamber.get(name, 0.0) + amount
        for slice_name in outflow_routes:
            heart.chamber(slice_name, "out")  # chamber exists even before anything reaches it

        phase = heart.script[heart.phase_index % len(heart.script)] if heart.script else None
        heart._run_hooks("pre")
        if phase is not None:
            valves = heart._resolve_valves(phase)
            contractions = heart._resolve_contractions(phase)
            heart._osmotic_rebalance(valves)
            heart._squeeze_in_chambers(valves, contractions)
            self._squeeze_out_chambers(phase, contractions, outflow_routes)
            heart.phase_index = (heart.phase_index + 1) % max(len(heart.script), 1)
        heart._run_hooks("post")

    def _squeeze_out_chambers(
        self,
        phase: HeartPhase,
        contractions: Dict[str, float],
        outflow_routes: Dict[str, List[Tuple["SubEdge", FluxNode]]],
    ) -> None:
        """Contracting out-chambers expel to the network: their own
        slice's outflow subedges (exit_scope "own"), or every outflow
        subedge regardless of slice ("all", the total-supply exit) --
        split by subedge constriction either way. An out-chamber with no
        route (its slice currently has no outflow subedges, or exit
        scope resolves to an empty set) resists: contents stay held for
        a later beat.
        """
        all_routes = [pair for routes in outflow_routes.values() for pair in routes]
        for key, fraction in contractions.items():
            if not key.endswith("|out") or fraction <= 0.0:
                continue
            out_slice = key[: -len("|out")]
            mix = self.heart.chambers.get(key)
            if not mix:
                continue
            routes = all_routes if phase.exit_scope == "all" else outflow_routes.get(out_slice, [])
            if not routes:
                continue
            total_constriction = sum(sub.constriction for sub, _ in routes)
            if total_constriction <= 0.0:
                continue
            expelled: Dict[str, float] = {}
            for name, amount in list(mix.items()):
                moved = amount * min(1.0, fraction)
                if moved == 0.0:
                    continue
                mix[name] = amount - moved
                expelled[name] = moved
            if not expelled:
                continue
            for sub, far in routes:
                self._deposit_mixture(far, expelled, scale=sub.constriction / total_constriction)

    def _chamber_intake_plan(
        self, inflow: List[Tuple["SubEdge", FluxNode]], anchor: FluxNode
    ) -> List[Tuple[FluxNode, float]]:
        """What one chamber's inflow subedges would draw this beat --
        computed exactly like ordinary transport (anchor standing in as
        the destination), purely read-only, so every chamber can be
        measured from the same snapshot before anything is drained.
        """
        return [
            (far, amount)
            for sub, far in inflow
            if (amount := self._subedge_flow_amount(sub, far, anchor)) > 0.0
        ]

    def _drain_plan(self, plan: List[Tuple[FluxNode, float]]) -> Dict[str, float]:
        """Apply a measured intake plan, pooling every drained mix
        (solvent and solubles together) into one chamber-load."""
        pooled: Dict[str, float] = {}
        for far, amount in plan:
            for name, moved in self._drain_mixture(far, amount).items():
                pooled[name] = pooled.get(name, 0.0) + moved
        return pooled

