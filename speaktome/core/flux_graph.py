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
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensors import AbstractTensor
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional tensor backend
    torch = None  # type: ignore
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
    traversal_key: Tuple[int, ...]  # endpoint pair, or full path when endpoints have >1 causal route
    direction: str  # "forward" | "reverse"
    constriction: float = 1.0

    def is_open_at(self, node_id: int) -> bool:
        """Whether this subedge currently allows volume transfer at node_id.

        Closed everywhere except its own two endpoints.
        """
        return node_id in (self.traversal_key[0], self.traversal_key[-1])


@dataclass
class Edge:
    """One connection between two nodes, as a real object -- not just a
    parent_ids/children_ids pointer pair with conductance recomputed from
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
    # Last tick's fluid flow through this pipe: signed net volume that
    # crossed it (+ = from_id -> to_id, i.e. parent -> child), and the
    # pressure drop across it. Reset and re-accumulated each fluid phase
    # (see FluxGraph._reset_edge_flow / _record_edge_flow); the same fluid
    # passes every cross-section of a pipe, so every edge on a traversal's
    # path records that traversal's whole flow. Display-only.
    flow: float = 0.0
    pressure_drop: float = 0.0
    # Signed material flow: ``solvent`` is water; every other key is one ion.
    component_flows: Dict[str, float] = field(default_factory=dict)


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


@dataclass
class IonReservoir:
    """A seed heart's directional ion store.

    The design storage is one ion unit per token represented by the seed.
    Storage volume is elastic: it begins at that design size and grows
    whenever the held mixture requires more room. The ion gate is
    bidirectional. Its finite per-tick throughput is the fraction
    ``opening_coverage * exchange_probability`` of the correction needed
    at the current boundary; there is no unrelated absolute flow cap.

    Concentration limits come from the moving population mean plus/minus
    one population standard deviation. The semipermeable membrane moves
    solvent only, while the ion gate moves only ``ion_name``.
    """

    ion_name: str
    design_storage: float
    ion_amount: float = 0.0
    solvent: float = 0.0
    opening_coverage: float = 1.0
    exchange_probability: float = 1.0
    membrane_permeability: float = 1.0
    window_size: int = 2
    concentration_window: List[float] = field(default_factory=list)

    @property
    def mixture_volume(self) -> float:
        return self.ion_amount + self.solvent

    @property
    def storage_volume(self) -> float:
        return max(self.design_storage, self.mixture_volume)

    @property
    def concentration(self) -> float:
        volume = self.mixture_volume
        return self.ion_amount / volume if volume > 0.0 else 0.0

    @property
    def fullness(self) -> float:
        volume = self.storage_volume
        return self.mixture_volume / volume if volume > 0.0 else 0.0

    def observe(self, concentration: float) -> None:
        self.concentration_window.append(max(0.0, min(1.0, float(concentration))))
        if len(self.concentration_window) > self.window_size:
            del self.concentration_window[: len(self.concentration_window) - self.window_size]

    def concentration_band(self) -> Tuple[float, float, float]:
        samples = self.concentration_window or [self.concentration]
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        sigma = math.sqrt(variance)
        return max(0.0, mean - sigma), mean, min(1.0, mean + sigma)

    def exchange_with(self, chamber: Dict[str, float]) -> None:
        """Skim/supply one chamber, then exchange solvent through the membrane."""
        volume = sum(chamber.values())
        chamber_ions = chamber.get(self.ion_name, 0.0)
        chamber_concentration = chamber_ions / volume if volume > 0.0 else 0.0
        low, target, high = self.concentration_band()
        gate = max(0.0, min(1.0, self.opening_coverage * self.exchange_probability))

        # The same gate skims excess ions or supplies deficient chambers.
        # An empty new reservoir begins with a zero band, so the first
        # matching chamber supply is excess and establishes its history.
        if volume > 0.0 and target > 0.0:
            chamber_concentration = chamber_ions / volume
            if chamber_concentration > high:
                correction = (
                    (chamber_ions - target * volume) / (1.0 - target)
                    if target < 1.0 else 0.0
                )
                moved = min(chamber_ions, correction * gate)
                chamber[self.ion_name] = chamber_ions - moved
                self.ion_amount += moved
            elif chamber_concentration < low and self.ion_amount > 0.0:
                correction = (
                    (target * volume - chamber_ions) / (1.0 - target)
                    if target < 1.0 else self.ion_amount
                )
                moved = min(self.ion_amount, correction * gate)
                chamber[self.ion_name] = chamber_ions + moved
                self.ion_amount -= moved
        elif volume > 0.0 and chamber_ions > 0.0:
            moved = chamber_ions * gate
            chamber[self.ion_name] = chamber_ions - moved
            self.ion_amount += moved

        # Solvent crosses separately. It expands the physical mixture toward
        # the concentration window's mean without carrying ions through the
        # semipermeable membrane.
        membrane = max(0.0, min(1.0, self.membrane_permeability))
        if target > 0.0 and membrane > 0.0:
            desired_solvent = max(0.0, self.ion_amount / target - self.ion_amount)
            solvent_delta = desired_solvent - self.solvent
            if solvent_delta > 0.0:
                moved = min(chamber.get("solvent", 0.0), solvent_delta * membrane)
                chamber["solvent"] = chamber.get("solvent", 0.0) - moved
                self.solvent += moved
            elif solvent_delta < 0.0:
                moved = min(self.solvent, -solvent_delta * membrane)
                self.solvent -= moved
                chamber["solvent"] = chamber.get("solvent", 0.0) + moved
        self.observe(self.concentration)

    def drain(self) -> Dict[str, float]:
        mixture: Dict[str, float] = {}
        if self.ion_amount:
            mixture[self.ion_name] = self.ion_amount
        if self.solvent:
            mixture["solvent"] = self.solvent
        self.ion_amount = 0.0
        self.solvent = 0.0
        return mixture


@dataclass
class MaterialFactory:
    """A generic synthesis/circulatory role a node may service.

    Nothing assigns factories automatically. A caller declares the input
    recipe, output recipe, throughput, and which fluid domain carries the
    reaction. ``medium`` is ``"circulatory"``, ``"csf"``, or ``"both"``.
    The special output name ``"auxin"`` feeds the node's auxin source;
    every other output remains dissolved in the working mixture.
    """

    name: str
    inputs: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, float] = field(default_factory=dict)
    waste_outputs: Dict[str, float] = field(default_factory=dict)
    medium: str = "circulatory"
    throughput: float = 1.0
    enabled: bool = True


class Heart:
    """The anchor's pump, made explicit: per-slice chambers, a nexus
    valve matrix, seed ion reservoirs, a scripted beat, and an attachment
    API.

    Chambers are keyed "<slice>|in" / "<slice>|out" and hold real
    mixture dicts (same "solvent"-plus-soluble-names form the rest of
    transport uses) between ticks -- the heart is the one place in the
    system allowed to hold volume that belongs to no node; the anchor
    node itself still never holds anything. Chambers appear lazily as
    slices appear and simply sit empty when their slice dies.

    While its pump node is a seed, the heart also owns one reservoir for
    each directional slice. Reservoirs skim and supply the chambers
    through independently parameterized ion gates and adjust solvent
    through a semipermeable membrane. Losing seed ownership full-pumps
    those stores into the corresponding out-chambers.

    The script is a list of HeartPhases advanced one per tick (beat),
    wrapping -- rhythm is data, not code. set_script installs a named
    preset from HEART_SCRIPTS or a custom phase list. attach/detach
    manage HeartHooks (see that docstring for pre/post and the
    privileged "total" scope).
    """

    def __init__(self) -> None:
        self.chambers: Dict[str, Dict[str, float]] = {}
        self.seed_owner_id: Optional[int] = None
        self.reservoirs: Dict[str, IonReservoir] = {}
        self.script_name = "crossover"
        self.script: List[HeartPhase] = list(HEART_SCRIPTS["crossover"])
        self.phase_index = 0
        self.hooks: List[HeartHook] = []
        self.valve_modulator: Optional[Callable[[str, str, float], float]] = None

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

    def configure_seed_reservoirs(
        self,
        owner_id: int,
        ion_names: List[str],
        design_storage: float,
        initially_full: bool,
        opening_coverage: float,
        exchange_probability: float,
        membrane_permeability: float,
        window_size: int,
    ) -> None:
        """Attach token-scaled directional stores to the current seed."""
        if self.seed_owner_id == owner_id and set(self.reservoirs) == set(ion_names):
            return
        self.seed_owner_id = owner_id
        self.reservoirs = {
            ion_name: IonReservoir(
                ion_name=ion_name,
                design_storage=design_storage,
                ion_amount=design_storage if initially_full else 0.0,
                opening_coverage=opening_coverage,
                exchange_probability=exchange_probability,
                membrane_permeability=membrane_permeability,
                window_size=max(1, window_size),
                concentration_window=[1.0] if initially_full else [],
            )
            for ion_name in ion_names
        }

    def exchange_seed_reservoirs(self) -> None:
        """Run every reservoir gate against both matching heart chambers."""
        for ion_name, reservoir in self.reservoirs.items():
            for stage in ("in", "out"):
                reservoir.exchange_with(self.chamber(ion_name, stage))

    def release_seed_reservoirs(self) -> None:
        """Full-pump every store into its matching supply chamber."""
        for ion_name, reservoir in self.reservoirs.items():
            out = self.chamber(ion_name, "out")
            for name, amount in reservoir.drain().items():
                out[name] = out.get(name, 0.0) + amount
        self.seed_owner_id = None

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
            valves = {(s, _flip_slice(s)): 1.0 for s in in_slices if _flip_slice(s) in out_slices}
        elif phase.valves == "all":
            valves = {(i, o): 1.0 for i in in_slices for o in out_slices}
        else:
            valves = dict(phase.valves)
        if self.valve_modulator is not None:
            valves = {
                (in_slice, out_slice): self.valve_modulator(
                    in_slice, out_slice, throttle
                )
                for (in_slice, out_slice), throttle in valves.items()
            }
        return valves

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
    # Canonical causal ownership. A backward beam creates parents and a
    # forward beam creates children, so branching through the seed layer is
    # a DAG: one node may have several simultaneous parents. parent_id is
    # retained as the primary/legacy path for callers that need one path;
    # topology and the auditor use this complete list.
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)
    # Signed position on the one global reading axis:
    # backward-farthest < ... < seed layer (0) < ... < forward-farthest.
    # Parent ownership always points from the lower level to the higher
    # level. depth remains abs(level) for existing radial consumers.
    level: Optional[int] = None
    # Nutrient stress accumulates rather than becoming an instant topology
    # override. A forward-side cell lacking backward ions raises backward
    # (parent-growing) interest; a backward-side cell lacking forward ions
    # raises forward (child-growing) interest.
    backward_growth_interest: float = 0.0
    forward_growth_interest: float = 0.0
    # Which seed-layer causal focus this node currently belongs to. A node
    # created at level 0 becomes its own center; ordinary growth inherits
    # the source node's center.
    center_id: Optional[int] = None
    # World-facing shell. hull_permeability gates all passive exchange;
    # a named pore can further throttle one material without changing the
    # hull. Missing pore entries mean fully open for that material.
    hull_permeability: float = 1.0
    pore_permeabilities: Dict[str, float] = field(default_factory=dict)
    # Optional, declarative service roles. Nodes have no factory by
    # default; later synthesis/circulatory specialization is data.
    factories: List[MaterialFactory] = field(default_factory=list)
    factory_auxin: float = 0.0
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

    def __post_init__(self) -> None:
        if self.parent_id is not None and self.parent_id not in self.parent_ids:
            self.parent_ids.insert(0, self.parent_id)
        if self.parent_id is None and self.parent_ids:
            self.parent_id = self.parent_ids[0]
        if self.level is None:
            if self.direction is Direction.BACKWARD:
                self.level = -self.depth
            elif self.direction is Direction.FORWARD:
                self.level = self.depth
            else:
                self.level = 0

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
        return float(self.level or 0)


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
    # Nutrient-driven cross-growth has its own shapes, independent of the
    # ordinary forward/backward beam controls above. A sprout is forward
    # growth launched from backward tissue; an air root is backward growth
    # launched from forward tissue. Defaults commit one candidate for one
    # round, preventing model beam width from silently becoming heart count.
    sprout_branch_factor: int = 1
    sprout_hot_loop_depth: int = 1
    air_root_branch_factor: int = 1
    air_root_hot_loop_depth: int = 1    # Per-direction candidate selection. "topk" (default): exactly
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
    # Seed-layer environmental exchange. Forward-side chamber solutes leak
    # into the shared background through the level-zero interface. The
    # forward ions needed by roots then cross a second, less-permeable soil
    # boundary before backward cells can take them up through their hulls
    # and named pores.
    level_zero_background_permeability: float = 0.1
    soil_forward_ion_permeability: float = 0.025
    root_soil_uptake_permeability: float = 1.0
    # Named beat preset every region's Heart runs (see HEART_SCRIPTS):
    # "crossover" (default) or "total_mix". Applied to each heart as it's
    # created; individual hearts can still be re-scripted at runtime via
    # graph.hearts[region].set_script.
    heart_script: str = "crossover"
    # Seed-heart reservoir boundaries. Both directional ion gates start
    # completely open, matching the near-instant boundary transport model;
    # their product is still a finite 0..1 fraction of the required
    # concentration correction per tick. The membrane moves solvent only.
    seed_ion_gate_opening_coverage: float = 1.0
    seed_ion_exchange_probability: float = 1.0
    seed_reservoir_membrane_permeability: float = 1.0
    # Material scarcity is continuous rather than binary. A node's need is
    # the fractional shortfall below this required opposite-ion concentration.
    growth_target_ion_concentration: float = 0.1
    # Every cousin heart shares this one graph-global CSF bath. These active
    # defaults make that link real without instant mixing; lymph returns a
    # smaller fraction to the active seed's heart each tick.
    csf_link_rate: float = 0.05
    lymph_return_rate: float = 0.02
    # One conserved rhizome belongs to the single active seed. It pumps
    # non-solvent material out of global CSF, then exudes a slower fraction
    # into soil as salts available for root uptake.
    rhizome_csf_pump_rate: float = 0.1
    rhizome_soil_exudation_rate: float = 0.01
    # Differentiable physiology. The graph's structural decisions remain
    # discrete, but every continuous valve/pore is a sigmoid-constrained
    # torch parameter when the active AbstractTensor backend is PyTorch.
    # A tick rewards the gates supporting the currently best audited
    # traversal and charges every open gate a small resource cost, so the
    # optimum is selective living circulation rather than "everything open".
    physiology_learning_enabled: bool = False
    physiology_learning_rate: float = 0.05
    physiology_resource_cost: float = 0.1
    physiology_initial_opening: float = 0.8
    physiology_traversal_temperature: float = 0.5
    physiology_track_model_gradients: bool = False


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
        # One Heart per region (see Heart) -- "main" beats at the anchor,
        # each "net:<root>" cousin network beats at its own root. Parallel,
        # mostly silent toward each other (probability chains are region-
        # locked, see _run_graph_auditor), created lazily by _heart_for as
        # regions appear. Keyed by region string.
        self.hearts: Dict[str, Heart] = {}
        # The CSF bath: one graph-level mixture every pipe sits in. Hearts
        # exchange with it (the csf_link hook) rather than with each other
        # directly; burned nodes and dissolved networks spill into it;
        # it drains slowly home to the seed heart (lymph return). Dormant
        # until its throttles are dialed above 0 (see FluxGraphConfig).
        self.bath: Dict[str, float] = {}
        # Environmental mixtures outside circulation. Forward material
        # crosses level-zero into background; root-needed forward ions
        # cross again, more slowly, into soil.
        self.background: Dict[str, float] = {}
        self.soil: Dict[str, float] = {}
        self.rhizome: Dict[str, float] = {}
        self.rhizome_owner_id: Optional[int] = None
        # Stable named logits for every continuous valve and pore. Keeping
        # ownership here rather than on mutable dataclasses makes parameters
        # easy to optimize, persist, and enumerate even as topology grows.
        self.physiology_parameters: Dict[str, Any] = {}
        self.physiology_last_loss: Optional[float] = None
        self.physiology_best_traversal: Optional[Tuple[int, ...]] = None
        self.physiology_best_score: Optional[float] = None
        self.physiology_steps: int = 0
        self.physiology_error: Optional[str] = None

        self.nodes: Dict[int, FluxNode] = {}
        # Phase 1 edge scaffold (see Edge's own docstring) -- keyed by
        # (parent_id, child_id) at creation time. Re-rooting moves the focus
        # but never reverses this intrinsic causal ownership.
        self.edges: Dict[Tuple[int, int], Edge] = {}
        # Every causal path (ancestor, descendant) that's been enumerated
        # and scored so far -- see Traversal's own docstring and
        # FluxGraph._run_graph_auditor, the only thing that populates
        # this. Normally keyed by (start_id, end_id); when a DAG contains
        # more than one route between the same endpoints, the full node
        # path is the key so those routes remain distinct.
        self.traversals: Dict[Tuple[int, ...], Traversal] = {}
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
        # Lightweight, aggregate progress is intentionally separate from
        # published_snapshot. The latter must remain a complete graph image;
        # this object may change during a tick and contains no mutable graph
        # structures, so a server can poll it without seeing torn topology.
        self.tick_status: Dict[str, Any] = {}
        self._tick_started_at: Optional[float] = None
        self._status_phase: Optional[str] = None
        self._status_phase_started_at: Optional[float] = None
        self._status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
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

    def set_status_callback(
        self, callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        """Receive aggregate progress updates while a tick is in flight."""
        self._status_callback = callback

    def _emit_status(
        self,
        phase: str,
        detail: str = "",
        current: Optional[int] = None,
        total: Optional[int] = None,
        **extra: Any,
    ) -> None:
        now = time.monotonic()
        if phase != self._status_phase:
            self._status_phase = phase
            self._status_phase_started_at = now
        started = self._tick_started_at
        phase_started = self._status_phase_started_at
        status: Dict[str, Any] = {
            "tick": self.tick_count,
            "phase": phase,
            "detail": detail,
            "current": current,
            "total": total,
            "live_nodes": sum(1 for node in self.nodes.values() if not node.burned),
            "elapsed_seconds": round(now - started, 3) if started is not None else 0.0,
            "phase_elapsed_seconds": (
                round(now - phase_started, 3)
                if phase_started is not None else 0.0
            ),
        }
        status.update(extra)
        self.tick_status = status
        callback = self._status_callback
        if callback is not None:
            try:
                callback(dict(status))
            except Exception:
                # Observability must never be able to stop graph growth.
                pass

    # ------------------------------------------------------------------
    # Differentiable valve and pore physiology
    # ------------------------------------------------------------------
    @staticmethod
    def _gate_logit(opening: float) -> float:
        opening = max(1e-4, min(1.0 - 1e-4, float(opening)))
        return math.log(opening / (1.0 - opening))

    def _physiology_parameter(self, key: str, initial: Optional[float] = None):
        """Return one stable torch logit, creating it lazily."""
        parameter = self.physiology_parameters.get(key)
        if parameter is not None:
            return parameter
        if torch is None:
            return None
        opening = (
            self.config.physiology_initial_opening
            if initial is None else float(initial)
        )
        parameter = torch.nn.Parameter(
            torch.tensor(
                self._gate_logit(opening),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.physiology_parameters[key] = parameter
        return parameter

    def _gate_tensor(self, key: str, initial: Optional[float] = None):
        parameter = self._physiology_parameter(key, initial)
        return torch.sigmoid(parameter) if parameter is not None else None

    def _gate_value(self, key: str, initial: Optional[float] = None) -> float:
        if not self.config.physiology_learning_enabled:
            return 1.0
        opening = self._gate_tensor(key, initial)
        return float(opening.detach().item()) if opening is not None else 1.0

    @staticmethod
    def _traversal_gate_prefix(traversal_key: Tuple[int, ...]) -> str:
        return "traversal:" + ",".join(str(value) for value in traversal_key)

    def _subedge_gate_key(self, sub: SubEdge) -> str:
        return f"{self._traversal_gate_prefix(sub.traversal_key)}:{sub.direction}"

    _SUBEDGE_ARCHETYPE_FEATURES = (
        "bias",
        "pressure_drive",
        "source_water_fraction",
        "destination_water_fraction",
        "volume_drive",
        "source_osmotic_fraction",
        "destination_osmotic_fraction",
        "path_quality",
    )

    @classmethod
    def _subedge_archetype_key(cls, direction: str, feature: str) -> str:
        return f"archetype:subedge:{direction}:{feature}"

    def _ensure_subedge_archetype(self) -> None:
        """Create one shared state-response model for each flow direction.

        Old saves can contain thousands of path-specific traversal logits.
        Their mean opening seeds the two archetype biases once, after which
        those dormant path parameters are discarded.
        """
        if torch is None or not self.config.physiology_learning_enabled:
            return
        legacy = {
            key: parameter
            for key, parameter in self.physiology_parameters.items()
            if key.startswith("traversal:")
        }
        for direction in ("forward", "reverse"):
            direction_legacy = [
                float(torch.sigmoid(parameter).detach().item())
                for key, parameter in legacy.items()
                if key.endswith(f":{direction}")
            ]
            initial = (
                sum(direction_legacy) / len(direction_legacy)
                if direction_legacy
                else self.config.physiology_initial_opening
            )
            for feature in self._SUBEDGE_ARCHETYPE_FEATURES:
                key = self._subedge_archetype_key(direction, feature)
                if key in self.physiology_parameters:
                    continue
                if feature == "bias":
                    self._physiology_parameter(key, initial)
                else:
                    self.physiology_parameters[key] = torch.nn.Parameter(
                        torch.zeros(
                            (), dtype=torch.float32, device=self.device
                        )
                    )
        for key in legacy:
            del self.physiology_parameters[key]

    def _subedge_archetype_openings(
        self,
        start_pressure,
        end_pressure,
        start_volume,
        end_volume,
        start_solvent,
        end_solvent,
        path_score,
    ):
        """Return [traversal, forward/reverse] state-conditioned openings."""
        if torch is None:
            return None
        self._ensure_subedge_archetype()
        eps = 1e-12
        start_water = start_solvent / start_volume.clamp_min(eps)
        end_water = end_solvent / end_volume.clamp_min(eps)
        start_osmotic = (start_volume - start_solvent).clamp_min(0.0) / (
            start_volume.clamp_min(eps)
        )
        end_osmotic = (end_volume - end_solvent).clamp_min(0.0) / (
            end_volume.clamp_min(eps)
        )
        quality = torch.exp(path_score).clamp(0.0, 1.0)
        ones = torch.ones_like(start_volume)
        forward_features = torch.stack(
            (
                ones,
                torch.tanh(start_pressure - end_pressure),
                start_water,
                end_water,
                torch.tanh(start_volume - end_volume),
                start_osmotic,
                end_osmotic,
                quality,
            ),
            dim=1,
        )
        reverse_features = torch.stack(
            (
                ones,
                torch.tanh(end_pressure - start_pressure),
                end_water,
                start_water,
                torch.tanh(end_volume - start_volume),
                end_osmotic,
                start_osmotic,
                quality,
            ),
            dim=1,
        )
        weights = torch.stack(
            [
                torch.stack(
                    [
                        self.physiology_parameters[
                            self._subedge_archetype_key(direction, feature)
                        ]
                        for feature in self._SUBEDGE_ARCHETYPE_FEATURES
                    ]
                )
                for direction in ("forward", "reverse")
            ]
        )
        logits = torch.stack(
            (
                (forward_features * weights[0]).sum(dim=1),
                (reverse_features * weights[1]).sum(dim=1),
            ),
            dim=1,
        )
        return torch.sigmoid(logits)

    @staticmethod
    def _edge_gate_key(parent_id: int, child_id: int, direction: str) -> str:
        return f"edge:{parent_id}:{child_id}:{direction}"

    @staticmethod
    def _node_hull_gate_key(node_id: int) -> str:
        return f"node:{node_id}:hull"

    @staticmethod
    def _node_pore_gate_key(node_id: int, material: str) -> str:
        return f"node:{node_id}:pore:{material}"

    def _learned_subedge_opening(self, sub: SubEdge) -> float:
        if not self.config.physiology_learning_enabled or torch is None:
            return max(0.0, min(1.0, float(sub.constriction)))
        traversal = self.traversals.get(sub.traversal_key)
        if traversal is None:
            return max(0.0, min(1.0, float(sub.constriction)))
        start = self.nodes.get(traversal.start_id)
        end = self.nodes.get(traversal.end_id)
        if start is None or end is None:
            return max(0.0, min(1.0, float(sub.constriction)))
        dtype = torch.float32
        device = self.device
        openings = self._subedge_archetype_openings(
            torch.tensor([start.pressure], dtype=dtype, device=device),
            torch.tensor([end.pressure], dtype=dtype, device=device),
            torch.tensor([start.volume], dtype=dtype, device=device),
            torch.tensor([end.volume], dtype=dtype, device=device),
            torch.tensor([start.solvent], dtype=dtype, device=device),
            torch.tensor([end.solvent], dtype=dtype, device=device),
            torch.tensor([traversal.mean_score], dtype=dtype, device=device),
        )
        column = 0 if sub.direction == "forward" else 1
        opening = float(openings[0, column].detach().item())
        return max(0.0, min(1.0, float(sub.constriction) * opening))

    def _learned_node_permeability(self, node: FluxNode, material: str) -> float:
        hull = max(0.0, min(1.0, float(node.hull_permeability)))
        pore = max(
            0.0,
            min(1.0, float(node.pore_permeabilities.get(material, 1.0))),
        )
        return (
            hull
            * pore
            * self._gate_value(self._node_hull_gate_key(node.id))
            * self._gate_value(self._node_pore_gate_key(node.id, material))
        )

    def _heart_valve_modulator(
        self, region: str, in_slice: str, out_slice: str, throttle: float
    ) -> float:
        key = f"heart:{region}:valve:{in_slice}->{out_slice}"
        return max(0.0, min(1.0, float(throttle) * self._gate_value(key)))

    def _bind_heart_learning(self, region: str, heart: Heart) -> None:
        heart.valve_modulator = (
            lambda in_slice, out_slice, throttle, region=region:
            self._heart_valve_modulator(
                region, in_slice, out_slice, throttle
            )
        )

    def _apply_reservoir_learning(self) -> None:
        if not self.config.physiology_learning_enabled:
            return
        for region, heart in self.hearts.items():
            for name, reservoir in heart.reservoirs.items():
                reservoir.opening_coverage = self._gate_value(
                    f"heart:{region}:reservoir:{name}:coverage"
                )
                reservoir.exchange_probability = self._gate_value(
                    f"heart:{region}:reservoir:{name}:exchange"
                )
                reservoir.membrane_permeability = self._gate_value(
                    f"heart:{region}:reservoir:{name}:membrane"
                )

    def _ensure_physiology_parameters(self) -> None:
        if not self.config.physiology_learning_enabled or torch is None:
            return
        self._ensure_subedge_archetype()
        network_roots = self.orthogonal_network_roots()
        for (parent_id, child_id), edge in self.edges.items():
            if self.nodes[parent_id].burned or self.nodes[child_id].burned:
                continue
            self._physiology_parameter(
                self._edge_gate_key(parent_id, child_id, "forward")
            )
            self._physiology_parameter(
                self._edge_gate_key(parent_id, child_id, "reverse")
            )
        for node_id, node in self.nodes.items():
            if node.burned:
                continue
            self._physiology_parameter(self._node_hull_gate_key(node_id))
            materials = {"solvent", *node.solubles.keys()}
            slices = self._node_slice(node_id, node, network_roots)
            if slices is not None:
                materials.update(slices)
            for material in materials:
                self._physiology_parameter(
                    self._node_pore_gate_key(node_id, material)
                )
        for region, heart in self.hearts.items():
            self._bind_heart_learning(region, heart)
            if heart.script:
                phase = heart.script[heart.phase_index % len(heart.script)]
                for in_slice, out_slice in heart._resolve_valves(phase):
                    self._physiology_parameter(
                        f"heart:{region}:valve:{in_slice}->{out_slice}"
                    )
            for name in heart.reservoirs:
                for gate_name in ("coverage", "exchange", "membrane"):
                    self._physiology_parameter(
                        f"heart:{region}:reservoir:{name}:{gate_name}"
                    )

    def _learn_physiology(self) -> None:
        """Diffuse score-weighted reward over every audited traversal."""
        if not self.config.physiology_learning_enabled:
            return
        if torch is None:
            self.physiology_error = "PyTorch is unavailable"
            return
        live_traversals = [
            (traversal_key, traversal)
            for traversal_key, traversal in self.traversals.items()
            if all(
                node_id in self.nodes and not self.nodes[node_id].burned
                for node_id in traversal.node_ids
            )
        ]
        if not live_traversals:
            return
        self._ensure_physiology_parameters()
        if not self.physiology_parameters:
            return

        best_key, best = max(
            live_traversals, key=lambda item: item[1].mean_score
        )
        heart_support = {
            key for key in self.physiology_parameters
            if key.startswith("heart:main:")
        }
        pore_keys_by_node: Dict[int, List[str]] = {}
        for key in self.physiology_parameters:
            if not key.startswith("node:") or ":pore:" not in key:
                continue
            try:
                node_id = int(key.split(":", 2)[1])
            except (ValueError, IndexError):
                continue
            pore_keys_by_node.setdefault(node_id, []).append(key)

        for parameter in self.physiology_parameters.values():
            parameter.grad = None
        structural_keys = [
            key
            for key in self.physiology_parameters
            if not key.startswith("archetype:")
        ]
        structural_openings = torch.stack(
            [
                torch.sigmoid(self.physiology_parameters[key])
                for key in structural_keys
            ]
        ) if structural_keys else None
        score_tensor = torch.tensor(
            [
                float(traversal.mean_score)
                for _, traversal in live_traversals
            ],
            dtype=torch.float32,
            device=self.device,
        )
        temperature = max(
            1e-4, float(self.config.physiology_traversal_temperature)
        )
        traversal_weights = torch.softmax(
            score_tensor / temperature, dim=0
        )
        starts = [self.nodes[traversal.start_id] for _, traversal in live_traversals]
        ends = [self.nodes[traversal.end_id] for _, traversal in live_traversals]
        archetype_openings = self._subedge_archetype_openings(
            torch.tensor(
                [node.pressure for node in starts],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [node.pressure for node in ends],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [node.volume for node in starts],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [node.volume for node in ends],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [node.solvent for node in starts],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [node.solvent for node in ends],
                dtype=torch.float32,
                device=self.device,
            ),
            score_tensor,
        )
        expected_route_opening = (
            traversal_weights * archetype_openings.mean(dim=1)
        ).sum()

        detached_weights = traversal_weights.detach().cpu().tolist()
        gate_coefficients: Dict[str, float] = {}
        for weight, (traversal_key, traversal) in zip(
            detached_weights, live_traversals
        ):
            supporting = set(heart_support)
            for parent_id, child_id in zip(
                traversal.node_ids, traversal.node_ids[1:]
            ):
                supporting.add(
                    self._edge_gate_key(parent_id, child_id, "forward")
                )
                supporting.add(
                    self._edge_gate_key(parent_id, child_id, "reverse")
                )
            for node_id in traversal.node_ids:
                supporting.add(self._node_hull_gate_key(node_id))
                supporting.update(pore_keys_by_node.get(node_id, ()))
            supporting = {
                key for key in supporting
                if key in self.physiology_parameters
            }
            if supporting:
                share = float(weight) / len(supporting)
                for key in supporting:
                    gate_coefficients[key] = (
                        gate_coefficients.get(key, 0.0) + share
                    )
        expected_structural_opening = (
            torch.stack(
                [
                    torch.sigmoid(self.physiology_parameters[key])
                    * coefficient
                    for key, coefficient in gate_coefficients.items()
                ]
            ).sum()
            if gate_coefficients
            else expected_route_opening.new_zeros(())
        )
        expected_survival = (
            0.5 * expected_route_opening
            + 0.5 * expected_structural_opening
        )
        resource_opening = archetype_openings.mean()
        if structural_openings is not None:
            resource_opening = (
                resource_opening + structural_openings.mean()
            ) * 0.5
        loss = (
            -expected_survival
            + max(0.0, float(self.config.physiology_resource_cost))
            * resource_opening
        )
        loss.backward()
        learning_rate = max(0.0, float(self.config.physiology_learning_rate))
        with torch.no_grad():
            for parameter in self.physiology_parameters.values():
                if parameter.grad is None:
                    continue
                parameter.add_(parameter.grad, alpha=-learning_rate)
                parameter.clamp_(-9.0, 9.0)

        self.physiology_last_loss = float(loss.detach().item())
        self.physiology_best_traversal = tuple(best.node_ids)
        self.physiology_best_score = float(best.mean_score)
        self.physiology_steps += 1
        self.physiology_error = None

    def physiology_state(self) -> Dict[str, Any]:
        if self.config.physiology_learning_enabled:
            self._ensure_subedge_archetype()
        openings = {
            key: float(torch.sigmoid(parameter).detach().item())
            for key, parameter in self.physiology_parameters.items()
            if not key.startswith("archetype:")
        } if torch is not None else {}
        archetype_coefficients = {
            key: float(parameter.detach().item())
            for key, parameter in self.physiology_parameters.items()
            if key.startswith("archetype:")
        } if torch is not None else {}
        by_kind: Dict[str, List[float]] = {}
        for key, opening in openings.items():
            by_kind.setdefault(key.split(":", 1)[0], []).append(opening)
        values = list(openings.values())
        return {
            "enabled": self.config.physiology_learning_enabled,
            "parameter_count": len(self.physiology_parameters),
            "archetype_parameter_count": len(archetype_coefficients),
            "archetypes": {
                "subedge": {
                    "parameter_count": len(archetype_coefficients),
                    "coefficient_l1_mean": (
                        sum(abs(value) for value in archetype_coefficients.values())
                        / len(archetype_coefficients)
                        if archetype_coefficients else None
                    ),
                }
            },
            "opening_mean": sum(values) / len(values) if values else None,
            "opening_min": min(values) if values else None,
            "opening_max": max(values) if values else None,
            "by_kind": {
                kind: {
                    "count": len(kind_values),
                    "mean": sum(kind_values) / len(kind_values),
                }
                for kind, kind_values in by_kind.items()
            },
            "loss": self.physiology_last_loss,
            "best_traversal": list(self.physiology_best_traversal)
            if self.physiology_best_traversal is not None else None,
            "best_score": self.physiology_best_score,
            "steps": self.physiology_steps,
            "error": self.physiology_error,
            "model_gradient_tracking": bool(
                self.config.physiology_track_model_gradients
            ),
        }

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
            level=0,
            center_id=node_id,
            local_evidence=0.0,
            pressure=1.0 + self.config.found_bonus,
            created_tick=0,
            expanded=True,  # the anchor doesn't get "expanded" itself
        )
        self.anchor_id = node_id
        self.rhizome_owner_id = node_id
        self._configure_seed_heart("main", node_id, initially_full=True)
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
        self.rhizome_owner_id = anchor_id
        for node in self.nodes.values():
            if node.center_id is None:
                node.center_id = anchor_id
        self.anchor_tokens = list(anchor_tokens)
        self.anchor_local_evidence = anchor_local_evidence
        self.tick_count = tick_count
        self._next_id = max(nodes.keys()) + 1 if nodes else 0
        # Saved level/direction/depth fields are display metadata derived
        # from the current anchor, not topology.  In particular, a save
        # produced before the signed-level reroot fixes can contain a legal
        # causal graph with stale forward leaves after their ancestor moved
        # backward.  Never trust those cached coordinates: derive them from
        # the causal graph exactly as a live reroot does.  Contradictory
        # topology raises in _recompute_depths_from; no edge is silently
        # pruned to make a damaged save look plausible.
        self._recompute_depths_from(anchor_id)
        self._restore_edges_from_topology()
        self._publish_snapshot()

    def _restore_edges_from_topology(self) -> None:
        """Rehydrate physical Edges for every live saved causal adjacency.

        Node persistence stores canonical parent/child ownership, while
        Edge objects contain runtime channels, subedges, and flow state.
        Older saves did not persist Edge objects themselves.  Reconstructing
        them is therefore mandatory: otherwise the auditor can enumerate a
        visually connected path but fluid and pressure have no physical pipe
        to travel through.

        IDs are allocated monotonically.  For postfix growth the child is the
        newly allocated endpoint; for prefix growth the parent is.  That lets
        us recover formation direction without consulting saved display
        direction, which may legitimately change after rerooting.
        """
        self.edges.clear()
        self.traversals.clear()
        for parent_id, parent in self.nodes.items():
            if parent.burned:
                continue
            for child_id in parent.children_ids:
                child = self.nodes.get(child_id)
                if child is None or child.burned:
                    continue
                if parent_id not in child.parent_ids:
                    raise ValueError(
                        f"saved causal edge {parent_id}->{child_id} is not reciprocal"
                    )
                parent_level = int(parent.level or 0)
                child_level = int(child.level or 0)
                if child_level != parent_level + 1:
                    raise ValueError(
                        f"saved causal edge {parent_id}->{child_id} must advance one "
                        f"signed level ({parent_level}->{child_level})"
                    )
                prefix = parent_id > child_id
                grown_node = parent if prefix else child
                formation = "prefix_beam" if prefix else "postfix_beam"
                base = grown_node.local_value
                self.edges[(parent_id, child_id)] = Edge(
                    from_id=parent_id,
                    to_id=child_id,
                    forward=Channel(conductance=base),
                    reverse=Channel(
                        conductance=base * self.config.return_conductance_scale
                    ),
                    formation=formation,
                    seed_id_at_formation=self.anchor_id,
                    created_tick=max(parent.created_tick, child.created_tick),
                )

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

    def export_fluid_state(self) -> Dict[str, Any]:
        """The graph-level fluid state that lives outside individual nodes
        -- the CSF bath and every region heart's chambers, reservoirs,
        script, and phase -- for persistence. Node-local fluid
        (solvent/solubles) rides along with each node's own serialization,
        not here."""
        return {
            "bath": dict(self.bath),
            "background": dict(self.background),
            "soil": dict(self.soil),
            "rhizome": dict(self.rhizome),
            "rhizome_owner_id": self.rhizome_owner_id,
            "physiology": {
                "logits": {
                    key: float(parameter.detach().item())
                    for key, parameter in self.physiology_parameters.items()
                },
                "loss": self.physiology_last_loss,
                "best_traversal": list(self.physiology_best_traversal)
                if self.physiology_best_traversal is not None else None,
                "best_score": self.physiology_best_score,
                "steps": self.physiology_steps,
            },
            "hearts": {
                region: {
                    "chambers": {k: dict(v) for k, v in heart.chambers.items()},
                    "seed_owner_id": heart.seed_owner_id,
                    "reservoirs": {
                        name: {
                            "ion_name": reservoir.ion_name,
                            "design_storage": reservoir.design_storage,
                            "ion_amount": reservoir.ion_amount,
                            "solvent": reservoir.solvent,
                            "opening_coverage": reservoir.opening_coverage,
                            "exchange_probability": reservoir.exchange_probability,
                            "membrane_permeability": reservoir.membrane_permeability,
                            "window_size": reservoir.window_size,
                            "concentration_window": list(reservoir.concentration_window),
                        }
                        for name, reservoir in heart.reservoirs.items()
                    },
                    "script_name": heart.script_name,
                    "phase_index": heart.phase_index,
                }
                for region, heart in self.hearts.items()
            },
        }

    def import_fluid_state(self, state: Dict[str, Any]) -> None:
        """Restore export_fluid_state output. A heart resumed here gets
        its saved chambers/phase; its script comes from the saved name
        (custom phase lists aren't round-tripped -- they fall back to the
        configured script, since a phase list can hold arbitrary code)."""
        self.bath = dict(state.get("bath", {}))
        self.background = dict(state.get("background", {}))
        self.soil = dict(state.get("soil", {}))
        self.rhizome = dict(state.get("rhizome", {}))
        self.rhizome_owner_id = self.anchor_id
        for region, hstate in state.get("hearts", {}).items():
            heart = self._heart_for(region)
            heart.chambers = {k: dict(v) for k, v in hstate.get("chambers", {}).items()}
            heart.seed_owner_id = hstate.get("seed_owner_id")
            heart.reservoirs = {
                name: IonReservoir(
                    ion_name=rstate.get("ion_name", name),
                    design_storage=float(rstate.get("design_storage", 0.0)),
                    ion_amount=float(rstate.get("ion_amount", 0.0)),
                    solvent=float(rstate.get("solvent", 0.0)),
                    opening_coverage=float(rstate.get("opening_coverage", 1.0)),
                    exchange_probability=float(rstate.get("exchange_probability", 1.0)),
                    membrane_permeability=float(rstate.get("membrane_permeability", 1.0)),
                    window_size=max(1, int(rstate.get("window_size", 1))),
                    concentration_window=list(rstate.get("concentration_window", [])),
                )
                for name, rstate in hstate.get("reservoirs", {}).items()
            }
            name = hstate.get("script_name", self.config.heart_script)
            if name in HEART_SCRIPTS:
                heart.set_script(name)
            heart.phase_index = hstate.get("phase_index", 0)
        physiology = state.get("physiology", {})
        if torch is not None:
            self.physiology_parameters = {
                key: torch.nn.Parameter(
                    torch.tensor(
                        float(value), dtype=torch.float32, device=self.device
                    )
                )
                for key, value in physiology.get("logits", {}).items()
            }
        self.physiology_last_loss = physiology.get("loss")
        best_traversal = physiology.get("best_traversal")
        self.physiology_best_traversal = (
            tuple(int(value) for value in best_traversal)
            if best_traversal is not None else None
        )
        self.physiology_best_score = physiology.get("best_score")
        self.physiology_steps = int(physiology.get("steps", 0))
        # Old saves may contain hearts attached to display roots away from
        # level zero. Reconcile ownership immediately on restore.
        self._reap_dead_hearts(set(self._region_pump_nodes()))

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

    def _live_parents(self, node_id: int) -> List[int]:
        return [p for p in self.nodes[node_id].parent_ids if p in self.nodes and not self.nodes[p].burned]

    def _neighbors(self, node_id: int) -> List[int]:
        return list(dict.fromkeys(self._live_parents(node_id) + self._live_children(node_id)))

    def _outward_neighbors(self, node_id: int) -> List[int]:
        """Live neighbors farther from this node's current seed layer.

        This is radial growth geometry, deliberately separate from causal
        ownership: backward nodes grow outward through causal parents,
        forward nodes through causal children, and a center has both sides.
        """
        node = self.nodes[node_id]
        if (node.level or 0) < 0:
            candidates = self._live_parents(node_id)
        elif (node.level or 0) > 0:
            candidates = self._live_children(node_id)
        else:
            candidates = self._neighbors(node_id)
        return [
            neighbor_id for neighbor_id in candidates
            if abs(self.nodes[neighbor_id].level or 0) > abs(node.level or 0)
        ]

    def _centerward_neighbors(self, node_id: int) -> List[int]:
        """Live neighbors closer to a seed-layer center."""
        node = self.nodes[node_id]
        if (node.level or 0) < 0:
            candidates = self._live_children(node_id)
        elif (node.level or 0) > 0:
            candidates = self._live_parents(node_id)
        else:
            return []
        return [
            neighbor_id for neighbor_id in candidates
            if abs(self.nodes[neighbor_id].level or 0) < abs(node.level or 0)
        ]

    @staticmethod
    def _direction_for_level(level: int) -> Optional[Direction]:
        if level < 0:
            return Direction.BACKWARD
        if level > 0:
            return Direction.FORWARD
        return None

    def _connect(self, parent_id: int, child_id: int) -> None:
        """Install one causal edge under the global backward->forward rule."""
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        parent_level = int(parent.level or 0)
        child_level = int(child.level or 0)
        if child_level != parent_level + 1:
            raise ValueError(
                f"causal edge {parent_id}->{child_id} must advance one signed level "
                f"({parent_level}->{child_level})"
            )
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
        if parent_id not in child.parent_ids:
            child.parent_ids.append(parent_id)
        if child.parent_id is None:
            child.parent_id = parent_id

    def _path_to_center(self, node_id: int) -> List[int]:
        """One best-effort monotone path from node to its seed-layer center.

        The graph may have several causal parents. This helper is only for
        legacy single-sequence scoring/display callers; the auditor walks
        every causal route. It follows the primary available monotone edge
        toward level 0 and returns node..center inclusive.
        """
        path = [node_id]
        seen = {node_id}
        cur_id = node_id
        while self.nodes[cur_id].level != 0:
            cur = self.nodes[cur_id]
            candidates = (
                self._live_children(cur_id) if (cur.level or 0) < 0
                else self._live_parents(cur_id)
            )
            candidates = [
                nid for nid in candidates
                if abs(self.nodes[nid].level or 0) < abs(cur.level or 0)
            ]
            if not candidates:
                break
            next_id = max(candidates, key=lambda nid: self.nodes[nid].path_mean)
            if next_id in seen:
                break
            seen.add(next_id)
            path.append(next_id)
            cur_id = next_id
        return path

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
        direction = self.nodes[node_id].direction
        path = self._path_to_center(node_id)
        if path and self.nodes[path[-1]].level == 0:
            path = path[:-1]
        if direction is Direction.FORWARD:
            path.reverse()
        spans = [self.nodes[nid].tokens for nid in path]
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
            if node.direction is Direction.BACKWARD:
                if self._live_parents(node_id):
                    continue
            elif self._live_children(node_id):
                continue
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
            if leaf is None:
                continue
            path = self._path_to_center(leaf)
            for nid in path:
                if self.nodes[nid].level == 0:
                    continue
                total += self.nodes[nid].local_evidence
                count += 1
        mean_score = total / count if count > 0 else 0.0
        return tokens, mean_score

    # ------------------------------------------------------------------
    # Tick: pressure update, expansion, starvation
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """Advance one discrete step."""
        self.tick_count += 1
        self._tick_started_at = time.monotonic()
        self._emit_status("settling", "relaxing circuit", 0, self.config.max_relaxation_iterations)
        self._settle_circuit()
        self._emit_status("rerooting", "checking causal focus")
        self._maybe_reroot()
        self._emit_status("digesting", "rolling up path quality")
        self._digest()
        self._emit_status("auxin", "diffusing growth signals")
        self._diffuse_auxin()
        if self.config.graph_auditor_enabled:
            # Local missing-ion stress requests an opposite-direction root sprout.
            self._emit_status("nutrients", "reading local sprout demand")
            self._update_nutrient_growth_interest()
        self._emit_status("expanding", "selecting growth fronts")
        self._expand_top_pressure_nodes()
        self._emit_status("starvation", "reaping disconnected or starved tissue")
        self._starve_and_burn()
        if self.config.graph_auditor_enabled:
            self._emit_status("auditing", "enumerating causal paths")
            self._run_graph_auditor()
            self._emit_status(
                "learning", "diffusing traversal reward through valves and pores"
            )
            self._learn_physiology()
            self._apply_reservoir_learning()
            self._emit_status("humidity", "exchanging solvent and dissolved ions")
            self._exchange_humidity()
            self._emit_status("rings", "ingesting boundary supplies")
            self._ingest_from_rings()
            self._emit_status("transport", "moving fluid through traversals")
            self._transport_subedges()
            self._emit_status("pumping", "beating regional hearts")
            self._pump_hearts()
            self._emit_status("soil", "permeating the level-zero interface")
            self._permeate_background_into_soil()
            self._absorb_soil_by_roots()
            self._emit_status("factories", "running node synthesis")
            self._run_node_factories()
            self._emit_status("nutrients", "reconciling local sprout demand")
            self._update_nutrient_growth_interest(accumulate=False)
        self._publish_snapshot()
        self._emit_status("complete", "tick snapshot published", 1, 1)

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
        """Move the causal focus without reversing causal ownership.

        Parent->child always means backward->forward. Changing which point
        is the probability manifold's current center can rebase signed
        levels and display directions, but it cannot reverse an Edge.
        """
        old_anchor_id = self.anchor_id
        if new_root_id == old_anchor_id:
            return
        old_anchor_tokens = self.anchor_tokens
        old_anchor_local_evidence = self.anchor_local_evidence
        # The old seed heart does not travel with the focus. Re-rooting
        # dumps every chamber and reservoir into the shared CSF bath.
        self._spill_heart_to_csf("main")
        self.nodes[old_anchor_id].tokens = old_anchor_tokens
        self.nodes[old_anchor_id].local_evidence = old_anchor_local_evidence
        new_root = self.nodes[new_root_id]
        self.anchor_tokens, self.anchor_local_evidence = self._reset_to_anchor_invariants(new_root)
        self.anchor_id = new_root_id
        self.rhizome_owner_id = new_root_id
        new_root.center_id = new_root_id
        self._recompute_depths_from(new_root_id)
        # Signed levels changed: immediately purge any distributed heart
        # whose owner is no longer on the new level-zero seed tier.
        self._reap_dead_hearts(set(self._region_pump_nodes()))
        self._configure_seed_heart("main", new_root_id, initially_full=False)

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
        node.depth = 0
        node.level = 0
        node.local_evidence = 0.0
        node.cumulative_evidence = 0.0
        node.low_pressure_ticks = 0
        node.expanded = True
        return captured_tokens, captured_local_evidence

    def _recompute_depths_from(self, root_id: int) -> None:
        """Rebase signed levels around a new focus without changing edges."""
        levels = {root_id: 0}
        queue = [root_id]
        while queue:
            cur_id = queue.pop(0)
            cur_level = levels[cur_id]
            for parent_id in self._live_parents(cur_id):
                proposed = cur_level - 1
                if parent_id not in levels:
                    levels[parent_id] = proposed
                    queue.append(parent_id)
                elif levels[parent_id] != proposed:
                    raise ValueError(
                        f"inconsistent causal levels at edge {parent_id}->{cur_id}: "
                        f"{levels[parent_id]} versus required {proposed}"
                    )
            for child_id in self._live_children(cur_id):
                proposed = cur_level + 1
                if child_id not in levels:
                    levels[child_id] = proposed
                    queue.append(child_id)
                elif levels[child_id] != proposed:
                    raise ValueError(
                        f"inconsistent causal levels at edge {cur_id}->{child_id}: "
                        f"{levels[child_id]} versus required {proposed}"
                    )
        unreachable = [
            node_id for node_id, node in self.nodes.items()
            if not node.burned and node_id not in levels
        ]
        if unreachable:
            raise ValueError(
                "live saved topology is disconnected from the active seed: "
                + ", ".join(str(node_id) for node_id in sorted(unreachable))
            )
        for node_id, level in levels.items():
            node = self.nodes[node_id]
            node.level = level
            node.depth = abs(level)
            node.direction = self._direction_for_level(level)
        self.nodes[root_id].cumulative_evidence = 0.0

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
        explicit_cousins = {
            nid for nid, node in self.nodes.items()
            if not node.burned
            and node.center_id is not None
            and node.center_id != self.anchor_id
        }
        orthogonal: set = set(explicit_cousins)
        anchor = self.nodes[self.anchor_id]
        stack = self._neighbors(anchor.id)
        visited = {anchor.id}
        while stack:
            node_id = stack.pop()
            if node_id in visited or self.nodes[node_id].burned:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            for neighbor_id in self._neighbors(node_id):
                if neighbor_id in visited:
                    continue
                neighbor = self.nodes[neighbor_id]
                if node_id in orthogonal or (
                    node.direction is not None
                    and neighbor.direction is not None
                    and neighbor.direction != node.direction
                ):
                    orthogonal.add(neighbor_id)
                stack.append(neighbor_id)
        return orthogonal

    def orthogonal_network_roots(
        self, orthogonal: Optional["set[int]"] = None
    ) -> Dict[int, int]:
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
        explicit = {
            nid: node.center_id
            for nid, node in self.nodes.items()
            if not node.burned
            and node.center_id is not None
            and node.center_id != self.anchor_id
        }
        if orthogonal is None:
            orthogonal = self.orthogonal_node_ids()
        roots: Dict[int, int] = {}
        for node_id in sorted(orthogonal, key=lambda nid: self.nodes[nid].depth):
            if node_id in explicit:
                roots[node_id] = explicit[node_id]
                continue
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
        key = (
            (neighbor_id, node_id)
            if neighbor_id in self.nodes[node_id].parent_ids
            else (node_id, neighbor_id)
        )
        edge = self.edges.get(key)
        if edge is not None:
            token_node_id = edge.from_id if edge.formation == "prefix_beam" else edge.to_id
            return self.nodes[token_node_id].local_value
        # Compatibility for hand-built test graphs that predate Edge.
        if neighbor_id in self.nodes[node_id].parent_ids:
            return self.nodes[node_id].local_value
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
        parent_id, child_id = (
            (neighbor_id, node_id)
            if neighbor_id in self.nodes[node_id].parent_ids
            else (node_id, neighbor_id)
        )
        if neighbor_id in self.nodes[node_id].parent_ids:
            valve = self._gate_value(
                self._edge_gate_key(parent_id, child_id, "forward")
            )
            return base * valve
        valve = self._gate_value(
            self._edge_gate_key(parent_id, child_id, "reverse")
        )
        return base * self.config.return_conductance_scale * valve

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
        if torch is not None:
            live_nodes = [node for node in self.nodes.values() if not node.burned]
            if not live_nodes:
                return 0
            node_index = {
                node.id: index for index, node in enumerate(live_nodes)
            }
            receiving: List[int] = []
            supplying: List[int] = []
            conductance_values: List[float] = []
            for node in live_nodes:
                for neighbor_id in self._neighbors(node.id):
                    neighbor_index = node_index.get(neighbor_id)
                    if neighbor_index is None:
                        continue
                    receiving.append(node_index[node.id])
                    supplying.append(neighbor_index)
                    conductance_values.append(
                        self._directional_conductance(node.id, neighbor_id)
                    )

            device = self.device
            dtype = torch.float64
            pressure = torch.tensor(
                [float(node.pressure) for node in live_nodes],
                dtype=dtype,
                device=device,
            )
            intrinsic = torch.tensor(
                [
                    float(
                        shared_intrinsic.get(
                            node.id, node.local_value + cfg.found_bonus
                        )
                    )
                    for node in live_nodes
                ],
                dtype=dtype,
                device=device,
            )
            static_adjustment = torch.tensor(
                [
                    (
                        (
                            cfg.balance_weight
                            * max(
                                0.0,
                                (
                                    backward_reach - forward_reach
                                    if node.direction is Direction.FORWARD
                                    else forward_reach - backward_reach
                                ),
                            )
                            if cfg.balance_weight
                            and node.direction is not None
                            else 0.0
                        )
                        - cfg.head_pressure_coefficient * abs(node.height)
                    )
                    for node in live_nodes
                ],
                dtype=dtype,
                device=device,
            )
            population_cost = 0.0
            if cfg.population_target and len(live_nodes) > cfg.population_target:
                population_cost = (
                    len(live_nodes) - cfg.population_target
                ) / cfg.population_target
            fixed = torch.tensor(
                [
                    node.id == self.anchor_id and not cfg.anchor_can_decay
                    for node in live_nodes
                ],
                dtype=torch.bool,
                device=device,
            )
            if receiving:
                receiving_idx = torch.tensor(
                    receiving, dtype=torch.long, device=device
                )
                supplying_idx = torch.tensor(
                    supplying, dtype=torch.long, device=device
                )
                conductance = torch.tensor(
                    conductance_values, dtype=dtype, device=device
                )
                conductance_sum = torch.zeros(
                    len(live_nodes), dtype=dtype, device=device
                )
                conductance_sum.index_add_(0, receiving_idx, conductance)
            else:
                receiving_idx = supplying_idx = torch.empty(
                    0, dtype=torch.long, device=device
                )
                conductance = torch.empty(0, dtype=dtype, device=device)
                conductance_sum = torch.zeros(
                    len(live_nodes), dtype=dtype, device=device
                )

            for i in range(cfg.max_relaxation_iterations):
                weighted_sum = torch.zeros_like(pressure)
                if receiving:
                    weighted_sum.index_add_(
                        0,
                        receiving_idx,
                        conductance
                        * pressure.index_select(0, supplying_idx),
                    )
                inflow = weighted_sum / (1.0 + conductance_sum)
                candidate = torch.clamp_min(
                    intrinsic
                    + cfg.damping * inflow
                    + static_adjustment
                    - cfg.decay_rate * pressure
                    - population_cost,
                    0.0,
                )
                new_pressure = torch.where(fixed, pressure, candidate)
                delta = float(
                    torch.max(torch.abs(new_pressure - pressure)).item()
                )
                pressure = new_pressure
                self._emit_status(
                    "settling",
                    f"pressure delta {delta:.3g}",
                    i + 1,
                    cfg.max_relaxation_iterations,
                )
                if delta < cfg.relaxation_tolerance:
                    rows = pressure.detach().cpu().tolist()
                    for node, value in zip(live_nodes, rows):
                        node.pressure = value
                    return i + 1
            rows = pressure.detach().cpu().tolist()
            for node, value in zip(live_nodes, rows):
                node.pressure = value
            return cfg.max_relaxation_iterations

        for i in range(cfg.max_relaxation_iterations):
            delta = self._update_pressures(shared_intrinsic, forward_reach, backward_reach)
            self._emit_status(
                "settling",
                f"pressure delta {delta:.3g}",
                i + 1,
                cfg.max_relaxation_iterations,
            )
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
            outward_rollups = [
                self.nodes[neighbor_id].rollup_mean
                for neighbor_id in self._outward_neighbors(node.id)
            ]
            node.rollup_mean = max([node.path_mean] + outward_rollups)

        anchor = self.nodes[self.anchor_id]
        outward_rollups = [
            self.nodes[neighbor_id].rollup_mean
            for neighbor_id in self._outward_neighbors(self.anchor_id)
        ]
        anchor.rollup_mean = max(outward_rollups) if outward_rollups else 0.0

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
            outward = self._outward_neighbors(node.id)
            own_source = (
                math.exp(node.path_mean) if not outward else 0.0
            ) + max(0.0, node.factory_auxin)
            outward_best = max(
                (self.nodes[nid].subtree_auxin for nid in outward), default=0.0
            )
            node.subtree_auxin = max(own_source, outward_best * cfg.auxin_decay)

        for node in sorted(live_nodes, key=lambda n: n.depth):
            centerward = self._centerward_neighbors(node.id)
            siblings = {
                sibling_id
                for centerward_id in centerward
                for sibling_id in self._outward_neighbors(centerward_id)
                if sibling_id != node.id
            }
            siblings_best = max((self.nodes[s].subtree_auxin for s in siblings), default=0.0)
            centerward_ambient = max(
                (
                    0.0 if nid == self.anchor_id else self.nodes[nid].auxin_level
                    for nid in centerward
                ),
                default=0.0,
            )
            node.auxin_level = cfg.auxin_decay * max(siblings_best, centerward_ambient)

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
        centerward = self._centerward_neighbors(node.id)
        neighborhood_rollup = max(
            (self.nodes[nid].rollup_mean for nid in centerward), default=None
        )
        neighborhood_bonus = (
            cfg.rollup_weight * math.exp(neighborhood_rollup)
            if neighborhood_rollup is not None else 0.0
        )
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
        # Forward tips have no live children farther forward. Backward tips
        # have no live parents farther backward. Ownership follows the one
        # global backward->forward axis, not radial distance from a center.
        return [
            n for n in self.nodes.values()
            if not n.burned
            and n.direction is not None
            and (
                (n.direction is Direction.FORWARD and not self._live_children(n.id))
                or (n.direction is Direction.BACKWARD and not self._live_parents(n.id))
            )
        ]

    def _update_nutrient_growth_interest(self, accumulate: bool = True) -> None:
        """Accumulate cross-growth interest from continuous ion scarcity.

        Each node measures the concentration of its required opposite-side
        ion against ``growth_target_ion_concentration``. Its fractional
        shortfall (0..1), not a binary present/absent flag, is the demand
        added to sprout or air-root priority. Adequate supply clears demand.
        """
        target = max(1e-12, float(self.config.growth_target_ion_concentration))
        network_roots = self.orthogonal_network_roots()
        for nid, node in self.nodes.items():
            if node.burned or node.direction is None:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            _, needed_ion = slices
            concentration = (
                node.solubles.get(needed_ion, 0.0) / node.volume
                if node.volume > 0.0 else 0.0
            )
            scarcity = max(0.0, min(1.0, (target - concentration) / target))
            interest_name = (
                "backward_growth_interest"
                if node.direction is Direction.FORWARD
                else "forward_growth_interest"
            )
            if scarcity <= 0.0:
                setattr(node, interest_name, 0.0)
            elif accumulate:
                setattr(node, interest_name, getattr(node, interest_name) + scarcity)
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

        The anchor's two direct causal connections are its circulatory
        sides: live parents attach the backward side and live children
        attach the forward side. A missing side preempts ordinary frontier
        competition and is grown immediately. This covers both a total
        stall and the more urgent partial failure where one side remains
        healthy enough to keep winning the normal compute competition while
        the other side is absent.
        """
        missing_forward_side = not self._live_children(self.anchor_id)
        missing_backward_side = not self._live_parents(self.anchor_id)
        if missing_forward_side or missing_backward_side:
            if missing_forward_side:
                self._expand_forward(self.anchor_id)
            if missing_backward_side:
                self._expand_backward(self.anchor_id)
            return

        forward_reach = self._direction_reach(Direction.FORWARD)
        backward_reach = self._direction_reach(Direction.BACKWARD)

        def priority(n: FluxNode) -> float:
            return self._expansion_priority(n, forward_reach, backward_reach)

        # (node, nutrient-driven): ordinary tips continue away from the
        # center; nutrient stress contributes a competing center-seeking
        # action in the opposite growth direction. Both spend the same
        # finite compute budget. Any tissue can launch the cross-growth it
        # needs; sprout/air-root width and depth determine its committed shape.
        expandable = self._expandable_nodes()
        forward_pool = [
            (n, False) for n in expandable if n.direction is Direction.FORWARD
        ]
        backward_pool = [
            (n, False) for n in expandable if n.direction is Direction.BACKWARD
        ]
        forward_pool.extend(
            (n, True) for n in self.nodes.values()
            if not n.burned
            and n.direction is Direction.BACKWARD
            and n.forward_growth_interest > 0.0
        )
        backward_pool.extend(
            (n, True) for n in self.nodes.values()
            if not n.burned
            and n.direction is Direction.FORWARD
            and n.backward_growth_interest > 0.0
        )

        def action_priority(action) -> float:
            node, nutrient_driven = action
            stress = 0.0
            if nutrient_driven:
                stress = (
                    node.forward_growth_interest
                    if node.direction is Direction.BACKWARD
                    else node.backward_growth_interest
                )
            return priority(node) + stress

        forward_pool.sort(key=action_priority, reverse=True)
        backward_pool.sort(key=action_priority, reverse=True)

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
                    key=action_priority, reverse=True,
                )
                for action in remaining[:leftover_budget]:
                    node, nutrient_driven = action
                    growth_direction = (
                        Direction.FORWARD
                        if (not nutrient_driven and node.direction is Direction.FORWARD)
                        or (nutrient_driven and node.direction is Direction.BACKWARD)
                        else Direction.BACKWARD
                    )
                    if growth_direction is Direction.FORWARD:
                        forward_selected.append(action)
                    else:
                        backward_selected.append(action)

        if forward_selected:
            ordinary = [n for n, nutrient in forward_selected if not nutrient]
            stressed = [n for n, nutrient in forward_selected if nutrient]
            if ordinary:
                self._expand_batch_hot_loop(
                    ordinary, self._direction_hot_loop_depth(Direction.FORWARD)
                )
            for node in stressed:
                self._expand_nutrient_hot_loop(node, Direction.FORWARD)
        if backward_selected:
            ordinary = [n for n, nutrient in backward_selected if not nutrient]
            stressed = [n for n, nutrient in backward_selected if nutrient]
            if ordinary:
                self._expand_batch_hot_loop(
                    ordinary, self._direction_hot_loop_depth(Direction.BACKWARD)
                )
            for node in stressed:
                self._expand_nutrient_hot_loop(node, Direction.BACKWARD)

    def _expand_nutrient_hot_loop(
        self, source: FluxNode, growth_direction: Direction
    ) -> None:
        """Grow one configured sprout or air root from any needy node.

        Forward cross-growth from backward tissue is a sprout; backward
        cross-growth from forward tissue is an air root. Their width/depth
        controls are deliberately separate from ordinary forward/backward
        beam controls. Every round grows from exactly the nodes committed by
        the previous round, so width**depth is explicit configuration rather
        than an accidental multiplication by the model's ordinary beam.
        """
        if growth_direction is Direction.FORWARD:
            width = max(1, int(self.config.sprout_branch_factor))
            depth = max(1, int(self.config.sprout_hot_loop_depth))
        else:
            width = max(1, int(self.config.air_root_branch_factor))
            depth = max(1, int(self.config.air_root_hot_loop_depth))

        frontier = [source]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                first_new_id = self._next_id
                if growth_direction is Direction.FORWARD:
                    self._expand_forward(node.id, branch_limit=width)
                else:
                    self._expand_backward(node.id, branch_limit=width)
                next_frontier.extend(
                    self.nodes[node_id]
                    for node_id in range(first_new_id, self._next_id)
                    if node_id in self.nodes and not self.nodes[node_id].burned
                )
            frontier = next_frontier
            if not frontier:
                break
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
        rounds = max(1, depth)
        for round_index in range(rounds):
            if not frontier:
                return
            self._emit_status(
                "expanding",
                f"hot-loop round {round_index + 1}; {len(frontier)} growth fronts",
                round_index + 1,
                rounds,
                frontier_nodes=len(frontier),
            )
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
        """Burn one node, then reap every live component cut off from seed.

        Connectivity is undirected here on purpose. Causal ownership always
        runs backward->forward, but heart access runs through the physical
        graph in either direction. A causal-child cascade therefore reaped
        the correct side of a forward branch while leaving farther-back
        causal parents orphaned on a backward branch. Reachability from the
        active seed is the direction-independent definition we actually need.
        """
        if node_id not in self.nodes or self.nodes[node_id].burned:
            return

        def detach(cur_id: int) -> None:
            cur = self.nodes[cur_id]
            for parent_id in list(cur.parent_ids):
                parent = self.nodes.get(parent_id)
                if parent is not None and cur_id in parent.children_ids:
                    parent.children_ids.remove(cur_id)
            for child_id in list(cur.children_ids):
                child = self.nodes.get(child_id)
                if child is None:
                    continue
                child.parent_ids = [pid for pid in child.parent_ids if pid != cur_id]
                if child.parent_id == cur_id:
                    child.parent_id = child.parent_ids[0] if child.parent_ids else None
            cur.parent_ids = []
            cur.parent_id = None
            cur.children_ids = []

        def mark_burned(cur_id: int) -> None:
            cur = self.nodes[cur_id]
            cur.burned = True
            if cur.solvent:
                self.bath["solvent"] = self.bath.get("solvent", 0.0) + cur.solvent
                cur.solvent = 0.0
            for name, amount in cur.solubles.items():
                if amount:
                    self.bath[name] = self.bath.get(name, 0.0) + amount
            cur.solubles = {}
            if self.config.verbose:
                print(f"  [burn] node {cur_id} (tokens={cur.tokens}, dir={cur.direction}) starved out")
            detach(cur_id)

        mark_burned(node_id)

        reachable: "set[int]" = set()
        if (
            self.anchor_id in self.nodes
            and not self.nodes[self.anchor_id].burned
        ):
            stack = [self.anchor_id]
            while stack:
                cur_id = stack.pop()
                if cur_id in reachable:
                    continue
                reachable.add(cur_id)
                stack.extend(
                    neighbor_id for neighbor_id in self._neighbors(cur_id)
                    if neighbor_id not in reachable
                )

        orphaned = [
            nid for nid, node in self.nodes.items()
            if not node.burned and nid not in reachable
        ]
        for orphan_id in orphaned:
            mark_burned(orphan_id)

        newly_burned = [node_id] + orphaned
        self._prune_traversals_touching(newly_burned)
        burned_set = set(newly_burned)
        for edge_key in [
            key for key in self.edges
            if key[0] in burned_set or key[1] in burned_set
        ]:
            self.edges.pop(edge_key, None)
        self._rehome_nodes_from_dead_centers(burned_set)

    def _rehome_nodes_from_dead_centers(self, burned_ids: "set[int]") -> None:
        """Attach surviving nodes from a dead center to a reachable live center."""
        affected = [
            node for node in self.nodes.values()
            if not node.burned and node.center_id in burned_ids
        ]
        if not affected:
            return
        live_centers = {
            nid for nid, node in self.nodes.items()
            if not node.burned
            and (nid == self.anchor_id or node.center_id == nid)
        }
        if not live_centers:
            return
        for node in affected:
            queue = [node.id]
            seen = {node.id}
            replacement = None
            while queue and replacement is None:
                next_queue: List[int] = []
                for cur_id in queue:
                    if cur_id in live_centers:
                        replacement = cur_id
                        break
                    for neighbor_id in self._neighbors(cur_id):
                        if neighbor_id not in seen:
                            seen.add(neighbor_id)
                            next_queue.append(neighbor_id)
                queue = next_queue
            node.center_id = replacement if replacement is not None else self.anchor_id

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
        dead_keys = {
            key for key, traversal in self.traversals.items()
            if any(nid in burned_set for nid in traversal.node_ids)
        }
        self._drop_traversals(dead_keys)

    def _drop_traversals(self, dead_keys: "set[Tuple[int, ...]]") -> None:
        """Remove a set of traversals and strip their SubEdges from every
        real Edge's hull. Each traversal's own two SubEdges (see
        _record_traversal) sit in every Edge along its old path -- same
        unbounded-growth risk as self.traversals itself, so they go too."""
        if not dead_keys:
            return
        for key in dead_keys:
            self.traversals.pop(key, None)
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
        set_tracking = getattr(
            self.model_wrapper, "set_gradient_tracking", None
        )
        previous_tracking = getattr(
            self.model_wrapper, "track_gradients", False
        )

        def invoke():
            if callable(set_tracking):
                set_tracking(
                    self.config.physiology_track_model_gradients
                )
            try:
                return fn()
            finally:
                if callable(set_tracking):
                    set_tracking(previous_tracking)

        try:
            return invoke()
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            self._fallback_to_cpu()
            return invoke()

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
        chunks = self._plan_expand_chunks(all_row_lens, vocab_size)
        for chunk_index, (start, end) in enumerate(chunks):
            self._emit_status(
                "model_inference",
                f"scoring rows {start + 1}-{end} of {len(rows)}",
                chunk_index + 1,
                len(chunks),
                rows_complete=start,
                rows_total=len(rows),
            )
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
            del logits, log_probs, outputs, batch_tokens

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
            del logits, log_probs, outputs, batch_tokens

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

    def _expand_forward(
        self, node_id: int, branch_limit: Optional[int] = None
    ) -> None:
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

        branch_factor = (
            self._direction_branch_factor(Direction.FORWARD)
            if branch_limit is None
            else max(1, int(branch_limit))
        )
        scores, indices = self.choice_policy.choose(last_logits, k=branch_factor)
        spans = [[i] for i in indices.tolist()[0]]
        self._attach_children(node_id, Direction.FORWARD, scores.tolist()[0], spans)

    def _expand_backward(
        self, node_id: int, branch_limit: Optional[int] = None
    ) -> None:
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
        branch_factor = (
            self._direction_branch_factor(Direction.BACKWARD)
            if branch_limit is None
            else max(1, int(branch_limit))
        )
        top_scores, top_idx = AbstractTensor.topk(
            raw_scores, k=branch_factor, dim=0
        )
        candidate_spans = [[int(pool[i].item())] for i in top_idx.tolist()]
        self._attach_children(node_id, Direction.BACKWARD, top_scores.tolist(), candidate_spans)

    def _attach_children(
        self, parent_id: int, direction: Direction, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        """Attach model growth under the one global causal ownership rule.

        Forward growth creates children. Backward growth creates parents;
        the historical method name remains as a compatibility entry point
        for callers while the topology no longer mislabels prefix growth.
        """
        if direction is Direction.BACKWARD:
            self._attach_backward_parents(parent_id, scores, token_spans)
        else:
            self._attach_forward_children(parent_id, scores, token_spans)

    def _growth_pressure(self, token_key: Tuple[int, ...], score: float, live_token_spans) -> float:
        cfg = self.config
        if live_token_spans is not None and token_key in live_token_spans:
            return 0.0
        return cfg.found_bonus + math.exp(float(score))

    def _live_token_spans(self):
        if not self.config.shared_token_pressure_enabled:
            return None
        return {tuple(n.tokens) for n in self.nodes.values() if not n.burned and n.tokens}

    def _new_growth_node(
        self,
        source: FluxNode,
        level: int,
        score: float,
        tokens: List[int],
        live_token_spans,
    ) -> FluxNode:
        node_id = self._alloc_id()
        token_key = tuple(int(t) for t in tokens)
        direction = self._direction_for_level(level)
        center_id = node_id if level == 0 else source.center_id
        cumulative = source.cumulative_evidence + float(score)
        depth = abs(level)
        node = FluxNode(
            id=node_id,
            tokens=list(token_key),
            direction=direction,
            parent_id=None,
            parent_ids=[],
            depth=depth,
            level=level,
            center_id=center_id,
            local_evidence=float(score),
            pressure=self._growth_pressure(token_key, score, live_token_spans),
            created_tick=self.tick_count,
            cumulative_evidence=cumulative,
            rollup_mean=cumulative / max(depth, 1),
        )
        self.nodes[node_id] = node
        if level == 0 and node_id != self.anchor_id:
            self._configure_seed_heart(f"net:{node_id}", node_id, initially_full=False)
        if live_token_spans is not None:
            live_token_spans.add(token_key)
        return node

    def _record_growth_edge(
        self, parent_id: int, child_id: int, grown_node: FluxNode, formation: str
    ) -> None:
        self._connect(parent_id, child_id)
        base = grown_node.local_value
        self.edges[(parent_id, child_id)] = Edge(
            from_id=parent_id,
            to_id=child_id,
            forward=Channel(conductance=base),
            reverse=Channel(conductance=base * self.config.return_conductance_scale),
            formation=formation,
            seed_id_at_formation=self.anchor_id,
            created_tick=self.tick_count,
        )

    def _attach_forward_children(
        self, source_id: int, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        source = self.nodes[source_id]
        live_token_spans = self._live_token_spans()
        for score, tokens in zip(scores, token_spans):
            child = self._new_growth_node(
                source, int(source.level or 0) + 1, score, tokens, live_token_spans
            )
            self._record_growth_edge(source_id, child.id, child, "postfix_beam")

    def _attach_backward_parents(
        self, source_id: int, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        source = self.nodes[source_id]
        live_token_spans = self._live_token_spans()
        for score, tokens in zip(scores, token_spans):
            parent = self._new_growth_node(
                source, int(source.level or 0) - 1, score, tokens, live_token_spans
            )
            self._record_growth_edge(parent.id, source_id, parent, "prefix_beam")

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

        Goes everywhere: the walk up follows the whole real parent chain
        regardless of region. Traversals may span a cousin boundary, and
        that's deliberately allowed -- a cousin heart cross-linking into
        another region is something the system is meant to be robust to,
        not something the auditor forbids. Every heart simply works on
        whatever supply actually reaches its own pump node.
        """
        live_ids = [nid for nid, n in self.nodes.items() if not n.burned]
        live_set = set(live_ids)
        total = len(live_ids)
        for end_index, end_id in enumerate(live_ids):
            if end_index == 0 or end_index + 1 == total or end_index % max(1, total // 25) == 0:
                self._emit_status(
                    "auditing",
                    f"{len(self.traversals)} traversals recorded",
                    end_index + 1,
                    total,
                    traversals=len(self.traversals),
                )
            # Stack entries are paths in reverse causal order: end..ancestor.
            stack = [([end_id], parent_id) for parent_id in self._live_parents(end_id)]
            while stack:
                reverse_path, cur_id = stack.pop()
                if cur_id not in live_set or cur_id in reverse_path:
                    continue
                extended = reverse_path + [cur_id]
                node_ids = list(reversed(extended))
                self._record_traversal_path(node_ids)
                for parent_id in self._live_parents(cur_id):
                    stack.append((extended, parent_id))

    def _record_traversal(self, start_id: int, end_id: int) -> None:
        """Compatibility helper: record the primary causal route."""
        node_ids: List[int] = []
        cur_id = end_id
        while cur_id != start_id:
            node_ids.append(cur_id)
            cur_id = self.nodes[cur_id].parent_id
        node_ids.append(start_id)
        node_ids.reverse()  # start -> end, tree order along the real chain
        self._record_traversal_path(node_ids)

    def _record_traversal_path(self, node_ids: List[int]) -> None:
        start_id, end_id = node_ids[0], node_ids[-1]
        endpoint_key: Tuple[int, ...] = (start_id, end_id)
        existing = self.traversals.get(endpoint_key)
        if existing is not None and existing.node_ids == node_ids:
            return
        key = endpoint_key if existing is None else tuple(node_ids)
        if key in self.traversals:
            return

        scores = []
        for a, b in zip(node_ids, node_ids[1:]):
            edge = self.edges.get((a, b))
            if edge is None:
                scores.append(self.nodes[b].local_evidence)
            else:
                token_id = a if edge.formation == "prefix_beam" else b
                scores.append(self.nodes[token_id].local_evidence)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Every Traversal adds two SubEdges -- one going one way, one
        # going the other -- held by every real Edge its path crosses.
        forward_sub = SubEdge(traversal_key=key, direction="forward")
        reverse_sub = SubEdge(traversal_key=key, direction="reverse")
        for a, b in zip(node_ids, node_ids[1:]):
            edge = self.edges.get((a, b))
            if edge is not None:
                edge.subedges.append(forward_sub)
                edge.subedges.append(reverse_sub)

        self.traversals[key] = Traversal(
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
        return self._learned_subedge_opening(sub) * max(
            0.0, pressure_diff + volume_diff
        )

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

    def _move_volume(self, src: FluxNode, dst: FluxNode, amount: float) -> Dict[str, float]:
        """Move ``amount`` of solution from src to dst -- solvent and
        solubles together, in their current proportions; transport
        carries the mixture, it doesn't convert anything."""
        mixture = self._drain_mixture(src, amount)
        self._deposit_mixture(dst, mixture)
        return mixture

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
        """Exchange solvent and dissolved ambient materials through pores.

        ``humidity`` remains the ambient water amount. Every other scalar
        field in the slice is a dissolved ambient material. Their total
        osmoles lower ambient water activity; node-local osmoles increase
        its solvent target. Material fields themselves diffuse toward
        their ambient values through material-specific pores, so humidity
        can carry ions rather than merely changing a scalar water term.

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
            center_id = node.center_id if node.center_id in self.nodes else self.anchor_id
            center_level = self.nodes[center_id].level or 0
            radius = abs((node.level or 0) - center_level)
            field_names = set(self.config.scalar_fields)
            field_names.update(self.config.slice_scalar_fields.get(own_slice, {}))
            ambient_humidity = max(
                0.0, self._field_value(own_slice, "humidity", float(radius))
            )
            ambient_solutes = {
                name: max(0.0, self._field_value(own_slice, name, float(radius)))
                for name in field_names
                if name != "humidity"
            }
            ambient_osmoles = sum(ambient_solutes.values())
            ambient_activity = (
                ambient_humidity / (ambient_humidity + ambient_osmoles)
                if ambient_humidity + ambient_osmoles > 0.0 else 0.0
            )
            node_osmoles = sum(max(0.0, amount) for amount in node.solubles.values())
            target_solvent = ambient_humidity + node_osmoles * ambient_activity
            shell = self._learned_node_permeability(node, "solvent")
            flow = node.humidity_exchange * shell * (target_solvent - node.solvent)
            if heart_low:
                flow = max(0.0, flow)
            elif heart_high:
                flow = min(0.0, flow)
            node.solvent = max(0.0, node.solvent + flow)

            for name, ambient_amount in ambient_solutes.items():
                permeability = self._learned_node_permeability(node, name)
                delta = (
                    node.humidity_exchange
                    * permeability
                    * (ambient_amount - node.solubles.get(name, 0.0))
                )
                node.solubles[name] = max(
                    0.0, node.solubles.get(name, 0.0) + delta
                )

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

    def _transport_subedges_scalar(self) -> None:
        """One pass of volume transport through every traversal's subedges.

        Iterates traversals, not edges: a subedge is only open at its own
        traversal's start_id/end_id (see SubEdge.is_open_at), which for a
        multi-hop traversal usually aren't the same as any one real Edge's
        own from_id/to_id -- Edge.subedges is a query index (the "hull"),
        not the right iteration path, and walking it would see each
        subedge once per real edge it crosses instead of once overall.

        Skips any traversal that touches a region's pump node (the anchor
        or a cousin network's root) -- those are that region heart's job
        (_pump_hearts), not ordinary node-to-node transport; a pump node's
        own volume is never touched by a regular subedge.
        """
        self._reset_edge_flow()
        network_roots = self.orthogonal_network_roots()
        pump_ids = set(self._region_pump_nodes(network_roots).values())
        traversals = list(self.traversals.values())
        total = len(traversals)
        for traversal_index, traversal in enumerate(traversals):
            if traversal_index == 0 or traversal_index + 1 == total or traversal_index % max(1, total // 25) == 0:
                self._emit_status(
                    "transport",
                    "moving traversal fluid",
                    traversal_index + 1,
                    total,
                )
            start_id, end_id = traversal.start_id, traversal.end_id
            if start_id in pump_ids or end_id in pump_ids:
                continue
            start = self.nodes.get(start_id)
            end = self.nodes.get(end_id)
            if start is None or end is None or start.burned or end.burned:
                continue
            # Named ions diffuse down their own concentration gradients in
            # addition to riding whatever bulk solution pressure moves. Plan
            # both directions from one pre-transfer snapshot so two ions can
            # counterflow without sequential-order oscillation.
            osmotic_forward, osmotic_reverse = self._osmotic_ion_transfer(
                start, end, traversal.subedges
            )
            if osmotic_forward:
                self._record_edge_flow(
                    traversal.node_ids,
                    downward=True,
                    amount=0.0,
                    mixture=osmotic_forward,
                )
            if osmotic_reverse:
                self._record_edge_flow(
                    traversal.node_ids,
                    downward=False,
                    amount=0.0,
                    mixture=osmotic_reverse,
                )
            for sub in traversal.subedges:
                if not (sub.is_open_at(start_id) and sub.is_open_at(end_id)):
                    continue
                src, dst = (start, end) if sub.direction == "forward" else (end, start)
                flow = self._subedge_flow_amount(sub, src, dst)
                if flow > 0.0:
                    mixture = self._move_volume(src, dst, flow)
                    # Same fluid crosses every edge on the path: stamp each
                    # one, signed + when flow runs parent->child (start is
                    # the shallower endpoint, so a forward subedge whose src
                    # is start pushes downward).
                    self._record_edge_flow(
                        traversal.node_ids, downward=(src is start), amount=flow, mixture=mixture
                    )

    def _transport_subedges(self) -> None:
        """Move all non-pump traversal fluid in two batched tensor passes.

        Ion diffusion is planned from one endpoint snapshot and reduced by
        source node/species before it is applied.  Bulk solution transport is
        then planned from the post-osmosis snapshot and reduced by source
        node.  The reductions prevent the many overlapping traversals from
        overdrawing a shared endpoint without making transport depend on
        Python traversal order.

        Python is used only to marshal the graph's sparse dictionaries into a
        dense node/component matrix and to publish the reduced result back to
        the object model.  Pressure/volume differences, per-ion equilibrium,
        availability limiting, mixture movement, and path-flow accumulation
        are tensor operations.
        """
        if torch is None:
            self._transport_subedges_scalar()
            return

        self._reset_edge_flow()
        network_roots = self.orthogonal_network_roots()
        pump_ids = set(self._region_pump_nodes(network_roots).values())
        live_nodes = [node for node in self.nodes.values() if not node.burned]
        node_index = {node.id: index for index, node in enumerate(live_nodes)}
        edge_items = list(self.edges.items())
        edge_index = {key: index for index, (key, _) in enumerate(edge_items)}
        component_names = ["solvent"] + sorted(
            {
                name
                for node in live_nodes
                for name, amount in node.solubles.items()
                if amount != 0.0
            }
        )
        component_index = {
            name: index for index, name in enumerate(component_names)
        }

        starts: List[int] = []
        ends: List[int] = []
        forward_constrictions: List[float] = []
        reverse_constrictions: List[float] = []
        path_edge_indices: List[int] = []
        path_traversal_indices: List[int] = []
        accepted_traversals: List[Traversal] = []
        traversals = list(self.traversals.values())
        total = len(traversals)
        for traversal_index, traversal in enumerate(traversals):
            if (
                traversal_index == 0
                or traversal_index + 1 == total
                or traversal_index % max(1, total // 25) == 0
            ):
                self._emit_status(
                    "transport",
                    "indexing traversal transport",
                    traversal_index + 1,
                    total,
                )
            start_id, end_id = traversal.start_id, traversal.end_id
            if (
                start_id in pump_ids
                or end_id in pump_ids
                or start_id not in node_index
                or end_id not in node_index
            ):
                continue
            constrictions = {
                sub.direction: max(0.0, min(1.0, float(sub.constriction)))
                for sub in traversal.subedges
                if sub.is_open_at(start_id) and sub.is_open_at(end_id)
            }
            if not constrictions:
                continue
            batch_index = len(accepted_traversals)
            accepted_traversals.append(traversal)
            starts.append(node_index[start_id])
            ends.append(node_index[end_id])
            forward_constrictions.append(
                constrictions.get("forward", 0.0)
            )
            reverse_constrictions.append(
                constrictions.get("reverse", 0.0)
            )
            for a, b in zip(traversal.node_ids, traversal.node_ids[1:]):
                physical_index = edge_index.get((a, b))
                if physical_index is not None:
                    path_edge_indices.append(physical_index)
                    path_traversal_indices.append(batch_index)

        if not accepted_traversals:
            return

        device = self.device
        dtype = torch.float32
        value_rows = []
        for node in live_nodes:
            row = [0.0] * len(component_names)
            row[0] = max(0.0, float(node.solvent))
            for name, amount in node.solubles.items():
                column = component_index.get(name)
                if column is not None:
                    row[column] = max(0.0, float(amount))
            value_rows.append(row)
        values = torch.tensor(value_rows, dtype=dtype, device=device)
        pressures = torch.tensor(
            [float(node.pressure) for node in live_nodes],
            dtype=dtype,
            device=device,
        )
        start_idx = torch.tensor(starts, dtype=torch.long, device=device)
        end_idx = torch.tensor(ends, dtype=torch.long, device=device)
        constriction = torch.stack(
            (
                torch.tensor(
                    forward_constrictions, dtype=dtype, device=device
                ),
                torch.tensor(
                    reverse_constrictions, dtype=dtype, device=device
                ),
            ),
            dim=1,
        )
        if self.config.physiology_learning_enabled:
            start_snapshot = values.index_select(0, start_idx)
            end_snapshot = values.index_select(0, end_idx)
            archetype_opening = self._subedge_archetype_openings(
                pressures.index_select(0, start_idx),
                pressures.index_select(0, end_idx),
                start_snapshot.sum(dim=1),
                end_snapshot.sum(dim=1),
                start_snapshot[:, 0],
                end_snapshot[:, 0],
                torch.tensor(
                    [
                        traversal.mean_score
                        for traversal in accepted_traversals
                    ],
                    dtype=dtype,
                    device=device,
                ),
            ).detach()
            gate = constriction * archetype_opening
        else:
            gate = constriction
        forward_gate = gate[:, 0]
        reverse_gate = gate[:, 1]

        self._emit_status(
            "transport",
            "diffusing ions in one tensor batch",
            1,
            3,
        )
        component_flow = torch.zeros(
            (len(accepted_traversals), len(component_names)),
            dtype=dtype,
            device=device,
        )
        if len(component_names) > 1:
            start_values = values.index_select(0, start_idx)
            end_values = values.index_select(0, end_idx)
            start_volume = start_values.sum(dim=1, keepdim=True)
            end_volume = end_values.sum(dim=1, keepdim=True)
            total_volume = start_volume + end_volume
            solute_total = start_values[:, 1:] + end_values[:, 1:]
            safe_total = total_volume.clamp_min(1e-12)
            start_target = solute_total * (start_volume / safe_total)
            excess = start_values[:, 1:] - start_target
            valid_volume = total_volume > 0.0
            raw_osmotic = torch.where(
                excess > 0.0,
                excess * forward_gate[:, None],
                (-excess) * reverse_gate[:, None],
            ) * valid_volume
            osmotic_source = torch.where(
                excess > 0.0, start_idx[:, None], end_idx[:, None]
            ).expand_as(raw_osmotic)
            osmotic_destination = torch.where(
                excess > 0.0, end_idx[:, None], start_idx[:, None]
            ).expand_as(raw_osmotic)

            source_demand = torch.zeros(
                (len(live_nodes), len(component_names) - 1),
                dtype=dtype,
                device=device,
            )
            source_demand.scatter_add_(0, osmotic_source, raw_osmotic)
            available_solutes = values[:, 1:]
            source_scale = torch.where(
                source_demand > 0.0,
                torch.minimum(
                    torch.ones_like(source_demand),
                    available_solutes / source_demand.clamp_min(1e-12),
                ),
                torch.ones_like(source_demand),
            )
            osmotic_moved = raw_osmotic * torch.gather(
                source_scale, 0, osmotic_source
            )
            osmotic_delta = torch.zeros_like(available_solutes)
            osmotic_delta.scatter_add_(0, osmotic_source, -osmotic_moved)
            osmotic_delta.scatter_add_(
                0, osmotic_destination, osmotic_moved
            )
            values[:, 1:] += osmotic_delta
            component_flow[:, 1:] += torch.where(
                excess > 0.0, osmotic_moved, -osmotic_moved
            )

        self._emit_status(
            "transport",
            "moving bulk solution in one tensor batch",
            2,
            3,
        )
        start_values = values.index_select(0, start_idx)
        end_values = values.index_select(0, end_idx)
        start_volume = start_values.sum(dim=1)
        end_volume = end_values.sum(dim=1)
        drive = (
            pressures.index_select(0, start_idx)
            - pressures.index_select(0, end_idx)
            + start_volume
            - end_volume
        )
        forward = drive >= 0.0
        bulk_source = torch.where(forward, start_idx, end_idx)
        bulk_destination = torch.where(forward, end_idx, start_idx)
        bulk_gate = torch.where(forward, forward_gate, reverse_gate)
        raw_bulk = drive.abs() * bulk_gate
        source_bulk_demand = torch.zeros(
            len(live_nodes), dtype=dtype, device=device
        )
        source_bulk_demand.index_add_(0, bulk_source, raw_bulk)
        node_volume = values.sum(dim=1)
        bulk_scale = torch.where(
            source_bulk_demand > 0.0,
            torch.minimum(
                torch.ones_like(source_bulk_demand),
                node_volume / source_bulk_demand.clamp_min(1e-12),
            ),
            torch.ones_like(source_bulk_demand),
        )
        bulk_moved = raw_bulk * bulk_scale.index_select(0, bulk_source)
        source_values = values.index_select(0, bulk_source)
        source_volume = source_values.sum(dim=1, keepdim=True)
        mixture = (
            source_values
            / source_volume.clamp_min(1e-12)
            * bulk_moved[:, None]
        )
        mixture *= source_volume > 0.0
        values.index_add_(0, bulk_source, -mixture)
        values.index_add_(0, bulk_destination, mixture)
        direction_sign = torch.where(
            forward,
            torch.ones_like(bulk_moved),
            -torch.ones_like(bulk_moved),
        )
        signed_bulk = bulk_moved * direction_sign
        component_flow += mixture * direction_sign[:, None]

        self._emit_status(
            "transport",
            "reducing traversal flow onto physical pipes",
            3,
            3,
        )
        edge_flow = torch.zeros(len(edge_items), dtype=dtype, device=device)
        edge_components = torch.zeros(
            (len(edge_items), len(component_names)),
            dtype=dtype,
            device=device,
        )
        if path_edge_indices:
            physical_idx = torch.tensor(
                path_edge_indices, dtype=torch.long, device=device
            )
            traversal_idx = torch.tensor(
                path_traversal_indices, dtype=torch.long, device=device
            )
            edge_flow.index_add_(
                0, physical_idx, signed_bulk.index_select(0, traversal_idx)
            )
            edge_components.index_add_(
                0,
                physical_idx,
                component_flow.index_select(0, traversal_idx),
            )

        node_rows = values.clamp_min(0.0).detach().cpu().tolist()
        for node, row in zip(live_nodes, node_rows):
            node.solvent = row[0]
            node.solubles = {
                name: row[column]
                for column, name in enumerate(component_names[1:], start=1)
                if row[column] != 0.0
            }
        edge_flow_rows = edge_flow.detach().cpu().tolist()
        edge_component_rows = edge_components.detach().cpu().tolist()
        for (_, edge), flow, row in zip(
            edge_items, edge_flow_rows, edge_component_rows
        ):
            edge.flow = flow
            edge.component_flows = {
                name: row[column]
                for column, name in enumerate(component_names)
                if row[column] != 0.0
            }

    def _osmotic_ion_transfer(
        self,
        start: FluxNode,
        end: FluxNode,
        subedges: List[SubEdge],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Diffuse every named ion independently between two pipe endpoints.

        Concentrations and target amounts are computed from one snapshot of
        both endpoint mixtures. The forward/reverse subedge constrictions
        gate their corresponding direction, while solvent is excluded: it
        continues to move only as bulk current. Returned mixtures contain
        component-only flow telemetry for the real edges along the traversal.
        """
        start_volume = start.volume
        end_volume = end.volume
        total_volume = start_volume + end_volume
        if total_volume <= 0.0:
            return {}, {}

        constriction = {
            sub.direction: self._learned_subedge_opening(sub)
            for sub in subedges
            if sub.is_open_at(start.id) and sub.is_open_at(end.id)
        }
        forward_gate = constriction.get("forward", 0.0)
        reverse_gate = constriction.get("reverse", 0.0)
        forward: Dict[str, float] = {}
        reverse: Dict[str, float] = {}
        plans = []
        for name in set(start.solubles) | set(end.solubles):
            start_amount = max(0.0, start.solubles.get(name, 0.0))
            end_amount = max(0.0, end.solubles.get(name, 0.0))
            target_concentration = (start_amount + end_amount) / total_volume
            start_target = target_concentration * start_volume
            if start_amount > start_target and forward_gate > 0.0:
                moved = min(start_amount, (start_amount - start_target) * forward_gate)
                if moved > 0.0:
                    plans.append((name, moved, start, end, forward))
            elif start_amount < start_target and reverse_gate > 0.0:
                moved = min(end_amount, (start_target - start_amount) * reverse_gate)
                if moved > 0.0:
                    plans.append((name, moved, end, start, reverse))

        for name, moved, source, destination, telemetry in plans:
            source.solubles[name] = source.solubles.get(name, 0.0) - moved
            destination.solubles[name] = destination.solubles.get(name, 0.0) + moved
            telemetry[name] = telemetry.get(name, 0.0) + moved
        return forward, reverse
    def _reset_edge_flow(self) -> None:
        """Zero every edge's flow at the start of the fluid phase, and
        refresh its pressure drop from the current endpoint pressures --
        so a pipe with no flow this tick still shows an honest gradient."""
        for (a, b), edge in self.edges.items():
            edge.flow = 0.0
            edge.component_flows.clear()
            na, nb = self.nodes.get(a), self.nodes.get(b)
            edge.pressure_drop = (na.pressure - nb.pressure) if na and nb else 0.0

    def _record_edge_flow(
        self, node_ids: List[int], downward: bool, amount: float,
        mixture: Optional[Dict[str, float]] = None,
    ) -> None:
        """Accumulate signed flow onto every real edge along a path.
        node_ids is shallow->deep tree order, so consecutive (a, b) is
        (parent, child) and the stored edge key is (a, b). + means
        parent->child."""
        signed = amount if downward else -amount
        for a, b in zip(node_ids, node_ids[1:]):
            edge = self.edges.get((a, b))
            if edge is not None:
                edge.flow += signed
                for name, component_amount in (mixture or {}).items():
                    edge.component_flows[name] = edge.component_flows.get(name, 0.0) + (
                        component_amount if downward else -component_amount
                    )

    @staticmethod
    def _is_forward_ion(name: str) -> bool:
        return name.endswith(":forward")

    def _permeate_heart_forward_to_background(
        self, heart: Heart
    ) -> None:
        """Exchange forward level-zero chamber solutes with background.

        The interface is permeable in both directions. Forward chambers can
        seed the environment, while material already outside always retains
        a route back into live circulation instead of becoming a one-way
        orphan store.
        """
        rate = max(0.0, min(1.0, self.config.level_zero_background_permeability))
        if rate <= 0.0:
            return
        for key, mixture in heart.chambers.items():
            slice_name = key.rsplit("|", 1)[0]
            if not slice_name.endswith(":forward"):
                continue
            for name in set(mixture) | set(self.background):
                if name == "solvent":
                    continue
                delta = rate * (
                    mixture.get(name, 0.0) - self.background.get(name, 0.0)
                )
                if delta > 0.0:
                    moved = min(delta, mixture.get(name, 0.0))
                    mixture[name] = mixture.get(name, 0.0) - moved
                    self.background[name] = self.background.get(name, 0.0) + moved
                elif delta < 0.0:
                    moved = min(-delta, self.background.get(name, 0.0))
                    self.background[name] = self.background.get(name, 0.0) - moved
                    mixture[name] = mixture.get(name, 0.0) + moved

    def _permeate_background_into_soil(self) -> None:
        """Move root-needed forward ions across the reduced soil boundary."""
        rate = max(0.0, min(1.0, self.config.soil_forward_ion_permeability))
        if rate <= 0.0:
            return
        for name, amount in list(self.background.items()):
            if not self._is_forward_ion(name) or amount <= 0.0:
                continue
            moved = amount * rate
            self.background[name] = amount - moved
            self.soil[name] = self.soil.get(name, 0.0) + moved

    def _absorb_soil_by_roots(self) -> None:
        """Let backward cells passively take their required forward ion."""
        base_rate = max(0.0, min(1.0, self.config.root_soil_uptake_permeability))
        if base_rate <= 0.0 or not self.soil:
            return
        network_roots = self.orthogonal_network_roots()
        roots_by_ion: Dict[str, List[Tuple[FluxNode, float]]] = {}
        for nid, node in self.nodes.items():
            if node.burned or node.direction is not Direction.BACKWARD:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            _, needed_ion = slices
            permeability = self._learned_node_permeability(
                node, needed_ion
            )
            if permeability > 0.0:
                roots_by_ion.setdefault(needed_ion, []).append((node, permeability))

        for ion_name, roots in roots_by_ion.items():
            available = self.soil.get(ion_name, 0.0)
            total_weight = sum(weight for _, weight in roots)
            if available <= 0.0 or total_weight <= 0.0:
                continue
            moved = available * base_rate
            self.soil[ion_name] = available - moved
            for node, weight in roots:
                share = moved * weight / total_weight
                node.solubles[ion_name] = node.solubles.get(ion_name, 0.0) + share

    @staticmethod
    def _run_factory_reaction(
        node: FluxNode,
        factory: MaterialFactory,
        mixture: Dict[str, float],
        throughput: float,
        waste_sink: Optional[Dict[str, float]] = None,
    ) -> float:
        """Run up to ``throughput`` recipe batches in one fluid mixture."""
        requirements = {
            name: amount for name, amount in factory.inputs.items() if amount > 0.0
        }
        if not requirements or throughput <= 0.0:
            return 0.0
        batches = throughput
        for name, required in requirements.items():
            available = mixture.get(name, 0.0)
            batches = min(batches, available / required)
        if batches <= 0.0:
            return 0.0
        for name, required in requirements.items():
            mixture[name] = mixture.get(name, 0.0) - required * batches
        for name, produced in factory.outputs.items():
            amount = produced * batches
            if name == "auxin":
                node.factory_auxin += amount
            else:
                mixture[name] = mixture.get(name, 0.0) + amount
        if waste_sink is not None:
            for name, produced in factory.waste_outputs.items():
                amount = produced * batches
                if amount:
                    waste_sink[name] = waste_sink.get(name, 0.0) + amount
        return batches

    def _run_node_factories(self) -> None:
        """Execute configured node roles against circulation and/or CSF."""
        for node in self.nodes.values():
            node.factory_auxin = 0.0
            if node.burned:
                continue
            circulation = dict(node.solubles)
            circulation["solvent"] = node.solvent
            changed_circulation = False
            for factory in node.factories:
                if not factory.enabled or factory.throughput <= 0.0:
                    continue
                medium = factory.medium.lower()
                remaining = factory.throughput
                if medium in ("circulatory", "both"):
                    used = self._run_factory_reaction(
                        node, factory, circulation, remaining, self.bath
                    )
                    remaining -= used
                    changed_circulation = changed_circulation or used > 0.0
                if medium in ("csf", "both") and remaining > 0.0:
                    self._run_factory_reaction(
                        node, factory, self.bath, remaining, self.bath
                    )
                elif medium not in ("circulatory", "csf", "both"):
                    raise ValueError(
                        f"factory {factory.name!r} has unknown medium {factory.medium!r}"
                    )
            if changed_circulation:
                node.solvent = circulation.pop("solvent", 0.0)
                node.solubles = {
                    name: amount for name, amount in circulation.items() if amount
                }

    def _heart_for(self, region: str) -> Heart:
        """The Heart for a region, created lazily the first time its
        network appears. Each new heart gets the configured beat script
        and the csf_link hook -- the first real consumer of the Heart
        attachment API: a post-beat, total-scope hook exchanging every
        chamber with the graph's CSF bath at csf_link_rate, so hearts
        connect to one another *through* the shared bath rather than
        directly."""
        heart = self.hearts.get(region)
        if heart is None:
            heart = Heart()
            heart.set_script(self.config.heart_script)
            heart.attach("csf_link", when="post", scope="total", fn=self._csf_link_hook)
            self.hearts[region] = heart
        self._bind_heart_learning(region, heart)
        return heart

    def _configure_seed_heart(
        self, region: str, pump_id: int, initially_full: bool
    ) -> Heart:
        """Make ``pump_id`` the owner of one region's seed reservoirs.

        Storage and the moving-window length both follow the represented
        token count. The original anchor's represented tokens live in
        anchor_tokens while it is a seed; cousin seeds retain their own
        token span until promoted.
        """
        heart = self._heart_for(region)
        if heart.seed_owner_id is not None and heart.seed_owner_id != pump_id:
            self._spill_heart_to_csf(region)
            heart = self._heart_for(region)
        represented_tokens = (
            self.anchor_tokens
            if pump_id == self.anchor_id
            else self.nodes[pump_id].tokens
        )
        token_count = max(1, len(represented_tokens))
        heart.configure_seed_reservoirs(
            owner_id=pump_id,
            ion_names=[f"{region}:forward", f"{region}:backward"],
            design_storage=float(token_count),
            initially_full=initially_full,
            opening_coverage=self.config.seed_ion_gate_opening_coverage,
            exchange_probability=self.config.seed_ion_exchange_probability,
            membrane_permeability=self.config.seed_reservoir_membrane_permeability,
            window_size=token_count,
        )
        return heart

    def _csf_link_hook(self, chambers: Dict[str, Dict[str, float]]) -> None:
        """Exchange every chamber's contents with the CSF bath toward
        equalizing concentration, throttled by config.csf_link_rate.
        Dormant (a no-op) while the rate is 0."""
        rate = self.config.csf_link_rate
        if rate <= 0.0:
            return
        for mix in chambers.values():
            names = set(mix) | set(self.bath)
            for name in names:
                delta = rate * (mix.get(name, 0.0) - self.bath.get(name, 0.0))
                if delta == 0.0:
                    continue
                mix[name] = mix.get(name, 0.0) - delta
                self.bath[name] = self.bath.get(name, 0.0) + delta

    def _pump_csf_to_rhizome(self) -> None:
        """Let only the active seed clean global CSF into one rhizome.

        Cousin hearts all exchange with ``bath`` through their CSF hooks,
        but none owns or duplicates this store. Re-rooting only updates
        ``rhizome_owner_id``; every stored amount remains conserved.
        """
        self.rhizome_owner_id = self.anchor_id
        if "main" not in self.hearts or self.hearts["main"].seed_owner_id != self.anchor_id:
            return
        rate = max(0.0, min(1.0, self.config.rhizome_csf_pump_rate))
        if rate <= 0.0:
            return
        for name, amount in list(self.bath.items()):
            if name == "solvent" or amount <= 0.0:
                continue
            moved = amount * rate
            self.bath[name] = amount - moved
            self.rhizome[name] = self.rhizome.get(name, 0.0) + moved

    def _exude_rhizome_to_soil(self) -> None:
        """Exude stored non-solvent material as conserved soil salts."""
        if self.rhizome_owner_id != self.anchor_id:
            return
        rate = max(0.0, min(1.0, self.config.rhizome_soil_exudation_rate))
        if rate <= 0.0:
            return
        for name, amount in list(self.rhizome.items()):
            if amount <= 0.0:
                continue
            moved = amount * rate
            self.rhizome[name] = amount - moved
            self.soil[name] = self.soil.get(name, 0.0) + moved
    def _region_pump_nodes(self, network_roots: Optional[Dict[int, int]] = None) -> Dict[str, int]:
        """Heart pumps exist only on the current seed tier (signed level 0).

        ``network_roots`` is accepted for compatibility with older callers,
        but orthogonal display roots are not heart owners: they may occur at
        any radius. The anchor owns ``main``; every other live level-zero
        seed owns exactly ``net:<id>``.
        """
        pumps: Dict[str, int] = {"main": self.anchor_id}
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id or int(node.level or 0) != 0:
                continue
            pumps[f"net:{node_id}"] = node_id
        return pumps
    def _pump_hearts(self) -> None:
        """Beat every region's heart (see Heart), each at its own pump
        node -- the anchor for main, each cousin network's root for its
        own. Region-locked traversals mean a heart only ever sees its own
        network's supply. Lymph return runs first (bath -> seed heart's
        in-chambers), then each region beats independently."""
        network_roots = self.orthogonal_network_roots()
        pumps = self._region_pump_nodes(network_roots)
        self._reap_dead_hearts(set(pumps))
        self._lymph_return()
        for region, pump_id in pumps.items():
            self._pump_region(region, pump_id, network_roots)
        self._pump_csf_to_rhizome()
        self._exude_rhizome_to_soil()

    def _spill_heart_to_csf(self, region: str) -> None:
        """Remove one heart and conserve all chamber/store contents in CSF."""
        heart = self.hearts.pop(region, None)
        if heart is None:
            return
        for mixture in heart.chambers.values():
            for name, amount in mixture.items():
                if amount:
                    self.bath[name] = self.bath.get(name, 0.0) + amount
        for reservoir in heart.reservoirs.values():
            for name, amount in reservoir.drain().items():
                if amount:
                    self.bath[name] = self.bath.get(name, 0.0) + amount

    def _reap_dead_hearts(self, live_regions: "set[str]") -> None:
        """Spill hearts that no longer belong to a live level-zero seed.

        Region survival alone is insufficient: after re-rooting a stale
        heart key can remain while its owner has moved off the seed tier.
        """
        dead_regions = []
        for region, heart in self.hearts.items():
            owner_id = heart.seed_owner_id
            owner = self.nodes.get(owner_id) if owner_id is not None else None
            owner_is_seed = (
                owner is not None and not owner.burned and int(owner.level or 0) == 0
            )
            owner_matches_region = (
                (region == "main" and owner_id == self.anchor_id)
                or (region != "main" and region == f"net:{owner_id}")
            )
            if region not in live_regions or not owner_is_seed or not owner_matches_region:
                dead_regions.append(region)
        for region in dead_regions:
            self._spill_heart_to_csf(region)
    def _pump_region(self, region: str, pump_id: int, network_roots: Dict[int, int]) -> None:
        """One region's heart beat: sort every pump-node-open subedge into
        its slice's in/out chamber routes, measure intake from one pre-
        beat snapshot, land it, then run the script phase (valves,
        reservoir exchange, contractions, osmotic rebalance, squeeze).
        The pump node's own solubles are never touched -- chambers and
        reservoirs hold supply, the node doesn't."""
        inflow_routes: Dict[str, List[Tuple[SubEdge, FluxNode]]] = {}
        outflow_routes: Dict[str, List[Tuple[SubEdge, FluxNode]]] = {}
        for traversal in self.traversals.values():
            start_id, end_id = traversal.start_id, traversal.end_id
            if pump_id not in (start_id, end_id):
                continue
            for sub in traversal.subedges:
                if not sub.is_open_at(pump_id):
                    continue
                src_id, dst_id = (start_id, end_id) if sub.direction == "forward" else (end_id, start_id)
                is_inflow = dst_id == pump_id
                far_id = src_id if is_inflow else dst_id
                far = self.nodes.get(far_id)
                if far is None or far.burned:
                    continue
                slices = self._node_slice(far_id, far, network_roots)
                if slices is None:
                    continue
                routes = inflow_routes if is_inflow else outflow_routes
                routes.setdefault(slices[0], []).append((sub, far))

        if not inflow_routes and not outflow_routes and region not in self.hearts:
            return  # nothing to pump and no heart yet -- don't materialize one

        pump = self.nodes[pump_id]
        heart = self._configure_seed_heart(region, pump_id, initially_full=False)

        intake_plans = {
            slice_name: self._chamber_intake_plan(routes, pump)
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
        heart.exchange_seed_reservoirs()
        self._permeate_heart_forward_to_background(heart)
        if phase is not None:
            valves = heart._resolve_valves(phase)
            contractions = heart._resolve_contractions(phase)
            heart._osmotic_rebalance(valves)
            heart._squeeze_in_chambers(valves, contractions)
            self._squeeze_out_chambers(phase, contractions, outflow_routes, heart)
            heart.phase_index = (heart.phase_index + 1) % max(len(heart.script), 1)
        heart._run_hooks("post")

    def _lymph_return(self) -> None:
        """Drain a fraction of the CSF bath home into the seed heart's
        in-chambers -- lymph returning to the brain. Split evenly across
        the main heart's in-chambers; dormant while lymph_return_rate is
        0 or the main heart hasn't formed yet."""
        rate = self.config.lymph_return_rate
        if rate <= 0.0 or not self.bath:
            return
        heart = self.hearts.get("main")
        if heart is None:
            return
        in_keys = [k for k in heart.chambers if k.endswith("|in")]
        if not in_keys:
            return
        share = 1.0 / len(in_keys)
        for name, amount in list(self.bath.items()):
            drained = amount * rate
            if drained == 0.0:
                continue
            self.bath[name] = amount - drained
            for key in in_keys:
                heart.chambers[key][name] = heart.chambers[key].get(name, 0.0) + drained * share

    def _squeeze_out_chambers(
        self,
        phase: HeartPhase,
        contractions: Dict[str, float],
        outflow_routes: Dict[str, List[Tuple["SubEdge", FluxNode]]],
        heart: "Heart",
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
            mix = heart.chambers.get(key)
            if not mix:
                continue
            routes = all_routes if phase.exit_scope == "all" else outflow_routes.get(out_slice, [])
            if not routes:
                continue
            total_constriction = sum(
                self._learned_subedge_opening(sub) for sub, _ in routes
            )
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
                opening = self._learned_subedge_opening(sub)
                self._deposit_mixture(
                    far, expelled, scale=opening / total_constriction
                )

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

