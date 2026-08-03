#!/usr/bin/env python3
"""A graph that grows bidirectionally from an anchor and is never "done."

Nodes hang into the substrate from an anchor sequence, one token per edge,
growing forward (append) and backward (prepend) at once. There is no
committed sentence and no finalization step -- at any point you can ask
for the current best path, but the graph keeps growing and re-scoring for
as long as you keep ticking it.

Every node gets pressure -- a live physical value, never an accumulated
language score. Each tick couples node casings, the ordered lumen segments
of every audited path tube, larger per-edge hulls, and a spatial passive
CSF/lymph bath. Hearts and exchangers move conserved water and named ions;
local hydration, osmotic loading, and compliance then determine pressure.
Model score remains a reward and reproduction signal: it can teach useful
physiology and favor future growth, but it cannot manufacture pressure.

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
import random
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
    # One persistent lumen compartment per physical edge crossed.  The list
    # is ordered in this tube's direction of travel.  A traversal therefore
    # has two independent continuous tubes, not an endpoint-to-endpoint
    # teleport whose result is merely painted onto the intervening edges.
    segment_edge_keys: List[Tuple[int, int]] = field(default_factory=list)
    segment_solvent: List[float] = field(default_factory=list)
    segment_solubles: List[Dict[str, float]] = field(default_factory=list)
    segment_pressures: List[float] = field(default_factory=list)
    delivered_utility: float = 0.0

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
    # Slow, non-trainable developmental memory. Useful named-material flow
    # hardens a pipe; disuse softens it. Shared physiology archetypes remain
    # the only learned parameters, so topology growth does not grow the
    # optimizer state.
    maturity: float = 0.0
    # Fluid in the larger per-edge casing around all traversal lumens.
    # Its node valves are independent of every inner SubEdge valve.
    hull_solvent: float = 0.0
    hull_solubles: Dict[str, float] = field(default_factory=dict)
    hull_pressure: float = 0.0


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

        # The ion gate sits in the chamber's own water -- an ion cannot cross
        # it, in either direction, unless the chamber actually has solvent to
        # be dissolved in right now. No water at the membrane means no ion
        # transport math at all, regardless of concentration pressure.
        chamber_has_water = chamber.get("solvent", 0.0) > 0.0

        # The same gate skims excess ions or supplies deficient chambers.
        # An empty new reservoir begins with a zero band, so the first
        # matching chamber supply is excess and establishes its history.
        if chamber_has_water and volume > 0.0 and target > 0.0:
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
        elif chamber_has_water and volume > 0.0 and chamber_ions > 0.0:
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
    # Direction-specific local growth commitment. Sustained opposite-ion
    # scarcity produces it, nearby same-lineage tissue shares a little, and
    # new growth inherits part of it. These historical names remain for save
    # compatibility and UI continuity.
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
    # A bare pressure inequality with zero cooldown means the causal
    # center random-walks to whichever node has the highest *instantaneous*
    # pressure every single tick -- a freshly grown leaf and a settling
    # anchor (actively losing pressure to humidity/factory/bulk-transfer
    # outflow every tick) cross constantly, never a rare event. That
    # thrashes _reroot's own heart teardown/rebuild every tick (see
    # _reroot's _spill_heart_to_csf) instead of letting one lineage
    # actually get explored -- "crowds with hearts, doesn't beam search."
    # reroot_margin requires a real, decisive win, not a coin flip;
    # reroot_cooldown_ticks guarantees a newly-rooted lineage gets that
    # many ticks to actually develop before it can be displaced again.
    # Neither throttles the anchor_can_decay forced-replacement path --
    # that one is survival (the anchor is already dead), not exploration
    # stability, and must never be blocked by a cooldown.
    reroot_margin: float = 0.05
    reroot_cooldown_ticks: int = 3
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
    air_root_hot_loop_depth: int = 1
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
    # dominance still means something in topp mode too. "stddev" keeps
    # branch_factor children but samples them across standardized score
    # bands, preserving a representative cross-section of the distribution.
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
    # Positive developmental counterpart to inhibitory auxin. Remaining
    # post-physiology scarcity produces a bounded, slowly-decaying local
    # commitment. Above the threshold it outranks ordinary actions within
    # that direction's existing compute budget.
    growth_commitment_gain: float = 1.0
    growth_commitment_retention: float = 0.85
    growth_commitment_diffusion: float = 0.15
    growth_commitment_inheritance: float = 0.5
    growth_commitment_after_growth: float = 0.25
    growth_commitment_threshold: float = 3.0
    growth_commitment_max: float = 6.0
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
    # Finite seed-relative habitat patches. A new active seed discovers one
    # patch containing this much of each currently-existing network-side ion;
    # returning to an old seed returns to the same depleted patch.
    habitat_ring_ion_amount: float = 4.0
    ring_uptake_rate: float = 1.0
    # Useful solute flow slowly hardens physical branches. Maturity is scalar
    # graph state, not a per-edge learned parameter.
    branch_maturity_gain: float = 0.1
    branch_maturity_retention: float = 0.995
    branch_maturity_conductance_bonus: float = 1.0
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
    # Coupled fluid solver. Pressure is generated only from local hydration
    # and ion loading (plus heart/pump transfers performed immediately before
    # the solve); language-model score never enters this equation.
    fluid_solver_substeps: int = 8
    fluid_time_step: float = 0.08
    fluid_bulk_conductance: float = 0.35
    fluid_diffusion_conductance: float = 0.6
    fluid_osmotic_pressure: float = 0.25
    node_compliance: float = 1.0
    tube_compliance: float = 0.5
    hull_compliance: float = 2.0
    bath_compliance: float = 4.0
    hull_node_valve: float = 0.35
    hull_bath_permeability: float = 0.08
    node_bath_permeability: float = 0.04
    bath_graph_conductance: float = 0.2


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
        # Spatial CSF/lymph/interstitial compartments. ``bath`` remains the
        # compatibility inlet/outlet reservoir; material entering it is
        # deposited at the active seed before each coupled solve.
        self.bath_by_node: Dict[int, Dict[str, float]] = {}
        self._pending_traversal_tubes: Dict[str, Any] = {}
        # Live UI sessions may install a synchronous delegate. The solver
        # publishes an immutable work packet and blocks here until that client
        # returns a conservation-checked relaxation result.
        self.fluid_work_delegate: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None
        self.last_fluid_proof: Optional[Dict[str, Any]] = None
        # Environmental mixtures outside circulation. Forward material
        # crosses level-zero into background; root-needed forward ions
        # cross again, more slowly, into soil.
        self.background: Dict[str, float] = {}
        self.soil: Dict[str, float] = {}
        self.rhizome: Dict[str, float] = {}
        self.rhizome_owner_id: Optional[int] = None
        # Finite environmental stores indexed by seed identity. The active
        # anchor selects the current patch; changing anchor is the organism's
        # movement through phrase space. Patch inventories survive departure,
        # so returning does not manufacture fresh material.
        self.habitats: Dict[int, Dict[str, float]] = {}
        self.habitat_signatures: Dict[int, List[int]] = {}
        self.movement_count: int = 0
        # Ticks elapsed since the last reroot -- see FluxGraphConfig.
        # reroot_cooldown_ticks. Large (not 0) so a reroot is allowed
        # immediately if the very first challenger check already clears
        # the cooldown at tick 1.
        self._ticks_since_reroot: int = 10**9
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
        # domain's observations (see _ingest_from_habitat_shells). Latest payload
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
    _EDGE_ARCHETYPE_FEATURES = _SUBEDGE_ARCHETYPE_FEATURES
    _NODE_ARCHETYPE_FEATURES = (
        "bias",
        "pressure",
        "water_fraction",
        "osmotic_fraction",
        "material_fraction",
        "scarcity",
    )
    _HEART_ARCHETYPE_FEATURES = (
        "bias",
        "declared_throttle",
        "circulating_fraction",
        "csf_fraction",
    )
    _RESERVOIR_ARCHETYPE_FEATURES = (
        "bias",
        "fullness",
        "ion_fraction",
        "solvent_fraction",
    )

    @classmethod
    def _subedge_archetype_key(cls, direction: str, feature: str) -> str:
        return f"archetype:subedge:{direction}:{feature}"

    @staticmethod
    def _archetype_key(kind: str, feature: str) -> str:
        return f"archetype:{kind}:{feature}"

    def _ensure_linear_archetype(
        self,
        kind: str,
        features: Tuple[str, ...],
        initial: Optional[float] = None,
    ) -> None:
        opening = (
            self.config.physiology_initial_opening
            if initial is None else initial
        )
        for feature in features:
            key = self._archetype_key(kind, feature)
            if key in self.physiology_parameters:
                continue
            if feature == "bias":
                self._physiology_parameter(key, opening)
            else:
                self.physiology_parameters[key] = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.float32, device=self.device)
                )

    def _linear_archetype_openings(
        self, kind: str, features: Tuple[str, ...], feature_tensor
    ):
        weights = torch.stack(
            [
                self.physiology_parameters[self._archetype_key(kind, name)]
                for name in features
            ]
        )
        return torch.sigmoid(feature_tensor @ weights)

    def _ensure_state_archetypes(self) -> None:
        """Collapse old instance logits into fixed-size state archetypes."""
        if torch is None or not self.config.physiology_learning_enabled:
            return
        legacy = {
            key: parameter
            for key, parameter in self.physiology_parameters.items()
            if not key.startswith("archetype:")
        }

        def mean_opening(predicate) -> float:
            values = [
                float(torch.sigmoid(parameter).detach().item())
                for key, parameter in legacy.items()
                if predicate(key)
            ]
            return (
                sum(values) / len(values)
                if values else self.config.physiology_initial_opening
            )

        for direction in ("forward", "reverse"):
            self._ensure_linear_archetype(
                f"edge:{direction}",
                self._EDGE_ARCHETYPE_FEATURES,
                mean_opening(
                    lambda key, direction=direction:
                    key.startswith("edge:") and key.endswith(f":{direction}")
                ),
            )
        self._ensure_linear_archetype(
            "node:hull",
            self._NODE_ARCHETYPE_FEATURES,
            mean_opening(lambda key: key.startswith("node:") and key.endswith(":hull")),
        )
        self._ensure_linear_archetype(
            "node:pore",
            self._NODE_ARCHETYPE_FEATURES,
            mean_opening(lambda key: key.startswith("node:") and ":pore:" in key),
        )
        self._ensure_linear_archetype(
            "heart:valve",
            self._HEART_ARCHETYPE_FEATURES,
            mean_opening(lambda key: key.startswith("heart:") and ":valve:" in key),
        )
        for gate_name in ("coverage", "exchange", "membrane"):
            self._ensure_linear_archetype(
                f"reservoir:{gate_name}",
                self._RESERVOIR_ARCHETYPE_FEATURES,
                mean_opening(
                    lambda key, gate_name=gate_name:
                    key.startswith("heart:")
                    and ":reservoir:" in key
                    and key.endswith(f":{gate_name}")
                ),
            )
        for key in legacy:
            del self.physiology_parameters[key]

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
        self._ensure_state_archetypes()

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
        # Kept as a zero-valued migration column so older saved archetype
        # tensors remain shape-compatible. Score is a teaching/reproduction
        # reward, never an input that opens a physical valve by itself.
        quality = torch.zeros_like(path_score)
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
        if not self.config.physiology_learning_enabled or torch is None:
            return hull * pore
        self._ensure_subedge_archetype()
        volume = max(node.volume, 1e-12)
        water_fraction = max(0.0, node.solvent) / volume
        osmotic_fraction = max(0.0, volume - node.solvent) / volume
        material_fraction = max(0.0, node.solubles.get(material, 0.0)) / volume
        features = torch.tensor(
            [[
                1.0,
                math.tanh(float(node.pressure)),
                water_fraction,
                osmotic_fraction,
                material_fraction,
                1.0 - min(1.0, material_fraction),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        hull_opening = self._linear_archetype_openings(
            "node:hull", self._NODE_ARCHETYPE_FEATURES, features
        )[0]
        pore_opening = self._linear_archetype_openings(
            "node:pore", self._NODE_ARCHETYPE_FEATURES, features
        )[0]
        learned = float((hull_opening * pore_opening).detach().item())
        return hull * pore * learned

    def node_archetype_opening_state(
        self, nodes: Optional[List[FluxNode]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Evaluate all displayed node hulls and pores in two tensor batches."""
        selected = nodes or [
            node for node in self.nodes.values() if not node.burned
        ]
        if not selected:
            return {}
        materials = sorted(
            {
                "solvent",
                *(
                    name
                    for node in selected
                    for name in (
                        set(node.solubles) | set(node.pore_permeabilities)
                    )
                ),
            }
        )
        if not self.config.physiology_learning_enabled or torch is None:
            return {
                node.id: {
                    "hull": 1.0,
                    "pores": {name: 1.0 for name in materials},
                }
                for node in selected
            }
        self._ensure_subedge_archetype()
        rows = []
        material_rows = []
        for node in selected:
            volume = max(node.volume, 1e-12)
            water = max(0.0, node.solvent) / volume
            osmotic = max(0.0, volume - node.solvent) / volume
            pressure = math.tanh(node.pressure)
            rows.append([
                1.0, pressure, water, osmotic, osmotic,
                1.0 - min(1.0, osmotic),
            ])
            material_rows.extend(
                [
                    1.0,
                    pressure,
                    water,
                    osmotic,
                    (
                        max(0.0, node.solvent)
                        if name == "solvent"
                        else max(0.0, node.solubles.get(name, 0.0))
                    ) / volume,
                    1.0 - min(
                        1.0,
                        (
                            max(0.0, node.solvent)
                            if name == "solvent"
                            else max(0.0, node.solubles.get(name, 0.0))
                        ) / volume,
                    ),
                ]
                for name in materials
            )
        hull = self._linear_archetype_openings(
            "node:hull",
            self._NODE_ARCHETYPE_FEATURES,
            torch.tensor(rows, dtype=torch.float32, device=self.device),
        ).detach().cpu().tolist()
        pore = self._linear_archetype_openings(
            "node:pore",
            self._NODE_ARCHETYPE_FEATURES,
            torch.tensor(
                material_rows, dtype=torch.float32, device=self.device
            ),
        ).reshape(len(selected), len(materials)).detach().cpu().tolist()
        return {
            node.id: {
                "hull": hull_value,
                "pores": dict(zip(materials, pore_values)),
            }
            for node, hull_value, pore_values in zip(selected, hull, pore)
        }

    def _learned_edge_opening(
        self, parent_id: int, child_id: int, direction: str
    ) -> float:
        if not self.config.physiology_learning_enabled or torch is None:
            return 1.0
        self._ensure_subedge_archetype()
        start = self.nodes[parent_id]
        end = self.nodes[child_id]
        if direction == "reverse":
            start, end = end, start
        start_volume = max(start.volume, 1e-12)
        end_volume = max(end.volume, 1e-12)
        features = torch.tensor(
            [[
                1.0,
                math.tanh(start.pressure - end.pressure),
                max(0.0, start.solvent) / start_volume,
                max(0.0, end.solvent) / end_volume,
                math.tanh(start_volume - end_volume),
                max(0.0, start_volume - start.solvent) / start_volume,
                max(0.0, end_volume - end.solvent) / end_volume,
                math.exp(min(0.0, end.local_evidence)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        return float(
            self._linear_archetype_openings(
                f"edge:{direction}", self._EDGE_ARCHETYPE_FEATURES, features
            )[0].detach().item()
        )

    def edge_archetype_opening_state(
        self, edge_keys: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Evaluate every requested physical edge valve in two batches."""
        keys = edge_keys or list(self.edges)
        keys = [
            (parent_id, child_id)
            for parent_id, child_id in keys
            if parent_id in self.nodes and child_id in self.nodes
        ]
        if not keys:
            return {}
        if not self.config.physiology_learning_enabled or torch is None:
            return {key: (1.0, 1.0) for key in keys}
        self._ensure_subedge_archetype()
        rows = {"forward": [], "reverse": []}
        for parent_id, child_id in keys:
            parent = self.nodes[parent_id]
            child = self.nodes[child_id]
            quality = math.exp(min(0.0, child.local_evidence))
            for direction, source, destination in (
                ("forward", parent, child),
                ("reverse", child, parent),
            ):
                source_volume = max(source.volume, 1e-12)
                destination_volume = max(destination.volume, 1e-12)
                rows[direction].append([
                    1.0,
                    math.tanh(source.pressure - destination.pressure),
                    max(0.0, source.solvent) / source_volume,
                    max(0.0, destination.solvent) / destination_volume,
                    math.tanh(source_volume - destination_volume),
                    max(0.0, source_volume - source.solvent)
                    / source_volume,
                    max(0.0, destination_volume - destination.solvent)
                    / destination_volume,
                    quality,
                ])
        openings = {
            direction: self._linear_archetype_openings(
                f"edge:{direction}",
                self._EDGE_ARCHETYPE_FEATURES,
                torch.tensor(
                    direction_rows,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ).detach().cpu().tolist()
            for direction, direction_rows in rows.items()
        }
        return {
            key: (
                openings["forward"][index],
                openings["reverse"][index],
            )
            for index, key in enumerate(keys)
        }

    def _heart_valve_modulator(
        self, region: str, in_slice: str, out_slice: str, throttle: float
    ) -> float:
        if not self.config.physiology_learning_enabled or torch is None:
            return max(0.0, min(1.0, float(throttle)))
        self._ensure_subedge_archetype()
        heart = self.hearts.get(region)
        circulating = (
            sum(sum(max(0.0, value) for value in mix.values())
                for mix in heart.chambers.values())
            if heart is not None else 0.0
        )
        csf = sum(max(0.0, value) for value in self.bath.values())
        total = circulating + csf
        features = torch.tensor(
            [[
                1.0,
                max(0.0, min(1.0, float(throttle))),
                circulating / max(total, 1e-12),
                csf / max(total, 1e-12),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        opening = self._linear_archetype_openings(
            "heart:valve", self._HEART_ARCHETYPE_FEATURES, features
        )[0]
        return max(
            0.0,
            min(1.0, float(throttle) * float(opening.detach().item())),
        )

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
        self._ensure_physiology_parameters()
        for region, heart in self.hearts.items():
            for name, reservoir in heart.reservoirs.items():
                total = max(reservoir.storage_volume, 1e-12)
                features = torch.tensor(
                    [[
                        1.0,
                        reservoir.fullness,
                        max(0.0, reservoir.ion_amount) / total,
                        max(0.0, reservoir.solvent) / total,
                    ]],
                    dtype=torch.float32,
                    device=self.device,
                )
                reservoir.opening_coverage = float(
                    self._linear_archetype_openings(
                        "reservoir:coverage",
                        self._RESERVOIR_ARCHETYPE_FEATURES,
                        features,
                    )[0].detach().item()
                )
                reservoir.exchange_probability = float(
                    self._linear_archetype_openings(
                        "reservoir:exchange",
                        self._RESERVOIR_ARCHETYPE_FEATURES,
                        features,
                    )[0].detach().item()
                )
                reservoir.membrane_permeability = float(
                    self._linear_archetype_openings(
                        "reservoir:membrane",
                        self._RESERVOIR_ARCHETYPE_FEATURES,
                        features,
                    )[0].detach().item()
                )

    def _ensure_physiology_parameters(self) -> None:
        if not self.config.physiology_learning_enabled or torch is None:
            return
        self._ensure_subedge_archetype()
        for region, heart in self.hearts.items():
            self._bind_heart_learning(region, heart)

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
        for parameter in self.physiology_parameters.values():
            parameter.grad = None
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
        physical_utility = torch.tensor(
            [
                sum(sub.delivered_utility for sub in traversal.subedges)
                for _, traversal in live_traversals
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if bool((physical_utility > 0.0).any()):
            physical_utility = physical_utility / physical_utility.max().clamp_min(1e-12)
        expected_route_opening = (
            traversal_weights
            * physical_utility
            * archetype_openings.mean(dim=1)
        ).sum()
        reward_terms = [expected_route_opening]
        resource_terms = [archetype_openings.mean()]

        live_edges = [
            (parent_id, child_id)
            for parent_id, child_id in self.edges
            if parent_id in self.nodes
            and child_id in self.nodes
            and not self.nodes[parent_id].burned
            and not self.nodes[child_id].burned
        ]
        if live_edges:
            edge_quality = []
            forward_rows = []
            reverse_rows = []
            for parent_id, child_id in live_edges:
                parent = self.nodes[parent_id]
                child = self.nodes[child_id]
                parent_volume = max(parent.volume, 1e-12)
                child_volume = max(child.volume, 1e-12)
                quality = math.exp(min(0.0, child.local_evidence))
                edge_quality.append(quality)
                forward_rows.append([
                    1.0,
                    math.tanh(parent.pressure - child.pressure),
                    max(0.0, parent.solvent) / parent_volume,
                    max(0.0, child.solvent) / child_volume,
                    math.tanh(parent_volume - child_volume),
                    max(0.0, parent_volume - parent.solvent) / parent_volume,
                    max(0.0, child_volume - child.solvent) / child_volume,
                    quality,
                ])
                reverse_rows.append([
                    1.0,
                    math.tanh(child.pressure - parent.pressure),
                    max(0.0, child.solvent) / child_volume,
                    max(0.0, parent.solvent) / parent_volume,
                    math.tanh(child_volume - parent_volume),
                    max(0.0, child_volume - child.solvent) / child_volume,
                    max(0.0, parent_volume - parent.solvent) / parent_volume,
                    quality,
                ])
            edge_quality_tensor = torch.tensor(
                edge_quality, dtype=torch.float32, device=self.device
            )
            edge_weights = torch.softmax(
                edge_quality_tensor / temperature, dim=0
            )
            for direction, rows in (
                ("forward", forward_rows),
                ("reverse", reverse_rows),
            ):
                opening = self._linear_archetype_openings(
                    f"edge:{direction}",
                    self._EDGE_ARCHETYPE_FEATURES,
                    torch.tensor(
                        rows, dtype=torch.float32, device=self.device
                    ),
                )
                reward_terms.append((edge_weights * opening).sum())
                resource_terms.append(opening.mean())

        live_nodes = [node for node in self.nodes.values() if not node.burned]
        if live_nodes:
            node_rows = []
            node_quality = []
            for node in live_nodes:
                volume = max(node.volume, 1e-12)
                water = max(0.0, node.solvent) / volume
                osmotic = max(0.0, volume - node.solvent) / volume
                material = osmotic
                node_rows.append([
                    1.0,
                    math.tanh(node.pressure),
                    water,
                    osmotic,
                    material,
                    1.0 - min(1.0, material),
                ])
                node_quality.append(math.exp(min(0.0, node.local_evidence)))
            node_features = torch.tensor(
                node_rows, dtype=torch.float32, device=self.device
            )
            node_weights = torch.softmax(
                torch.tensor(
                    node_quality, dtype=torch.float32, device=self.device
                ) / temperature,
                dim=0,
            )
            for kind in ("node:hull", "node:pore"):
                opening = self._linear_archetype_openings(
                    kind, self._NODE_ARCHETYPE_FEATURES, node_features
                )
                reward_terms.append((node_weights * opening).sum())
                resource_terms.append(opening.mean())

        heart_rows = []
        for heart in self.hearts.values():
            circulating = sum(
                sum(max(0.0, value) for value in mix.values())
                for mix in heart.chambers.values()
            )
            csf = sum(max(0.0, value) for value in self.bath.values())
            total = circulating + csf
            heart_rows.append([
                1.0,
                1.0,
                circulating / max(total, 1e-12),
                csf / max(total, 1e-12),
            ])
        if heart_rows:
            heart_opening = self._linear_archetype_openings(
                "heart:valve",
                self._HEART_ARCHETYPE_FEATURES,
                torch.tensor(
                    heart_rows, dtype=torch.float32, device=self.device
                ),
            )
            reward_terms.append(heart_opening.mean())
            resource_terms.append(heart_opening.mean())

        reservoir_rows = []
        for heart in self.hearts.values():
            for reservoir in heart.reservoirs.values():
                total = max(reservoir.storage_volume, 1e-12)
                reservoir_rows.append([
                    1.0,
                    reservoir.fullness,
                    max(0.0, reservoir.ion_amount) / total,
                    max(0.0, reservoir.solvent) / total,
                ])
        if reservoir_rows:
            reservoir_features = torch.tensor(
                reservoir_rows, dtype=torch.float32, device=self.device
            )
            for gate_name in ("coverage", "exchange", "membrane"):
                opening = self._linear_archetype_openings(
                    f"reservoir:{gate_name}",
                    self._RESERVOIR_ARCHETYPE_FEATURES,
                    reservoir_features,
                )
                reward_terms.append(opening.mean())
                resource_terms.append(opening.mean())

        expected_survival = torch.stack(reward_terms).mean()
        resource_opening = torch.stack(resource_terms).mean()
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
            if key.startswith("archetype:") and key.endswith(":bias")
        } if torch is not None else {}
        archetype_coefficients = {
            key: float(parameter.detach().item())
            for key, parameter in self.physiology_parameters.items()
            if key.startswith("archetype:")
        } if torch is not None else {}
        by_kind: Dict[str, List[float]] = {}
        for key, opening in openings.items():
            parts = key.split(":")
            kind = ":".join(parts[1:-1])
            by_kind.setdefault(kind, []).append(opening)
        values = list(openings.values())
        archetype_groups: Dict[str, int] = {}
        for key in archetype_coefficients:
            parts = key.split(":")
            kind = ":".join(parts[1:-1])
            archetype_groups[kind] = archetype_groups.get(kind, 0) + 1
        return {
            "enabled": self.config.physiology_learning_enabled,
            "parameter_count": len(self.physiology_parameters),
            "archetype_parameter_count": len(archetype_coefficients),
            "archetypes": {
                kind: {
                    "parameter_count": count,
                }
                for kind, count in archetype_groups.items()
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
            # The anchor sits at the center and belongs to no slice (see
            # _node_slice), so it never receives the ordinary per-slice
            # ambient humidity exchange every other node gets -- without
            # its own starting reserve it would stay permanently dry and
            # (now that growth requires water) could never sprout its
            # first children at all. 1.0 matches uniform_field's default
            # ambient level, i.e. "the seed starts as hydrated as the
            # steady state everything else is drawn toward" -- not an
            # arbitrary number. Only while the fluid layer is actually on:
            # with it off there is no water concept in play (growth stays
            # ungated too, see _expandable_nodes), and plain score-driven
            # pressure derivation expects a freshly seeded graph at exactly
            # zero physical inventory.
            solvent=1.0 if self.config.graph_auditor_enabled else 0.0,
        )
        self.anchor_id = node_id
        self.rhizome_owner_id = node_id
        self._configure_seed_heart("main", node_id, initially_full=True)
        self._ensure_current_habitat()
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

    def _ensure_current_habitat(self) -> Dict[str, float]:
        """Return the active seed's persistent finite environmental patch.

        A habitat is created exactly once per seed node. Its initial named
        ions reflect the regional hearts present when that phrase-space
        location is first entered. Later topology can use those materials but
        cannot make the patch refill; revisiting the seed returns to the same
        inventory.
        """
        if self.anchor_id is None:
            return {}
        patch = self.habitats.get(self.anchor_id)
        if patch is not None:
            return patch
        amount = max(0.0, float(self.config.habitat_ring_ion_amount))
        patch = {}
        for region in self._region_pump_nodes():
            patch[f"{region}:forward"] = amount
            patch[f"{region}:backward"] = amount
        self.habitats[self.anchor_id] = patch
        tokens, _ = self.best_path()
        self.habitat_signatures[self.anchor_id] = list(tokens or self.anchor_tokens)
        return patch

    def export_fluid_state(self) -> Dict[str, Any]:
        """The graph-level fluid state that lives outside individual nodes
        -- the CSF bath and every region heart's chambers, reservoirs,
        script, and phase -- for persistence. Node-local fluid
        (solvent/solubles) rides along with each node's own serialization,
        not here."""
        return {
            "bath": dict(self.bath),
            "bath_by_node": {
                str(node_id): dict(mixture)
                for node_id, mixture in self.bath_by_node.items()
            },
            "background": dict(self.background),
            "soil": dict(self.soil),
            "rhizome": dict(self.rhizome),
            "rhizome_owner_id": self.rhizome_owner_id,
            "habitats": {
                str(seed_id): dict(mixture)
                for seed_id, mixture in self.habitats.items()
            },
            "habitat_signatures": {
                str(seed_id): list(tokens)
                for seed_id, tokens in self.habitat_signatures.items()
            },
            "movement_count": self.movement_count,
            "edge_maturity": {
                f"{parent_id}:{child_id}": edge.maturity
                for (parent_id, child_id), edge in self.edges.items()
                if edge.maturity
            },
            "edge_hulls": {
                f"{parent_id}:{child_id}": {
                    "solvent": edge.hull_solvent,
                    "solubles": dict(edge.hull_solubles),
                    "pressure": edge.hull_pressure,
                }
                for (parent_id, child_id), edge in self.edges.items()
                if edge.hull_solvent or edge.hull_solubles
            },
            "traversal_tubes": {
                ",".join(str(part) for part in key): [
                    {
                        "direction": sub.direction,
                        "edge_keys": [list(edge_key) for edge_key in sub.segment_edge_keys],
                        "solvent": list(sub.segment_solvent),
                        "solubles": [dict(mixture) for mixture in sub.segment_solubles],
                        "pressures": list(sub.segment_pressures),
                    }
                    for sub in traversal.subedges
                ]
                for key, traversal in self.traversals.items()
            },
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
        self.bath_by_node = {
            int(node_id): dict(mixture)
            for node_id, mixture in state.get("bath_by_node", {}).items()
        }
        self.background = dict(state.get("background", {}))
        self.soil = dict(state.get("soil", {}))
        self.rhizome = dict(state.get("rhizome", {}))
        self.rhizome_owner_id = self.anchor_id
        self.habitats = {
            int(seed_id): dict(mixture)
            for seed_id, mixture in state.get("habitats", {}).items()
        }
        self.habitat_signatures = {
            int(seed_id): list(tokens)
            for seed_id, tokens in state.get("habitat_signatures", {}).items()
        }
        self.movement_count = int(state.get("movement_count", 0))
        self._ensure_current_habitat()
        for key, value in state.get("edge_maturity", {}).items():
            try:
                parent_text, child_text = str(key).split(":", 1)
                edge = self.edges.get((int(parent_text), int(child_text)))
                if edge is not None:
                    edge.maturity = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        for key, hstate in state.get("edge_hulls", {}).items():
            try:
                parent_text, child_text = str(key).split(":", 1)
                edge = self.edges.get((int(parent_text), int(child_text)))
                if edge is not None:
                    edge.hull_solvent = max(0.0, float(hstate.get("solvent", 0.0)))
                    edge.hull_solubles = {
                        str(name): max(0.0, float(amount))
                        for name, amount in hstate.get("solubles", {}).items()
                    }
                    edge.hull_pressure = max(0.0, float(hstate.get("pressure", 0.0)))
            except (TypeError, ValueError):
                continue
        # Traversals are reconstructed lazily by the auditor. Restore tube
        # inventories only for paths already present (newer loaders may audit
        # before importing); otherwise keep the payload for that first audit.
        self._pending_traversal_tubes = dict(state.get("traversal_tubes", {}))
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
        if self.config.graph_auditor_enabled:
            self._emit_status("auditing", "materializing every causal path tube")
            self._run_graph_auditor()
            self._apply_reservoir_learning()
            self._emit_status("humidity", "exchanging solvent and dissolved ions")
            self._exchange_humidity()
            # A still-dry anchor's first children are withheld by
            # spawn_first_children() itself; retrying here (idempotent
            # once they exist) means they appear the same tick the anchor
            # finally hydrates, rather than needing a separate mechanism.
            self.spawn_first_children()
            self._emit_status("rings", "ingesting boundary supplies")
            self._ingest_from_habitat_shells()
            self._emit_status("pumping", "beating regional hearts")
            self._pump_hearts()
            self._emit_status("soil", "permeating the level-zero interface")
            self._permeate_background_into_soil()
            self._absorb_soil_by_roots()
            self._emit_status("factories", "running node synthesis")
            self._run_node_factories()
        self._emit_status(
            "settling", "relaxing coupled fluid compartments",
            0, self.config.fluid_solver_substeps,
        )
        self._settle_circuit()
        self._emit_status("rerooting", "checking causal focus")
        self._maybe_reroot()
        self._emit_status("digesting", "rolling up path quality")
        self._digest()
        self._emit_status("auxin", "diffusing growth signals")
        self._diffuse_auxin()
        if self.config.graph_auditor_enabled:
            # Except for the boot tick, this reads the fully reconciled
            # post-physiology state left by the preceding tick.
            self._emit_status("nutrients", "building local growth commitment")
            self._update_nutrient_growth_interest()
        self._emit_status("expanding", "selecting growth fronts")
        self._expand_top_pressure_nodes()
        self._emit_status("starvation", "reaping disconnected or starved tissue")
        self._starve_and_burn()
        if self.config.graph_auditor_enabled:
            # Growth can create fresh paths. Materialize their empty lumens
            # now so the next beat sees the exact current vascular topology.
            self._emit_status("auditing", "materializing newly grown path tubes")
            self._run_graph_auditor()
            self._emit_status(
                "learning", "using path score as reward and reproduction signal"
            )
            self._learn_physiology()
            self._emit_status("hardening", "maturing useful physical routes")
            self._update_branch_maturity()
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
        rather than waiting for a challenger to fairly out-score it. This
        forced path is survival, not exploration stability, so it ignores
        reroot_cooldown_ticks entirely -- a dead anchor cannot be left in
        place just because the clock hasn't run out.

        The ordinary challenger path below is gated by both
        reroot_cooldown_ticks (a newly-rooted lineage gets that many ticks
        before it can be displaced again) and reroot_margin (the challenger
        must clear the anchor by more than that, not just any epsilon) --
        without them the causal focus random-walks to whichever node has
        the highest *instantaneous* pressure every single tick (a fresh
        leaf and a settling anchor, which loses pressure to humidity/
        factory/bulk-transfer outflow every tick, cross constantly), tearing
        down and rebuilding every heart on every tick instead of letting one
        lineage actually get explored.
        """
        anchor = self.nodes[self.anchor_id]
        self._ticks_since_reroot += 1

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

        if self._ticks_since_reroot < max(0, int(self.config.reroot_cooldown_ticks)):
            return

        margin = max(0.0, float(self.config.reroot_margin))
        challenger_id = None
        challenger_pressure = anchor.pressure
        for node_id, node in self.nodes.items():
            if node.burned or node_id == self.anchor_id:
                continue
            if node.pressure > challenger_pressure + margin:
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
        self._ticks_since_reroot = 0
        old_anchor_tokens = self.anchor_tokens
        old_anchor_local_evidence = self.anchor_local_evidence
        # The old seed heart does not travel with the focus. Re-rooting
        # dumps every chamber and reservoir into the shared CSF bath.
        self._spill_heart_to_csf("main")
        old_anchor = self.nodes[old_anchor_id]
        # A demotion is not a destruction -- old_anchor stays alive -- but
        # material leaving a role goes through CSF/lymph in every case, no
        # exception for "the node itself survives." Its own solvent/
        # solubles land in its own bath_by_node entry (a live compartment,
        # since this node isn't burned) instead of continuing to sit
        # privately on the node the instant it stops being anchor.
        if old_anchor.solvent or old_anchor.solubles:
            local_bath = self.bath_by_node.setdefault(old_anchor_id, {})
            if old_anchor.solvent:
                local_bath["solvent"] = local_bath.get("solvent", 0.0) + old_anchor.solvent
                old_anchor.solvent = 0.0
            for name, amount in old_anchor.solubles.items():
                if amount:
                    local_bath[name] = local_bath.get(name, 0.0) + amount
            old_anchor.solubles = {}
        old_anchor.tokens = old_anchor_tokens
        old_anchor.local_evidence = old_anchor_local_evidence
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
        self.movement_count += 1
        self._ensure_current_habitat()

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
        # Score is a permanent reward observation. Becoming the active seed
        # changes neither this token's reward nor the accumulated reward of
        # its causal lineage; seed status has no hydraulic privilege.
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
        edge = self.edges.get((parent_id, child_id))
        maturity = edge.maturity if edge is not None else 0.0
        hardening = 1.0 + max(
            0.0, float(self.config.branch_maturity_conductance_bonus)
        ) * max(0.0, min(1.0, maturity))
        if neighbor_id in self.nodes[node_id].parent_ids:
            valve = self._learned_edge_opening(
                parent_id, child_id, "forward"
            )
            return base * valve * hardening
        valve = self._learned_edge_opening(
            parent_id, child_id, "reverse"
        )
        return (
            base
            * self.config.return_conductance_scale
            * valve
            * hardening
        )

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
        """Refresh node pressure from local physical inventory only.

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
        """Settle the coupled physical fluid system.

        Pressure is an observation of hydration and named-ion loading in
        nodes, tube lumens, edge hulls, and spatial bath compartments. Score,
        found bonus, population heuristics, and seed identity do not create
        pressure. They remain reward/reproduction signals used by growth.
        """
        if self.config.graph_auditor_enabled:
            self._run_graph_auditor()
            self._solve_coupled_fluid_system()
            return max(1, int(self.config.fluid_solver_substeps))

        # Without the fluid topology enabled, retain current inventories but
        # still derive node pressure from physical local state only.
        compliance = max(1e-6, float(self.config.node_compliance))
        for node in self.nodes.values():
            if node.burned:
                continue
            node.pressure = (
                max(0.0, node.solvent)
                + self.config.fluid_osmotic_pressure
                * sum(max(0.0, amount) for amount in node.solubles.values())
                / max(1e-6, node.solvent)
            ) / compliance
        return 1

        # Legacy score-driven resistor relaxation is intentionally unreachable
        # below. It remains temporarily as migration context for saved knobs.
        compliance = max(1e-6, float(self.config.node_compliance))
        max_delta = 0.0
        for node in self.nodes.values():
            if node.burned:
                continue
            new_pressure = (
                max(0.0, node.solvent)
                + self.config.fluid_osmotic_pressure
                * sum(max(0.0, amount) for amount in node.solubles.values())
                / max(1e-6, node.solvent)
            ) / compliance
            max_delta = max(max_delta, abs(new_pressure - node.pressure))
            node.pressure = new_pressure
        return max_delta

        # Legacy score-driven pressure code remains below only as migration
        # context for old configuration names; it is intentionally unreachable.
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

    def _stddev_sample_candidates(
        self,
        node: FluxNode,
        candidates: List[Tuple[List[int], float]],
        keep: int,
    ) -> List[Tuple[List[int], float]]:
        """Sample score-standardized bands while preserving probability mass.

        Every occupied one-standard-deviation band receives a representative
        before remaining seats are assigned in proportion to each band's
        model-probability mass. Sampling within a band is probability weighted
        and without replacement. Returned scores remain the true model scores.
        """
        keep = min(max(0, int(keep)), len(candidates))
        if keep == 0:
            return []
        scores = [float(score) for _, score in candidates]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        deviation = math.sqrt(variance)
        if deviation <= 1e-12:
            return candidates[:keep]

        bands: Dict[int, List[int]] = {}
        for index, score in enumerate(scores):
            z_score = (score - mean) / deviation
            # Do not let the enormous aggregate mass of individually
            # implausible tail items dominate the representative sample.
            if z_score < -2.0:
                continue
            bands.setdefault(math.floor(z_score), []).append(index)
        if not bands:
            bands[0] = [max(range(len(scores)), key=scores.__getitem__)]

        rng = random.Random(
            ((self.tick_count + 1) * 0x9E3779B1)
            ^ (node.id * 0x85EBCA77)
            ^ (1 if node.direction is Direction.FORWARD else 2)
        )
        remaining = {band: list(indices) for band, indices in bands.items()}
        selected: List[int] = []
        maximum = max(scores)

        def weight(index: int) -> float:
            return math.exp(scores[index] - maximum)

        def draw_from_band(band: int) -> None:
            indices = remaining[band]
            weights = [weight(index) for index in indices]
            target = rng.random() * sum(weights)
            running = 0.0
            chosen_at = len(indices) - 1
            for position, item_weight in enumerate(weights):
                running += item_weight
                if running >= target:
                    chosen_at = position
                    break
            selected.append(indices.pop(chosen_at))

        band_mass = {
            band: sum(weight(index) for index in indices)
            for band, indices in remaining.items()
        }
        for band in sorted(bands, key=band_mass.get, reverse=True)[:keep]:
            draw_from_band(band)

        while len(selected) < keep:
            live_bands = [band for band, indices in remaining.items() if indices]
            if not live_bands:
                break
            masses = [
                sum(weight(index) for index in remaining[band])
                for band in live_bands
            ]
            target = rng.random() * sum(masses)
            running = 0.0
            chosen_band = live_bands[-1]
            for band, mass in zip(live_bands, masses):
                running += mass
                if running >= target:
                    chosen_band = band
                    break
            draw_from_band(chosen_band)

        return sorted(
            (candidates[index] for index in selected),
            key=lambda pair: pair[1],
            reverse=True,
        )

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
        #
        # Growth is metabolic work: a node with no solvent has nothing to
        # spend and cannot put out new tissue, no matter how much pressure
        # it's carrying -- it simply waits at the frontier until ambient
        # humidity exchange (see _exchange_humidity, which runs earlier
        # this same tick) gives it some. Only enforced while the fluid
        # layer is actually simulating water at all (graph_auditor_enabled)
        # -- with it off there is no water concept in play, so growth stays
        # ungated, exactly as it always has.
        fluid_on = self.config.graph_auditor_enabled
        return [
            n for n in self.nodes.values()
            if not n.burned
            and n.direction is not None
            and (not fluid_on or n.solvent > 0.0)
            and (
                (n.direction is Direction.FORWARD and not self._live_children(n.id))
                or (n.direction is Direction.BACKWARD and not self._live_parents(n.id))
            )
        ]

    def _heart_growth_stresses(self) -> Dict[str, Dict[str, Any]]:
        """Return structural side scarcity for every seed-tier heart.

        A heart is connected on one side only when a direct live neighbor on
        that side belongs to the pump's metabolic lineage. A causal edge into
        an unrelated cousin network does not satisfy the requirement. Missing
        forward tissue is sprout stress; missing backward tissue is root
        stress. Seed-tier pumps have no direction of their own, so this check
        is deliberately separate from node-local ion scarcity.
        """
        stresses: Dict[str, Dict[str, Any]] = {}
        for region, pump_id in self._region_pump_nodes().items():
            pump = self.nodes[pump_id]
            lineage_id = (
                pump.center_id
                if pump.center_id is not None else pump_id
            )

            def same_lineage(node_id: int) -> bool:
                node = self.nodes[node_id]
                node_lineage = (
                    node.center_id
                    if node.center_id is not None else node.id
                )
                return node_lineage == lineage_id

            forward_connections = [
                child_id
                for child_id in self._live_children(pump_id)
                if same_lineage(child_id)
            ]
            backward_connections = [
                parent_id
                for parent_id in self._live_parents(pump_id)
                if same_lineage(parent_id)
            ]
            missing_forward = not forward_connections
            missing_backward = not backward_connections
            pump.forward_growth_interest = (
                1.0 if missing_forward else 0.0
            )
            pump.backward_growth_interest = (
                1.0 if missing_backward else 0.0
            )
            stresses[region] = {
                "pump_id": pump_id,
                "lineage_id": lineage_id,
                "missing_forward": missing_forward,
                "missing_backward": missing_backward,
                "forward_connections": forward_connections,
                "backward_connections": backward_connections,
            }
        return stresses

    def _update_nutrient_growth_interest(self, accumulate: bool = True) -> None:
        """Update the local, direction-specific growth commitment field.

        Each directional node produces commitment from its remaining
        opposite-ion shortfall. Existing signal decays, while the strongest
        signal on adjacent same-lineage, same-side tissue diffuses locally.
        The update is simultaneous so dict iteration order cannot steer the
        hormone field. ``accumulate=False`` remains a compatibility-only
        reconciliation mode that clears a fully supplied site without
        advancing developmental time.
        """
        target = max(1e-12, float(self.config.growth_target_ion_concentration))
        network_roots = self.orthogonal_network_roots()
        scarcity_by_node: Dict[int, float] = {}
        field_by_node: Dict[int, str] = {}
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
            scarcity_by_node[nid] = scarcity
            field_by_node[nid] = interest_name

        if not accumulate:
            for nid, scarcity in scarcity_by_node.items():
                if scarcity <= 0.0:
                    setattr(self.nodes[nid], field_by_node[nid], 0.0)
            return

        retention = max(
            0.0, min(1.0, float(self.config.growth_commitment_retention))
        )
        diffusion = max(
            0.0, min(1.0, float(self.config.growth_commitment_diffusion))
        )
        gain = max(0.0, float(self.config.growth_commitment_gain))
        maximum = max(
            0.0,
            float(
                max(
                    self.config.growth_commitment_max,
                    self.config.growth_commitment_threshold,
                )
            ),
        )
        previous = {
            nid: max(0.0, float(getattr(self.nodes[nid], field_name)))
            for nid, field_name in field_by_node.items()
        }
        updated: Dict[int, float] = {}
        for nid, scarcity in scarcity_by_node.items():
            node = self.nodes[nid]
            field_name = field_by_node[nid]
            ambient = max(
                (
                    previous[neighbor_id]
                    for neighbor_id in self._neighbors(nid)
                    if neighbor_id in previous
                    and field_by_node[neighbor_id] == field_name
                    and self.nodes[neighbor_id].center_id == node.center_id
                ),
                default=0.0,
            )
            retained = retention * previous[nid]
            local_spread = diffusion * max(0.0, ambient - retained)
            updated[nid] = min(
                maximum,
                retained + gain * scarcity + local_spread,
            )
        for nid, value in updated.items():
            setattr(self.nodes[nid], field_by_node[nid], value)
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
        fluid_on = self.config.graph_auditor_enabled
        heart_stresses = self._heart_growth_stresses()
        if any(
            stress["missing_forward"] or stress["missing_backward"]
            for stress in heart_stresses.values()
        ):
            for stress in heart_stresses.values():
                pump_id = stress["pump_id"]
                pump = self.nodes[pump_id]
                if fluid_on and pump.solvent <= 0.0:
                    # No water at the pump means no growth from it this
                    # tick either, same as any other node -- it just waits.
                    continue
                if stress["missing_forward"]:
                    # For a heart this is ordinary shoot/sprout growth, not
                    # cross-growth from an already directional cell.
                    self._expand_forward(pump_id)
                if stress["missing_backward"]:
                    # An air root launched by a heart is simply its root side.
                    self._expand_backward(pump_id)
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
            and (not fluid_on or n.solvent > 0.0)
        )
        backward_pool.extend(
            (n, True) for n in self.nodes.values()
            if not n.burned
            and n.direction is Direction.FORWARD
            and n.backward_growth_interest > 0.0
            and (not fluid_on or n.solvent > 0.0)
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

        threshold = max(
            0.0, float(self.config.growth_commitment_threshold)
        )

        def action_rank(action) -> Tuple[bool, float]:
            node, nutrient_driven = action
            stress = (
                node.forward_growth_interest
                if nutrient_driven and node.direction is Direction.BACKWARD
                else (
                    node.backward_growth_interest
                    if nutrient_driven
                    else 0.0
                )
            )
            return nutrient_driven and stress >= threshold, action_priority(action)

        forward_pool.sort(key=action_rank, reverse=True)
        backward_pool.sort(key=action_rank, reverse=True)

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
                    key=action_rank, reverse=True,
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
            # Expansion precedes starvation in tick(). A node born during
            # this expansion has not yet reached the auditor, humidity,
            # transport, heart, soil, or factory phases below starvation.
            # Do not spend one of its survival strikes before it has had
            # that first complete physiology pass.
            if self.tick_count > 0 and node.created_tick == self.tick_count:
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
            # bath_by_node[cur_id] is permanently excluded from every future
            # coupled fluid solve once cur_id is burned (see
            # _solve_coupled_fluid_system's live_nodes filter) -- depositing
            # this node's water in its own entry would conserve it on paper
            # while actually orphaning it somewhere the pressure/flow solver
            # can never reach again. Hand it instead to a still-living
            # neighbor it was actually connected to, or the anchor as the
            # guaranteed-alive fallback -- the same destination
            # Heart._spill_heart_to_csf uses when a whole heart dies.
            destination_id = next(
                (
                    neighbor_id for neighbor_id in cur.parent_ids + cur.children_ids
                    if neighbor_id in self.nodes and not self.nodes[neighbor_id].burned
                ),
                None,
            )
            if (
                destination_id is None
                and self.anchor_id in self.nodes
                and not self.nodes[self.anchor_id].burned
            ):
                destination_id = self.anchor_id
            if destination_id is not None:
                local_bath = self.bath_by_node.setdefault(destination_id, {})
                if cur.solvent:
                    local_bath["solvent"] = local_bath.get("solvent", 0.0) + cur.solvent
                for name, amount in cur.solubles.items():
                    if amount:
                        local_bath[name] = local_bath.get(name, 0.0) + amount
            cur.solvent = 0.0
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
            selection_mode = self._direction_selection_mode(node.direction)
            base_shortlist_k = (
                self.config.top_p_shortlist_ceiling
                if selection_mode in {"topp", "stddev"}
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
                if selection_mode == "stddev":
                    pre_sample = [([tid], score) for tid, score in scored]
                    if word_trie is not None:
                        pre_sample = self._stddev_sample_candidates(
                            node, pre_sample, shortlist_k
                        )
                    round0 = [(tokens[0], score) for tokens, score in pre_sample]
                else:
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
        if (
            self._direction_selection_mode(node.direction) == "stddev"
            and candidates
        ):
            return self._stddev_sample_candidates(node, candidates, keep)
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
        """Seed one forward and one backward child directly off the anchor.

        The anchor is created with zero solvent (see seed()) -- while the
        fluid layer is on, a dry anchor has nothing to grow with yet, so
        this is a safe no-op instead of forcing growth out of nothing.
        Idempotent and meant to be retried every tick (see tick()) until
        ambient humidity exchange gives the anchor some water: the
        "already has a child on either side" check is what makes repeated
        calls safe rather than re-growing first children over and over.
        """
        anchor = self.nodes[self.anchor_id]
        if self._live_children(self.anchor_id) or self._live_parents(self.anchor_id):
            return
        if self.config.graph_auditor_enabled and anchor.solvent <= 0.0:
            return
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
        commitment_share: float = 1.0,
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
        inheritance = max(
            0.0, min(1.0, float(self.config.growth_commitment_inheritance))
        )
        inherited = inheritance * max(0.0, commitment_share)
        if level > int(source.level or 0):
            node.forward_growth_interest = (
                max(0.0, source.forward_growth_interest) * inherited
            )
        elif level < int(source.level or 0):
            node.backward_growth_interest = (
                max(0.0, source.backward_growth_interest) * inherited
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
        count = max(1, min(len(scores), len(token_spans)))
        for score, tokens in zip(scores, token_spans):
            child = self._new_growth_node(
                source,
                int(source.level or 0) + 1,
                score,
                tokens,
                live_token_spans,
                commitment_share=1.0 / count,
            )
            self._record_growth_edge(source_id, child.id, child, "postfix_beam")
        if count and scores and token_spans:
            source.forward_growth_interest *= max(
                0.0,
                min(1.0, float(self.config.growth_commitment_after_growth)),
            )

    def _attach_backward_parents(
        self, source_id: int, scores: List[float], token_spans: List[List[int]]
    ) -> None:
        source = self.nodes[source_id]
        live_token_spans = self._live_token_spans()
        count = max(1, min(len(scores), len(token_spans)))
        for score, tokens in zip(scores, token_spans):
            parent = self._new_growth_node(
                source,
                int(source.level or 0) - 1,
                score,
                tokens,
                live_token_spans,
                commitment_share=1.0 / count,
            )
            self._record_growth_edge(parent.id, source_id, parent, "prefix_beam")
        if count and scores and token_spans:
            source.backward_growth_interest *= max(
                0.0,
                min(1.0, float(self.config.growth_commitment_after_growth)),
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
        edge_keys = list(zip(node_ids, node_ids[1:]))
        forward_sub = SubEdge(
            traversal_key=key,
            direction="forward",
            segment_edge_keys=list(edge_keys),
            segment_solvent=[0.0] * len(edge_keys),
            segment_solubles=[{} for _ in edge_keys],
            segment_pressures=[0.0] * len(edge_keys),
        )
        reverse_sub = SubEdge(
            traversal_key=key,
            direction="reverse",
            segment_edge_keys=list(reversed(edge_keys)),
            segment_solvent=[0.0] * len(edge_keys),
            segment_solubles=[{} for _ in edge_keys],
            segment_pressures=[0.0] * len(edge_keys),
        )
        for a, b in edge_keys:
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
        saved_tubes = self._pending_traversal_tubes.pop(
            ",".join(str(part) for part in key), None
        )
        if saved_tubes:
            by_direction = {
                str(item.get("direction")): item for item in saved_tubes
            }
            for sub in (forward_sub, reverse_sub):
                item = by_direction.get(sub.direction)
                if not item or len(item.get("solvent", [])) != len(sub.segment_edge_keys):
                    continue
                sub.segment_solvent = [
                    max(0.0, float(value)) for value in item.get("solvent", [])
                ]
                sub.segment_solubles = [
                    {
                        str(name): max(0.0, float(amount))
                        for name, amount in mixture.items()
                    }
                    for mixture in item.get("solubles", [])
                ]
                sub.segment_pressures = [
                    max(0.0, float(value))
                    for value in item.get(
                        "pressures", [0.0] * len(sub.segment_edge_keys)
                    )
                ]

    def audit_edge_influence(
        self, ensure_current: bool = True
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
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
        if ensure_current:
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

    def _exchange_humidity_scalar(self) -> None:
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

            # A dissolved solute cannot move ambient-ward without a solvent
            # to be dissolved in -- a dry node holds its solutes exactly as
            # they are, no matter how large the ambient concentration gap.
            if node.solvent > 0.0:
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

    def _exchange_humidity(self) -> None:
        """Batch passive solvent and ambient-solute exchange by node/material."""
        if torch is None:
            self._exchange_humidity_scalar()
            return
        anchor_pressure = self.nodes[self.anchor_id].pressure
        heart_low = anchor_pressure < self.config.starvation_floor
        ceiling = self.config.overpressure_ceiling
        heart_high = ceiling > 0.0 and anchor_pressure > ceiling
        network_roots = self.orthogonal_network_roots()
        exchange_nodes: List[FluxNode] = []
        node_slices: List[str] = []
        radii: List[float] = []
        material_names = set(self.config.scalar_fields)
        for fields in self.config.slice_scalar_fields.values():
            material_names.update(fields)
        material_names.discard("humidity")
        materials = sorted(material_names)

        for nid, node in self.nodes.items():
            if node.burned or node.humidity_exchange <= 0.0:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            own_slice, _ = slices
            center_id = (
                node.center_id
                if node.center_id in self.nodes
                else self.anchor_id
            )
            center_level = self.nodes[center_id].level or 0
            exchange_nodes.append(node)
            node_slices.append(own_slice)
            radii.append(float(abs((node.level or 0) - center_level)))
        if not exchange_nodes:
            return

        self._emit_status(
            "humidity", "sampling ambient fields", 1, 3
        )
        ambient_humidity = []
        ambient_materials = []
        for slice_name, radius in zip(node_slices, radii):
            ambient_humidity.append(
                max(
                    0.0,
                    self._field_value(slice_name, "humidity", radius),
                )
            )
            ambient_materials.append(
                [
                    max(
                        0.0,
                        self._field_value(slice_name, name, radius),
                    )
                    for name in materials
                ]
            )

        dtype = torch.float32
        device = self.device
        solvent = torch.tensor(
            [max(0.0, node.solvent) for node in exchange_nodes],
            dtype=dtype,
            device=device,
        )
        solute_total = torch.tensor(
            [
                sum(max(0.0, value) for value in node.solubles.values())
                for node in exchange_nodes
            ],
            dtype=dtype,
            device=device,
        )
        volume = (solvent + solute_total).clamp_min(1e-12)
        water_fraction = solvent / volume
        osmotic_fraction = solute_total / volume
        humidity = torch.tensor(
            ambient_humidity, dtype=dtype, device=device
        )
        ambient = (
            torch.tensor(ambient_materials, dtype=dtype, device=device)
            if materials
            else torch.empty(
                (len(exchange_nodes), 0), dtype=dtype, device=device
            )
        )
        ambient_osmoles = ambient.sum(dim=1)
        ambient_activity = humidity / (
            humidity + ambient_osmoles
        ).clamp_min(1e-12)
        exchange_rate = torch.tensor(
            [node.humidity_exchange for node in exchange_nodes],
            dtype=dtype,
            device=device,
        )
        pressure_feature = torch.tanh(
            torch.tensor(
                [node.pressure for node in exchange_nodes],
                dtype=dtype,
                device=device,
            )
        )
        hull_features = torch.stack(
            (
                torch.ones_like(volume),
                pressure_feature,
                water_fraction,
                osmotic_fraction,
                osmotic_fraction,
                1.0 - osmotic_fraction.clamp_max(1.0),
            ),
            dim=1,
        )
        if self.config.physiology_learning_enabled:
            self._ensure_subedge_archetype()
            learned_hull = self._linear_archetype_openings(
                "node:hull", self._NODE_ARCHETYPE_FEATURES, hull_features
            )
        else:
            learned_hull = torch.ones_like(volume)
        declared_hull = torch.tensor(
            [
                max(0.0, min(1.0, node.hull_permeability))
                for node in exchange_nodes
            ],
            dtype=dtype,
            device=device,
        )
        hull_opening = declared_hull * learned_hull

        self._emit_status(
            "humidity", "evaluating pore archetype batch", 2, 3
        )
        all_materials = ["solvent", *materials]
        held = torch.zeros(
            (len(exchange_nodes), len(all_materials)),
            dtype=dtype,
            device=device,
        )
        held[:, 0] = solvent
        for column, name in enumerate(materials, start=1):
            held[:, column] = torch.tensor(
                [
                    max(0.0, node.solubles.get(name, 0.0))
                    for node in exchange_nodes
                ],
                dtype=dtype,
                device=device,
            )
        material_fraction = held / volume[:, None]
        pore_features = torch.stack(
            (
                torch.ones_like(material_fraction),
                pressure_feature[:, None].expand_as(material_fraction),
                water_fraction[:, None].expand_as(material_fraction),
                osmotic_fraction[:, None].expand_as(material_fraction),
                material_fraction,
                1.0 - material_fraction.clamp_max(1.0),
            ),
            dim=2,
        )
        if self.config.physiology_learning_enabled:
            learned_pore = self._linear_archetype_openings(
                "node:pore",
                self._NODE_ARCHETYPE_FEATURES,
                pore_features.reshape(
                    -1, len(self._NODE_ARCHETYPE_FEATURES)
                ),
            ).reshape_as(material_fraction)
        else:
            learned_pore = torch.ones_like(material_fraction)
        declared_pore = torch.tensor(
            [
                [
                    max(
                        0.0,
                        min(
                            1.0,
                            node.pore_permeabilities.get(name, 1.0),
                        ),
                    )
                    for name in all_materials
                ]
                for node in exchange_nodes
            ],
            dtype=dtype,
            device=device,
        )
        permeability = (
            hull_opening[:, None] * declared_pore * learned_pore
        )

        self._emit_status(
            "humidity", "applying exchange batch", 3, 3
        )
        target_solvent = humidity + solute_total * ambient_activity
        solvent_flow = (
            exchange_rate
            * permeability[:, 0]
            * (target_solvent - solvent)
        )
        if heart_low:
            solvent_flow = solvent_flow.clamp_min(0.0)
        elif heart_high:
            solvent_flow = solvent_flow.clamp_max(0.0)
        solvent = (solvent + solvent_flow).clamp_min(0.0)
        if materials:
            # Same rule as the scalar path: no solvent, no solute movement --
            # a dry node is not a medium anything can dissolve into or out of.
            has_water = (solvent > 0.0).to(dtype)[:, None]
            material_delta = (
                exchange_rate[:, None]
                * permeability[:, 1:]
                * (ambient - held[:, 1:])
                * has_water
            )
            new_materials = (held[:, 1:] + material_delta).clamp_min(0.0)
        else:
            new_materials = held[:, 1:]

        solvent_rows = solvent.detach().cpu().tolist()
        material_rows = new_materials.detach().cpu().tolist()
        for node, new_solvent, row in zip(
            exchange_nodes, solvent_rows, material_rows
        ):
            node.solvent = new_solvent
            for name, amount in zip(materials, row):
                node.solubles[name] = amount

    def _ingest_from_habitat_shells(self) -> None:
        """Conserved shell dispensing from the active seed's finite habitat.

        Each network-local nD habitat shell hands its own unique soluble to
        nearby nodes, at a rate set by distance in that network's dimensions.
        The backend has no mechanical coordinates, so the client publishes
        the result through the external-physics exchange (domain "client_nd",
        key "habitat_proximity": node_id -> 0..1). The projected S² radar
        position never participates in uptake.
        No client watching means no payload means no dispensing: ring
        ingestion is genuinely part of the gamified simulation, not a
        backend-only process wearing its name.

        One unique substance per pie slice (see _node_slice for naming).
        Uptake debits the same named ion from the backend-owned habitat;
        the node's opposite ion is untouched. This replaces the previous
        implicit transmutation of one ion identity into another.
        """
        client = self.external_physics.get("client_nd", {})
        proximity: Dict[int, float] = client.get("habitat_proximity") or {}
        if not proximity:
            return
        patch = self._ensure_current_habitat()
        target = max(
            0.0,
            min(1.0, float(self.config.growth_target_ion_concentration)),
        )
        uptake_rate = max(0.0, min(1.0, float(self.config.ring_uptake_rate)))
        if target <= 0.0 or uptake_rate <= 0.0:
            return
        network_roots = self.orthogonal_network_roots()
        for nid, node in self.nodes.items():
            if node.burned:
                continue
            # A habitat shell hands off a dissolved soluble -- there is
            # nothing to dissolve it into at a node holding no solvent.
            if node.solvent <= 0.0:
                continue
            near = proximity.get(nid)
            if not near:
                continue
            slices = self._node_slice(nid, node, network_roots)
            if slices is None:
                continue
            own_name, _ = slices
            available = max(0.0, patch.get(own_name, 0.0))
            if available <= 0.0:
                continue
            held = max(0.0, node.solubles.get(own_name, 0.0))
            volume = max(node.volume, 1.0)
            concentration = held / volume
            if concentration >= target:
                continue
            # Adding x solute also adds x total volume. This is the exact
            # correction needed to reach target before contact throttling.
            correction = (
                (target * volume - held) / max(1e-12, 1.0 - target)
            )
            amount = min(
                available,
                correction
                * min(1.0, max(0.0, float(near)))
                * uptake_rate,
            )
            if amount <= 0.0:
                continue
            patch[own_name] = available - amount
            node.solubles[own_name] = held + amount

    def _solve_coupled_fluid_system(self) -> None:
        """Relax nodes, path lumens, edge hulls, and spatial bath together.

        Every row is a real fluid compartment and every sparse connection is
        marshalled once, then all pressure, advection, diffusion, source
        limiting, and accumulation happens with Torch scatter operations.
        Audited tubes connect to nodes only at their terminals. Their ordered
        lumen segments remain stateful between ticks; intermediate graph
        nodes are deliberately absent from those connection lists.
        """
        if torch is None:
            # The corrected model intentionally has one implementation: a
            # vectorized coupled solve. Silently reverting to endpoint
            # teleportation would change the physical ontology.
            self.physiology_error = "PyTorch is required for coupled fluid transport"
            return

        cfg = self.config
        live_nodes = [node for node in self.nodes.values() if not node.burned]
        if not live_nodes:
            return
        self._reset_edge_flow()

        # Compatibility/global inputs enter the spatial bath at their actual
        # locality: the active seed. They cease being a globally mixed bath.
        anchor_bath = self.bath_by_node.setdefault(self.anchor_id, {})
        for name, amount in list(self.bath.items()):
            if amount:
                anchor_bath[name] = anchor_bath.get(name, 0.0) + max(0.0, float(amount))
        self.bath.clear()
        for node in live_nodes:
            self.bath_by_node.setdefault(node.id, {})

        compartments: List[Tuple[str, Any, Optional[int]]] = []
        compliance: List[float] = []
        node_comp: Dict[int, int] = {}
        bath_comp: Dict[int, int] = {}
        hull_comp: Dict[Tuple[int, int], int] = {}
        tube_comp: Dict[Tuple[int, int], int] = {}

        for node in live_nodes:
            node_comp[node.id] = len(compartments)
            compartments.append(("node", node, None))
            compliance.append(max(1e-6, float(cfg.node_compliance)))
        for node in live_nodes:
            bath_comp[node.id] = len(compartments)
            compartments.append(("bath", self.bath_by_node[node.id], None))
            compliance.append(max(1e-6, float(cfg.bath_compliance)))
        for edge_key, edge in self.edges.items():
            if edge_key[0] not in node_comp or edge_key[1] not in node_comp:
                continue
            hull_comp[edge_key] = len(compartments)
            compartments.append(("hull", edge, None))
            compliance.append(max(1e-6, float(cfg.hull_compliance)))
        live_traversals = [
            traversal for traversal in self.traversals.values()
            if traversal.start_id in node_comp and traversal.end_id in node_comp
        ]
        for traversal in live_traversals:
            for sub_index, sub in enumerate(traversal.subedges):
                sub.delivered_utility = 0.0
                for segment_index in range(len(sub.segment_edge_keys)):
                    tube_comp[(id(sub), segment_index)] = len(compartments)
                    compartments.append(("tube", sub, segment_index))
                    compliance.append(max(1e-6, float(cfg.tube_compliance)))

        names = {"solvent"}
        for kind, owner, segment_index in compartments:
            if kind == "node":
                names.update(owner.solubles)
            elif kind == "bath":
                names.update(owner)
            elif kind == "hull":
                names.update(owner.hull_solubles)
            else:
                names.update(owner.segment_solubles[segment_index])
        component_names = ["solvent"] + sorted(names - {"solvent"})
        component_index = {name: index for index, name in enumerate(component_names)}
        rows: List[List[float]] = []
        for kind, owner, segment_index in compartments:
            row = [0.0] * len(component_names)
            if kind == "node":
                row[0], solubles = owner.solvent, owner.solubles
            elif kind == "bath":
                row[0], solubles = owner.get("solvent", 0.0), owner
            elif kind == "hull":
                row[0], solubles = owner.hull_solvent, owner.hull_solubles
            else:
                row[0] = owner.segment_solvent[segment_index]
                solubles = owner.segment_solubles[segment_index]
            for name, amount in solubles.items():
                if name != "solvent" and name in component_index:
                    row[component_index[name]] = max(0.0, float(amount))
            rows.append(row)

        # Sparse arcs. Passive connections get both directions; tube arcs
        # are genuinely one-way and ordered from terminal to terminal.
        arc_src: List[int] = []
        arc_dst: List[int] = []
        arc_g: List[float] = []
        arc_edge: List[int] = []
        arc_sign: List[float] = []
        edge_items = list(self.edges.items())
        edge_index = {key: index for index, (key, _) in enumerate(edge_items)}

        def arc(src: int, dst: int, conductance: float,
                edge_key: Optional[Tuple[int, int]] = None, sign: float = 0.0) -> None:
            if conductance <= 0.0:
                return
            arc_src.append(src)
            arc_dst.append(dst)
            arc_g.append(conductance)
            arc_edge.append(edge_index.get(edge_key, -1) if edge_key else -1)
            arc_sign.append(sign)

        def passive(a: int, b: int, conductance: float,
                    edge_key: Optional[Tuple[int, int]] = None) -> None:
            sign = 1.0
            if edge_key and compartments[a][0] == "node":
                sign = 1.0 if compartments[a][1].id == edge_key[0] else -1.0
            arc(a, b, conductance, edge_key, sign)
            arc(b, a, conductance, edge_key, -sign)

        for edge_key, edge in edge_items:
            a, b = edge_key
            if edge_key not in hull_comp or a not in node_comp or b not in node_comp:
                continue
            hardening = 1.0 + cfg.branch_maturity_conductance_bonus * edge.maturity
            valve = max(0.0, float(cfg.hull_node_valve)) * hardening
            passive(node_comp[a], hull_comp[edge_key], valve, edge_key)
            passive(hull_comp[edge_key], node_comp[b], valve, edge_key)
            passive(bath_comp[a], bath_comp[b], cfg.bath_graph_conductance, edge_key)
            passive(hull_comp[edge_key], bath_comp[a], cfg.hull_bath_permeability)
            passive(hull_comp[edge_key], bath_comp[b], cfg.hull_bath_permeability)
        for node in live_nodes:
            passive(node_comp[node.id], bath_comp[node.id], cfg.node_bath_permeability)

        terminal_arcs: List[Tuple[int, SubEdge, int]] = []
        for traversal in live_traversals:
            for sub in traversal.subedges:
                count = len(sub.segment_edge_keys)
                if not count:
                    continue
                start_id = traversal.start_id if sub.direction == "forward" else traversal.end_id
                end_id = traversal.end_id if sub.direction == "forward" else traversal.start_id
                chain = [node_comp[start_id]] + [
                    tube_comp[(id(sub), i)] for i in range(count)
                ] + [node_comp[end_id]]
                gate = self._learned_subedge_opening(sub)
                for link_index, (src, dst) in enumerate(zip(chain, chain[1:])):
                    edge_key = sub.segment_edge_keys[min(link_index, count - 1)]
                    causal_sign = 1.0 if sub.direction == "forward" else -1.0
                    conductance = 1.0
                    if link_index in (0, count):
                        conductance *= gate
                    edge = self.edges.get(edge_key)
                    if edge is not None:
                        conductance *= (
                            1.0
                            + cfg.branch_maturity_conductance_bonus * edge.maturity
                        )
                    arc(src, dst, conductance, edge_key, causal_sign)
                    if link_index == count:
                        terminal_arcs.append((len(arc_src) - 1, sub, end_id))

        device, dtype = self.device, torch.float64
        initial_values = torch.tensor(rows, dtype=dtype, device=device)
        values = initial_values.clone()
        compliance_t = torch.tensor(compliance, dtype=dtype, device=device)
        if not arc_src:
            return
        src_idx = torch.tensor(arc_src, dtype=torch.long, device=device)
        dst_idx = torch.tensor(arc_dst, dtype=torch.long, device=device)
        conductance = torch.tensor(arc_g, dtype=dtype, device=device)
        dt = max(0.0, float(cfg.fluid_time_step))
        edge_flow = torch.zeros(len(edge_items), dtype=dtype, device=device)
        edge_components = torch.zeros(
            (len(edge_items), len(component_names)), dtype=dtype, device=device
        )
        delivered = torch.zeros(len(arc_src), dtype=dtype, device=device)

        def limited_transfer(raw: Any, source: Any, available: Any) -> Any:
            demand = torch.zeros_like(available)
            demand.index_add_(0, source, raw)
            scale = torch.where(
                demand > 0.0,
                torch.minimum(
                    torch.ones_like(demand),
                    available / demand.clamp_min(1e-12),
                ),
                torch.ones_like(demand),
            )
            return raw * scale.index_select(0, source)

        client_result = None
        if self.fluid_work_delegate is not None:
            packet = {
                "tick": self.tick_count,
                "components": component_names,
                "values": rows,
                "compliance": compliance,
                "compartments": [
                    (
                        {"kind": "node", "node_id": owner.id}
                        if kind == "node"
                        else {"kind": "bath", "node_id": next(
                            node_id for node_id, mixture in self.bath_by_node.items()
                            if mixture is owner
                        )}
                        if kind == "bath"
                        else {"kind": "hull", "edge": [owner.from_id, owner.to_id]}
                        if kind == "hull"
                        else {
                            "kind": "tube",
                            "traversal": list(owner.traversal_key),
                            "direction": owner.direction,
                            "segment": segment_index,
                            "edge": list(owner.segment_edge_keys[segment_index]),
                        }
                    )
                    for kind, owner, segment_index in compartments
                ],
                "arcs": {
                    "source": arc_src,
                    "destination": arc_dst,
                    "conductance": arc_g,
                    "edge": [
                        list(edge_items[index][0]) if index >= 0 else None
                        for index in arc_edge
                    ],
                    "sign": arc_sign,
                },
                "parameters": {
                    "substeps": max(1, int(cfg.fluid_solver_substeps)),
                    "time_step": dt,
                    "bulk_conductance": float(cfg.fluid_bulk_conductance),
                    "ion_diffusion": float(cfg.fluid_diffusion_conductance),
                    "osmotic_pressure": float(cfg.fluid_osmotic_pressure),
                },
            }
            client_result = self.fluid_work_delegate(packet)

        if client_result is not None:
            result_rows = client_result.get("values", [])
            arc_bulk = client_result.get("arc_bulk", [])
            arc_components = client_result.get("arc_components", [])
            expected_shape = (len(compartments), len(component_names))
            if (
                len(result_rows) != expected_shape[0]
                or any(len(row) != expected_shape[1] for row in result_rows)
                or len(arc_bulk) != len(arc_src)
                or len(arc_components) != len(arc_src)
                or any(len(row) != expected_shape[1] for row in arc_components)
            ):
                raise ValueError("client fluid result has the wrong tensor shape")
            values = torch.tensor(result_rows, dtype=dtype, device=device)
            if not bool(torch.isfinite(values).all()) or bool((values < -1e-9).any()):
                raise ValueError("client fluid result contains invalid material amounts")
            initial_total = initial_values.sum(dim=0)
            final_total = values.sum(dim=0)
            residual = float(torch.max(torch.abs(initial_total - final_total)).item())
            if residual > 1e-7:
                raise ValueError(
                    f"client fluid result violates conservation (residual {residual:.3g})"
                )
            delivered = torch.tensor(arc_bulk, dtype=dtype, device=device)
            arc_component_tensor = torch.tensor(
                arc_components, dtype=dtype, device=device
            )
            edge_ids = torch.tensor(arc_edge, dtype=torch.long, device=device)
            valid = edge_ids >= 0
            if bool(valid.any()):
                selected_edges = edge_ids[valid]
                signs = torch.tensor(arc_sign, dtype=dtype, device=device)[valid]
                edge_flow.index_add_(0, selected_edges, delivered[valid] * signs)
                edge_components.index_add_(
                    0,
                    selected_edges,
                    arc_component_tensor[valid] * signs[:, None],
                )
            self.last_fluid_proof = {
                "tick": self.tick_count,
                "worker": "client",
                "conservation_residual": residual,
                "substeps": max(1, int(cfg.fluid_solver_substeps)),
                "work_id": client_result.get("work_id"),
            }
        else:
          for _ in range(max(1, int(cfg.fluid_solver_substeps))):
            volume = values.sum(dim=1)
            solute = values[:, 1:].sum(dim=1) if values.shape[1] > 1 else torch.zeros_like(volume)
            solvent = values[:, 0]
            pressure = (
                solvent
                + cfg.fluid_osmotic_pressure
                * solute / solvent.clamp_min(1e-6)
            ) / compliance_t
            raw_bulk = (
                torch.relu(pressure.index_select(0, src_idx) - pressure.index_select(0, dst_idx))
                * conductance * cfg.fluid_bulk_conductance * dt
            )
            moved_bulk = limited_transfer(raw_bulk, src_idx, volume)
            source_values = values.index_select(0, src_idx)
            mixture = (
                source_values / source_values.sum(dim=1, keepdim=True).clamp_min(1e-12)
                * moved_bulk[:, None]
            )
            values.index_add_(0, src_idx, -mixture)
            values.index_add_(0, dst_idx, mixture)
            delivered += moved_bulk

            if values.shape[1] > 1:
                volume = values.sum(dim=1).clamp_min(1e-12)
                concentration = values[:, 1:] / volume[:, None]
                # Diffusion needs a continuous water medium on both ends --
                # a compartment holding zero solvent has nothing to dissolve
                # a solute out of, or into, no matter the concentration gap.
                has_water = values[:, 0] > 0.0
                water_gate = (
                    has_water.index_select(0, src_idx)
                    & has_water.index_select(0, dst_idx)
                ).to(dtype)[:, None]
                raw_diff = (
                    torch.relu(
                        concentration.index_select(0, src_idx)
                        - concentration.index_select(0, dst_idx)
                    )
                    * (conductance * cfg.fluid_diffusion_conductance * dt)[:, None]
                    * water_gate
                )
                source_matrix = src_idx[:, None].expand_as(raw_diff)
                demand = torch.zeros_like(values[:, 1:])
                demand.scatter_add_(0, source_matrix, raw_diff)
                scale = torch.where(
                    demand > 0.0,
                    torch.minimum(
                        torch.ones_like(demand),
                        values[:, 1:] / demand.clamp_min(1e-12),
                    ),
                    torch.ones_like(demand),
                )
                moved_diff = raw_diff * torch.gather(scale, 0, source_matrix)
                delta = torch.zeros_like(values[:, 1:])
                delta.scatter_add_(0, source_matrix, -moved_diff)
                delta.scatter_add_(0, dst_idx[:, None].expand_as(moved_diff), moved_diff)
                values[:, 1:] += delta
                mixture[:, 1:] += moved_diff

            edge_ids = torch.tensor(arc_edge, dtype=torch.long, device=device)
            valid = edge_ids >= 0
            if bool(valid.any()):
                selected_edges = edge_ids[valid]
                signs = torch.tensor(arc_sign, dtype=dtype, device=device)[valid]
                edge_flow.index_add_(0, selected_edges, moved_bulk[valid] * signs)
                edge_components.index_add_(
                    0, selected_edges, mixture[valid] * signs[:, None]
                )
          self.last_fluid_proof = {
              "tick": self.tick_count,
              "worker": "server",
              "conservation_residual": float(
                  torch.max(
                      torch.abs(initial_values.sum(dim=0) - values.sum(dim=0))
                  ).item()
              ),
              "substeps": max(1, int(cfg.fluid_solver_substeps)),
          }

        values = values.clamp_min(0.0)
        final_volume = values.sum(dim=1)
        final_pressure = (
            values[:, 0]
            + cfg.fluid_osmotic_pressure
            * values[:, 1:].sum(dim=1) / values[:, 0].clamp_min(1e-6)
        ) / compliance_t
        cpu_values = values.detach().cpu().tolist()
        cpu_pressure = final_pressure.detach().cpu().tolist()
        for row_index, ((kind, owner, segment_index), row, pressure_value) in enumerate(
            zip(compartments, cpu_values, cpu_pressure)
        ):
            solubles = {
                name: row[column]
                for column, name in enumerate(component_names[1:], start=1)
                if row[column] > 1e-12
            }
            if kind == "node":
                owner.solvent, owner.solubles, owner.pressure = row[0], solubles, pressure_value
            elif kind == "bath":
                owner.clear()
                if row[0] > 1e-12:
                    owner["solvent"] = row[0]
                owner.update(solubles)
            elif kind == "hull":
                owner.hull_solvent, owner.hull_solubles, owner.hull_pressure = row[0], solubles, pressure_value
            else:
                owner.segment_solvent[segment_index] = row[0]
                owner.segment_solubles[segment_index] = solubles
                owner.segment_pressures[segment_index] = pressure_value

        delivered_cpu = delivered.detach().cpu().tolist()
        for arc_index, sub, end_id in terminal_arcs:
            destination = self.nodes[end_id]
            before = max(destination.volume - delivered_cpu[arc_index], 1e-12)
            need = max(
                0.0,
                cfg.growth_target_ion_concentration
                - sum(destination.solubles.values()) / before,
            )
            sub.delivered_utility = delivered_cpu[arc_index] * need
        flow_rows = edge_flow.detach().cpu().tolist()
        component_rows = edge_components.detach().cpu().tolist()
        for (_, edge), flow, row in zip(edge_items, flow_rows, component_rows):
            edge.flow = flow
            edge.component_flows = {
                name: row[column]
                for column, name in enumerate(component_names)
                if abs(row[column]) > 1e-12
            }

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
        self._solve_coupled_fluid_system()
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

    def _update_branch_maturity(self) -> None:
        """Harden branches only when their tubes relieve terminal scarcity."""
        retention = max(
            0.0, min(1.0, float(self.config.branch_maturity_retention))
        )
        gain = max(0.0, float(self.config.branch_maturity_gain))
        utility_by_edge: Dict[Tuple[int, int], float] = {}
        for traversal in self.traversals.values():
            for sub in traversal.subedges:
                if sub.delivered_utility <= 0.0:
                    continue
                for edge_key in sub.segment_edge_keys:
                    utility_by_edge[edge_key] = (
                        utility_by_edge.get(edge_key, 0.0)
                        + sub.delivered_utility
                    )
        for edge_key, edge in self.edges.items():
            signal = 1.0 - math.exp(-utility_by_edge.get(edge_key, 0.0))
            edge.maturity = max(
                0.0,
                min(1.0, retention * edge.maturity + gain * signal),
            )

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
        """Execute configured node roles against circulation and/or CSF.

        A factory's own recipe only declares the specific inputs it
        consumes -- most don't name "solvent" as one of them, so nothing
        here would otherwise stop a reaction from running in a mixture
        that happens to hold zero water. Metabolism needs a medium to
        happen in: no water in the relevant compartment means that
        compartment's reactions simply don't run this tick, regardless of
        whether its other named inputs are present.
        """
        for node in self.nodes.values():
            node.factory_auxin = 0.0
            if node.burned:
                continue
            circulation = dict(node.solubles)
            circulation["solvent"] = node.solvent
            local_bath = self.bath_by_node.setdefault(node.id, {})
            circulation_has_water = circulation.get("solvent", 0.0) > 0.0
            bath_has_water = local_bath.get("solvent", 0.0) > 0.0
            changed_circulation = False
            for factory in node.factories:
                if not factory.enabled or factory.throughput <= 0.0:
                    continue
                medium = factory.medium.lower()
                remaining = factory.throughput
                if medium in ("circulatory", "both") and circulation_has_water:
                    used = self._run_factory_reaction(
                        node, factory, circulation, remaining, local_bath
                    )
                    remaining -= used
                    changed_circulation = changed_circulation or used > 0.0
                if medium in ("csf", "both") and remaining > 0.0 and bath_has_water:
                    self._run_factory_reaction(
                        node, factory, local_bath, remaining, local_bath
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
            heart.attach(
                "csf_link", when="post", scope="total",
                fn=lambda chambers, region=region:
                    self._csf_link_hook(chambers, region),
            )
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

    def _csf_link_hook(
        self, chambers: Dict[str, Dict[str, float]], region: str = "main"
    ) -> None:
        """Exchange every chamber's contents with the CSF bath toward
        equalizing concentration, throttled by config.csf_link_rate.
        Dormant (a no-op) while the rate is 0."""
        rate = self.config.csf_link_rate
        if rate <= 0.0:
            return
        heart = self.hearts.get(region)
        locality = (
            heart.seed_owner_id if heart is not None
            and heart.seed_owner_id in self.nodes else self.anchor_id
        )
        local_bath = self.bath_by_node.setdefault(locality, {})
        for mix in chambers.values():
            names = set(mix) | set(local_bath)
            for name in names:
                delta = rate * (
                    mix.get(name, 0.0) - local_bath.get(name, 0.0)
                )
                if delta == 0.0:
                    continue
                mix[name] = mix.get(name, 0.0) - delta
                local_bath[name] = local_bath.get(name, 0.0) + delta

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
        local_bath = self.bath_by_node.setdefault(self.anchor_id, {})
        for name, amount in list(self.bath.items()):
            if amount:
                local_bath[name] = local_bath.get(name, 0.0) + amount
        self.bath.clear()
        for name, amount in list(local_bath.items()):
            if name == "solvent" or amount <= 0.0:
                continue
            moved = amount * rate
            local_bath[name] = amount - moved
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
        locality = (
            heart.seed_owner_id
            if heart.seed_owner_id in self.nodes else self.anchor_id
        )
        local_bath = self.bath_by_node.setdefault(locality, {})
        for mixture in heart.chambers.values():
            for name, amount in mixture.items():
                if amount:
                    local_bath[name] = local_bath.get(name, 0.0) + amount
        for reservoir in heart.reservoirs.values():
            for name, amount in reservoir.drain().items():
                if amount:
                    local_bath[name] = local_bath.get(name, 0.0) + amount

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
        inflow_routes: Dict[str, List[Tuple[SubEdge, int]]] = {}
        outflow_routes: Dict[str, List[Tuple[SubEdge, int]]] = {}
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
                # The heart touches the lumen cross-section at its own
                # terminal. It never drains or fills the remote endpoint.
                segment_index = (
                    len(sub.segment_edge_keys) - 1 if is_inflow else 0
                )
                if segment_index < 0:
                    continue
                routes = inflow_routes if is_inflow else outflow_routes
                routes.setdefault(slices[0], []).append((sub, segment_index))

        if not inflow_routes and not outflow_routes and region not in self.hearts:
            return  # nothing to pump and no heart yet -- don't materialize one

        pump = self.nodes[pump_id]
        heart = self._configure_seed_heart(region, pump_id, initially_full=False)
        for slice_name in outflow_routes:
            heart.chamber(slice_name, "out")  # chamber exists even before anything reaches it
        for slice_name in inflow_routes:
            heart.chamber(slice_name, "in")  # ditto, so a brand-new slice can still pull this same beat

        # Intake is the pump's inhale, not a passive equalization: the same
        # beat-driven contraction fraction that governs how hard an
        # out-chamber pushes into the network now governs how hard an
        # in-chamber pulls from it (see _squeeze_out_chambers -- "when it
        # pushes, it pushes"; this is the same mechanic in reverse). No
        # phase this tick means no beat at all, so nothing pulls or pushes.
        phase = heart.script[heart.phase_index % len(heart.script)] if heart.script else None
        contractions = heart._resolve_contractions(phase) if phase is not None else {}
        for slice_name, routes in inflow_routes.items():
            chamber = heart.chamber(slice_name, "in")
            fraction = contractions.get(f"{slice_name}|in", 0.0)
            plan = self._chamber_intake_plan(routes, fraction)
            pooled = self._drain_plan(plan)
            for name, amount in pooled.items():
                chamber[name] = chamber.get(name, 0.0) + amount

        heart._run_hooks("pre")
        heart.exchange_seed_reservoirs()
        self._permeate_heart_forward_to_background(heart)
        if phase is not None:
            valves = heart._resolve_valves(phase)
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
        local_bath = self.bath_by_node.setdefault(self.anchor_id, {})
        if rate <= 0.0 or not local_bath:
            return
        heart = self.hearts.get("main")
        if heart is None:
            return
        in_keys = [k for k in heart.chambers if k.endswith("|in")]
        if not in_keys:
            return
        share = 1.0 / len(in_keys)
        for name, amount in list(local_bath.items()):
            drained = amount * rate
            if drained == 0.0:
                continue
            local_bath[name] = amount - drained
            for key in in_keys:
                heart.chambers[key][name] = heart.chambers[key].get(name, 0.0) + drained * share

    def _squeeze_out_chambers(
        self,
        phase: HeartPhase,
        contractions: Dict[str, float],
        outflow_routes: Dict[str, List[Tuple["SubEdge", int]]],
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
            for sub, segment_index in routes:
                opening = self._learned_subedge_opening(sub)
                self._deposit_segment_mixture(
                    sub, segment_index, expelled,
                    scale=opening / total_constriction,
                )

    def _chamber_intake_plan(
        self, inflow: List[Tuple["SubEdge", int]], fraction: float
    ) -> List[Tuple[SubEdge, int, float]]:
        """What one chamber's inflow subedges deliver this beat: an active
        pull, not a passive pressure/volume comparison.

        ``fraction`` is this beat's in-chamber contraction (see
        _resolve_contractions) -- the exact same number that governs how
        hard the matching out-chamber pushes into the network this same
        beat (_squeeze_out_chambers), applied here in reverse: the pump
        draws that fraction of whatever is actually sitting in each open
        segment, unconditionally. A real pump doesn't check whether its
        supply line has "enough pressure relative to me" before it
        inhales -- it inhales, and whatever's there comes. (An earlier
        version compared the segment against the chamber's, then the pump
        node's, own volume -- both were passive equalizations disguised
        as intake, and the node comparison in particular was a bar the
        segment could structurally never clear, since the pump node's own
        solvent/solubles are never touched by heart transport at all.)
        Purely read-only so every chamber can be measured from the same
        pre-beat snapshot before anything is drained.
        """
        plan = []
        if fraction <= 0.0:
            return plan
        pull = min(1.0, fraction)
        for sub, segment_index in inflow:
            segment_volume = (
                sub.segment_solvent[segment_index]
                + sum(sub.segment_solubles[segment_index].values())
            )
            if segment_volume <= 0.0:
                continue
            amount = self._learned_subedge_opening(sub) * segment_volume * pull
            if amount > 0.0:
                plan.append((sub, segment_index, amount))
        return plan

    def _drain_plan(
        self, plan: List[Tuple[SubEdge, int, float]]
    ) -> Dict[str, float]:
        """Apply a measured intake plan, pooling every drained mix
        (solvent and solubles together) into one chamber-load."""
        pooled: Dict[str, float] = {}
        for sub, segment_index, amount in plan:
            for name, moved in self._drain_segment_mixture(
                sub, segment_index, amount
            ).items():
                pooled[name] = pooled.get(name, 0.0) + moved
        return pooled

    @staticmethod
    def _drain_segment_mixture(
        sub: SubEdge, segment_index: int, amount: float
    ) -> Dict[str, float]:
        solvent = sub.segment_solvent[segment_index]
        solubles = sub.segment_solubles[segment_index]
        total = solvent + sum(solubles.values())
        if total <= 0.0 or amount <= 0.0:
            return {}
        fraction = min(1.0, amount / total)
        mixture = {"solvent": solvent * fraction}
        sub.segment_solvent[segment_index] -= mixture["solvent"]
        for name, held in list(solubles.items()):
            moved = held * fraction
            solubles[name] = held - moved
            if moved:
                mixture[name] = moved
        return mixture

    @staticmethod
    def _deposit_segment_mixture(
        sub: SubEdge, segment_index: int, mixture: Dict[str, float],
        scale: float = 1.0,
    ) -> None:
        sub.segment_solvent[segment_index] += (
            mixture.get("solvent", 0.0) * scale
        )
        solubles = sub.segment_solubles[segment_index]
        for name, amount in mixture.items():
            if name == "solvent":
                continue
            solubles[name] = solubles.get(name, 0.0) + amount * scale

