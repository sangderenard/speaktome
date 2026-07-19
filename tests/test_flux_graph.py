"""Tests for speaktome.core.flux_graph."""

import math

import torch

from tensors import AbstractTensor
from tensors.torch_backend import PyTorchTensorOperations
from speaktome.core.model_abstraction import (
    AbstractModelWrapper,
    PyTorchModelWrapper,
)
from speaktome.core.choice_policy import TopKPolicy
from speaktome.core.implicit_backpath import ImplicitBackpathScorer
from speaktome.core.noodle_explorer import Direction
from speaktome.core.flux_graph import (
    Channel,
    Edge,
    FluxGraph,
    FluxGraphConfig,
    FluxNode,
    IonReservoir,
    MaterialFactory,
)
from speaktome.core.poetic_attractor import PoeticAttractor


VOCAB = 5


class BigramDummyModel(AbstractModelWrapper):
    """logits at each position depend only on the token at that position."""

    def __init__(self, table):
        self.table = table

    def forward(self, input_ids, attention_mask, **kwargs):
        table_t = torch.tensor(self.table, dtype=torch.float32)
        return {"logits": table_t[input_ids]}

    def get_device(self):
        return "cpu"


# token i is strongly followed by token (i+1) % VOCAB, and nothing else.
def _cycle_table(vocab=VOCAB, peak=10.0):
    table = []
    for i in range(vocab):
        row = [0.0] * vocab
        row[(i + 1) % vocab] = peak
        table.append(row)
    return table


class FakeTokenizer:
    vocab_size = VOCAB

    def decode(self, ids):
        return f"<{ids[0]}>"


def _build_graph(branch_factor=2, compute_budget=2):
    table = _cycle_table()
    model = BigramDummyModel(table)
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=branch_factor, compute_budget_per_tick=compute_budget)
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    return graph


def test_seed_creates_anchor():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    assert graph.anchor_id == anchor_id
    anchor = graph.nodes[anchor_id]
    assert anchor.tokens == []
    assert anchor.direction is None
    assert anchor.local_evidence == 0.0
    assert anchor.pressure > 0


def test_tick_publishes_aggregate_mid_tick_status_without_graph_objects():
    graph = _build_graph(branch_factor=1, compute_budget=1)
    graph.seed([0])
    graph.spawn_first_children()
    statuses = []
    graph.set_status_callback(statuses.append)

    graph.tick()

    phases = [status["phase"] for status in statuses]
    assert phases[0] == "settling"
    assert "model_inference" in phases
    assert "auditing" in phases
    assert phases[-1] == "complete"
    assert statuses[-1]["tick"] == 1
    assert statuses[-1]["current"] == statuses[-1]["total"] == 1
    assert statuses[-1]["live_nodes"] > 0
    assert statuses[-1]["elapsed_seconds"] >= 0
    assert all("nodes" not in status and "edges" not in status for status in statuses)


def test_initial_seed_reservoirs_are_full_at_one_ion_unit_per_token():
    graph = _build_graph()
    anchor_id = graph.seed([0, 1, 2])

    heart = graph.hearts["main"]
    assert heart.seed_owner_id == anchor_id
    assert set(heart.reservoirs) == {"main:forward", "main:backward"}
    for reservoir in heart.reservoirs.values():
        assert reservoir.design_storage == 3.0
        assert reservoir.ion_amount == 3.0
        assert reservoir.storage_volume == 3.0
        assert reservoir.fullness == 1.0
        assert reservoir.concentration_band() == (1.0, 1.0, 1.0)


def test_seed_reservoir_gate_supplies_and_skims_ions_bidirectionally():
    reservoir = IonReservoir(
        ion_name="main:forward",
        design_storage=2.0,
        ion_amount=2.0,
        concentration_window=[0.5, 0.5],
        window_size=4,
    )
    deficient = {"solvent": 1.0}

    reservoir.exchange_with(deficient)

    assert deficient["main:forward"] > 0.0
    supplied_balance = reservoir.ion_amount

    saturated = {"main:forward": 1.0}
    reservoir.exchange_with(saturated)

    assert saturated["main:forward"] < 1.0
    assert reservoir.ion_amount > supplied_balance
    assert reservoir.storage_volume >= reservoir.mixture_volume


def test_seed_displacement_dumps_all_old_heart_contents_to_csf():
    graph = _build_graph(branch_factor=1)
    old_anchor_id = graph.seed([0, 1, 2])
    old_heart = graph.hearts["main"]
    old_heart.chamber("main:forward", "in").update({"solvent": 2.0, "salt": 0.5})
    graph._attach_forward_children(old_anchor_id, [-0.1], [[3]])
    new_anchor_id = graph.nodes[old_anchor_id].children_ids[-1]

    graph._reroot(new_anchor_id)

    heart = graph.hearts["main"]
    assert heart.seed_owner_id == new_anchor_id
    assert heart.chambers == {}
    assert all(reservoir.ion_amount == 0.0 for reservoir in heart.reservoirs.values())
    assert all(reservoir.design_storage == 1.0 for reservoir in heart.reservoirs.values())
    assert graph.bath["main:forward"] == 3.0
    assert graph.bath["main:backward"] == 3.0
    assert graph.bath["solvent"] == 2.0
    assert graph.bath["salt"] == 0.5


def test_off_seed_tier_heart_is_spilled_and_removed():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    graph._attach_forward_children(seed_id, [-0.1], [[1]])
    forward_id = graph.nodes[seed_id].children_ids[-1]
    stray = graph._configure_seed_heart(f"net:{forward_id}", forward_id, initially_full=False)
    stray.chamber("stray:forward", "in")["solvent"] = 1.25
    stray.reservoirs[f"net:{forward_id}:forward"].ion_amount = 0.75

    pumps = graph._region_pump_nodes()
    graph._reap_dead_hearts(set(pumps))

    assert pumps == {"main": seed_id}
    assert f"net:{forward_id}" not in graph.hearts
    assert graph.bath["solvent"] == 1.25
    assert graph.bath[f"net:{forward_id}:forward"] == 0.75

def test_seed_reservoir_state_round_trips_with_fluid_persistence():
    graph = _build_graph()
    graph.seed([0, 1])
    reservoir = graph.hearts["main"].reservoirs["main:forward"]
    reservoir.ion_amount = 1.25
    reservoir.solvent = 0.75
    reservoir.concentration_window = [0.4, 0.5, 0.6]
    graph.background["main:forward"] = 0.3
    graph.soil["main:forward"] = 0.1
    graph.rhizome["waste:salt"] = 0.4

    state = graph.export_fluid_state()
    restored = _build_graph()
    restored.seed([0, 1])
    restored.import_fluid_state(state)

    restored_reservoir = restored.hearts["main"].reservoirs["main:forward"]
    assert restored.hearts["main"].seed_owner_id == graph.anchor_id
    assert restored_reservoir.ion_amount == 1.25
    assert restored_reservoir.solvent == 0.75
    assert restored_reservoir.concentration_window == [0.4, 0.5, 0.6]
    assert restored.background == {"main:forward": 0.3}
    assert restored.soil == {"main:forward": 0.1}
    assert restored.rhizome == {"waste:salt": 0.4}
    assert restored.rhizome_owner_id == restored.anchor_id


def test_generic_node_factory_consumes_declared_materials_in_declared_medium():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    node = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]
    node.solubles = {"feedstock": 2.0}
    node.factories = [
        MaterialFactory(
            name="auxin synthesis",
            inputs={"feedstock": 2.0},
            outputs={"auxin": 0.5, "byproduct": 1.0},
            medium="circulatory",
            throughput=1.0,
        )
    ]

    graph._run_node_factories()

    assert node.solubles.get("feedstock", 0.0) == 0.0
    assert node.solubles["byproduct"] == 1.0
    assert node.factory_auxin == 0.5


def test_material_scarcity_drives_fractional_growth_interest():
    graph = _build_graph(branch_factor=1)
    graph.config.growth_target_ion_concentration = 0.2
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    node = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]
    node.solvent = 9.0
    node.solubles["main:backward"] = 1.0

    graph._update_nutrient_growth_interest()

    assert math.isclose(node.backward_growth_interest, 0.5)
    node.solubles["main:backward"] = 3.0
    graph._update_nutrient_growth_interest(accumulate=False)
    assert node.backward_growth_interest == 0.0


def test_factory_waste_dumps_into_global_csf():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    node = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]
    node.solubles = {"feedstock": 2.0}
    node.factories = [
        MaterialFactory(
            name="metabolism",
            inputs={"feedstock": 1.0},
            outputs={"useful": 0.5},
            waste_outputs={"waste:salt": 0.25},
            throughput=2.0,
        )
    ]

    graph._run_node_factories()

    assert node.solubles["useful"] == 1.0
    assert graph.bath["waste:salt"] == 0.5


def test_all_cousin_hearts_share_one_global_csf_bath():
    graph = _build_graph(branch_factor=1)
    graph.config.csf_link_rate = 0.5
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    forward_id = graph.nodes[anchor_id].children_ids[-1]
    graph._attach_backward_parents(forward_id, [-0.1], [[2]])
    cousin_id = [nid for nid in graph.nodes[forward_id].parent_ids if nid != anchor_id][0]
    cousin_region = f"net:{cousin_id}"
    main = graph.hearts["main"]
    cousin = graph.hearts[cousin_region]
    main.chamber("main:forward", "out")["main:forward"] = 2.0
    cousin.chamber(f"{cousin_region}:backward", "out")[f"{cousin_region}:backward"] = 2.0

    main._run_hooks("post")
    cousin._run_hooks("post")

    assert graph.bath["main:forward"] > 0.0
    assert graph.bath[f"{cousin_region}:backward"] > 0.0


def test_active_seed_owns_one_conserved_rhizome_and_exudes_soil_salts():
    graph = _build_graph(branch_factor=1)
    graph.config.rhizome_csf_pump_rate = 0.5
    graph.config.rhizome_soil_exudation_rate = 0.2
    anchor_id = graph.seed([0])
    graph.bath = {"solvent": 2.0, "main:forward": 10.0, "waste:salt": 4.0}

    graph._pump_csf_to_rhizome()
    graph._exude_rhizome_to_soil()

    assert graph.rhizome_owner_id == anchor_id
    assert graph.bath == {"solvent": 2.0, "main:forward": 5.0, "waste:salt": 2.0}
    assert graph.rhizome == {"main:forward": 4.0, "waste:salt": 1.6}
    assert graph.soil == {"main:forward": 1.0, "waste:salt": 0.4}
    graph._attach_forward_children(anchor_id, [-0.1], [[3]])
    new_anchor_id = graph.nodes[anchor_id].children_ids[-1]
    stored = dict(graph.rhizome)

    graph._reroot(new_anchor_id)

    assert graph.rhizome_owner_id == new_anchor_id
    assert graph.rhizome == stored

def test_level_zero_forward_supply_reaches_roots_through_reduced_soil_boundary():
    graph = _build_graph(branch_factor=2)
    graph.config.level_zero_background_permeability = 0.1
    graph.config.soil_forward_ion_permeability = 0.25
    graph.config.root_soil_uptake_permeability = 1.0
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    backward_nodes = [
        graph.nodes[parent_id] for parent_id in graph.nodes[anchor_id].parent_ids
    ]
    heart = graph.hearts["main"]
    heart.chamber("main:forward", "out")["main:forward"] = 100.0

    graph._permeate_heart_forward_to_background(heart)
    graph._permeate_background_into_soil()
    graph._absorb_soil_by_roots()

    assert math.isclose(graph.background["main:forward"], 7.5)
    assert math.isclose(graph.soil["main:forward"], 0.0)
    assert math.isclose(
        sum(node.solubles.get("main:forward", 0.0) for node in backward_nodes),
        2.5,
    )


def test_level_zero_background_material_can_permeate_back_into_circulation():
    graph = _build_graph()
    graph.config.level_zero_background_permeability = 0.1
    graph.seed([0])
    heart = graph.hearts["main"]
    heart.chamber("main:forward", "out")
    graph.background["main:forward"] = 10.0

    graph._permeate_heart_forward_to_background(heart)

    assert math.isclose(graph.background["main:forward"], 9.0)
    assert math.isclose(
        heart.chambers["main:forward|out"]["main:forward"], 1.0
    )


def test_humidity_dissolved_materials_permeate_and_raise_osmotic_water_demand():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    node = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]
    graph.config.scalar_fields = {
        "humidity": lambda _radius: 0.5,
        "ambient_salt": lambda _radius: 0.5,
    }

    graph._exchange_humidity()
    first_solvent = node.solvent
    graph._exchange_humidity()

    assert node.solubles["ambient_salt"] == 0.5
    assert first_solvent == 0.5
    assert node.solvent > first_solvent


def test_spawn_first_growth_creates_backward_parents_and_forward_children():
    graph = _build_graph(branch_factor=3)
    graph.seed([0])
    graph.spawn_first_children()

    anchor = graph.nodes[graph.anchor_id]
    children = [graph.nodes[c] for c in anchor.children_ids]
    parents = [graph.nodes[p] for p in anchor.parent_ids]

    assert len(children) == 3
    assert len(parents) == 3
    assert {c.direction for c in children} == {Direction.FORWARD}
    assert {p.direction for p in parents} == {Direction.BACKWARD}
    # The cycle table makes token 1 the overwhelmingly likely forward pick after 0.
    fwd_children = [c for c in children if c.direction is Direction.FORWARD]
    best_fwd = max(fwd_children, key=lambda c: c.local_evidence)
    assert best_fwd.tokens[0] == 1


def test_growth_creates_a_real_edge_object_in_causal_order():
    graph = _build_graph(branch_factor=3)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()

    anchor = graph.nodes[anchor_id]
    assert len(anchor.children_ids) == 3
    assert len(anchor.parent_ids) == 3
    for child_id in anchor.children_ids:
        edge = graph.edges.get((anchor_id, child_id))
        assert edge is not None
        assert edge.from_id == anchor_id
        assert edge.to_id == child_id
    for parent_id in anchor.parent_ids:
        edge = graph.edges.get((parent_id, anchor_id))
        assert edge is not None
        assert edge.from_id == parent_id
        assert edge.to_id == anchor_id


def test_edge_formation_is_postfix_beam_for_forward_growth_and_prefix_beam_for_backward():
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()

    anchor = graph.nodes[anchor_id]
    for child_id in anchor.children_ids:
        edge = graph.edges[(anchor_id, child_id)]
        assert edge.formation == "postfix_beam"
    for parent_id in anchor.parent_ids:
        assert graph.edges[(parent_id, anchor_id)].formation == "prefix_beam"


def test_backward_branching_gives_one_node_simultaneous_causal_parents():
    graph = _build_graph(branch_factor=3)
    anchor_id = graph.seed([0])

    graph._attach_backward_parents(anchor_id, [-0.1, -0.2, -0.3], [[1], [2], [3]])

    anchor = graph.nodes[anchor_id]
    assert len(anchor.parent_ids) == 3
    for parent_id in anchor.parent_ids:
        assert anchor_id in graph.nodes[parent_id].children_ids
        assert graph.nodes[parent_id].level == -1
        assert (parent_id, anchor_id) in graph.edges


def test_auditor_threads_every_parent_tree_through_the_seed_layer():
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph._attach_backward_parents(anchor_id, [-0.1, -0.2], [[1], [2]])
    graph._attach_forward_children(anchor_id, [-0.3], [[3]])
    child_id = graph.nodes[anchor_id].children_ids[-1]

    graph._run_graph_auditor()

    for parent_id in graph.nodes[anchor_id].parent_ids:
        traversal = graph.traversals[(parent_id, child_id)]
        assert traversal.node_ids == [parent_id, anchor_id, child_id]


def test_reciprocal_ion_shortage_accumulates_center_seeking_interest():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    graph._attach_backward_parents(anchor_id, [-0.1], [[2]])
    forward_id = graph.nodes[anchor_id].children_ids[-1]
    backward_id = graph.nodes[anchor_id].parent_ids[-1]

    graph._update_nutrient_growth_interest()
    assert graph.nodes[forward_id].backward_growth_interest == 1.0
    assert graph.nodes[backward_id].forward_growth_interest == 1.0

    graph.nodes[forward_id].solubles["main:backward"] = 1.0
    graph.nodes[backward_id].solubles["main:forward"] = 1.0
    graph._update_nutrient_growth_interest()
    assert graph.nodes[forward_id].backward_growth_interest == 0.0
    assert graph.nodes[backward_id].forward_growth_interest == 0.0


def test_nutrient_reconciliation_clears_supply_without_double_counting_shortage():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    forward = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]

    graph._update_nutrient_growth_interest()
    assert forward.backward_growth_interest == 1.0
    graph._update_nutrient_growth_interest(accumulate=False)
    assert forward.backward_growth_interest == 1.0

    forward.solubles["main:backward"] = 0.5
    graph._update_nutrient_growth_interest(accumulate=False)
    assert forward.backward_growth_interest == 0.0


def test_edge_records_water_and_each_ion_flow_separately():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    child_id = graph.nodes[anchor_id].children_ids[-1]
    edge = graph.edges[(anchor_id, child_id)]

    graph._record_edge_flow(
        [anchor_id, child_id], downward=True, amount=3.0,
        mixture={"solvent": 2.0, "main:forward": 0.75, "salt": 0.25},
    )
    assert edge.flow == 3.0
    assert edge.component_flows == {
        "solvent": 2.0, "main:forward": 0.75, "salt": 0.25,
    }

    graph._reset_edge_flow()
    assert edge.flow == 0.0
    assert edge.component_flows == {}

def test_each_named_ion_diffuses_osmotically_and_counterflows_without_bulk_current():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    start_id = graph.nodes[anchor_id].children_ids[-1]
    graph._attach_forward_children(start_id, [-0.1], [[2]])
    end_id = graph.nodes[start_id].children_ids[-1]
    start = graph.nodes[start_id]
    end = graph.nodes[end_id]
    start.pressure = end.pressure = 1.0
    start.solvent = end.solvent = 9.0
    start.solubles = {"main:forward": 1.0, "main:backward": 0.0}
    end.solubles = {"main:forward": 0.0, "main:backward": 1.0}
    graph._run_graph_auditor()

    graph._transport_subedges()

    assert math.isclose(start.solubles["main:forward"], 0.5)
    assert math.isclose(end.solubles["main:forward"], 0.5)
    assert math.isclose(start.solubles["main:backward"], 0.5)
    assert math.isclose(end.solubles["main:backward"], 0.5)
    edge = graph.edges[(start_id, end_id)]
    assert math.isclose(edge.flow, 0.0)
    assert math.isclose(edge.component_flows["main:forward"], 0.5)
    assert math.isclose(edge.component_flows["main:backward"], -0.5)

def test_tick_one_heart_count_uses_safe_cross_growth_defaults():
    graph = _build_graph(branch_factor=VOCAB, compute_budget=2)
    graph.config.graph_auditor_enabled = True
    graph.seed([0])
    graph.spawn_first_children()

    graph.tick()

    assert len(graph.hearts) == 1 + graph.config.compute_budget_per_tick


def test_air_root_width_and_depth_are_separate_from_backward_beam_controls():
    graph = _build_graph(branch_factor=1)
    graph.config.backward_branch_factor = VOCAB
    graph.config.backward_hot_loop_depth = 7
    graph.config.air_root_branch_factor = 2
    graph.config.air_root_hot_loop_depth = 2
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    source = graph.nodes[graph.nodes[anchor_id].children_ids[-1]]

    graph._expand_nutrient_hot_loop(source, Direction.BACKWARD)

    assert sum(node.level == 0 for node in graph.nodes.values()) == 3
    assert sum(node.level == -1 for node in graph.nodes.values()) == 4
    assert len(graph.hearts) == 3


def test_sprout_width_and_depth_are_separate_from_forward_beam_controls():
    graph = _build_graph(branch_factor=1)
    graph.config.forward_branch_factor = VOCAB
    graph.config.forward_hot_loop_depth = 7
    graph.config.sprout_branch_factor = 2
    graph.config.sprout_hot_loop_depth = 2
    anchor_id = graph.seed([0])
    graph._attach_backward_parents(anchor_id, [-0.1], [[1]])
    source = graph.nodes[graph.nodes[anchor_id].parent_ids[-1]]

    graph._expand_nutrient_hot_loop(source, Direction.FORWARD)

    assert sum(node.level == 0 for node in graph.nodes.values()) == 3
    assert sum(node.level == 1 for node in graph.nodes.values()) == 4
    assert len(graph.hearts) == 3


def test_any_needy_node_can_launch_cross_growth():
    graph = _build_graph(branch_factor=VOCAB, compute_budget=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    internal_id = graph.nodes[anchor_id].children_ids[-1]
    graph._attach_forward_children(internal_id, [-0.1], [[2]])
    leaf_id = graph.nodes[internal_id].children_ids[-1]
    graph._attach_backward_parents(anchor_id, [-0.1], [[3]])
    graph.nodes[internal_id].backward_growth_interest = 1_000.0
    graph.nodes[leaf_id].backward_growth_interest = 100.0
    graph.config.forward_budget_per_tick = 0
    graph.config.backward_budget_per_tick = 1
    calls = []

    def record_cross_growth(node, direction):
        calls.append((node.id, direction))

    graph._expand_nutrient_hot_loop = record_cross_growth

    graph._expand_top_pressure_nodes()

    assert calls == [(internal_id, Direction.BACKWARD)]

def test_backward_growth_from_forward_level_one_creates_a_cousin_center_with_a_heart():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_forward_children(anchor_id, [-0.1], [[1]])
    forward_id = graph.nodes[anchor_id].children_ids[-1]

    graph._attach_backward_parents(forward_id, [-0.2], [[2]])

    cousin_id = graph.nodes[forward_id].parent_ids[-1]
    cousin = graph.nodes[cousin_id]
    assert cousin.level == 0
    assert cousin.direction is None
    assert cousin.center_id == cousin_id
    assert set(graph.nodes[forward_id].parent_ids) == {anchor_id, cousin_id}
    assert f"net:{cousin_id}" in graph.hearts
    assert graph._region_pump_nodes()[f"net:{cousin_id}"] == cousin_id


def test_forward_growth_from_backward_level_one_creates_a_cousin_center_with_a_heart():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph._attach_backward_parents(anchor_id, [-0.1], [[1]])
    backward_id = graph.nodes[anchor_id].parent_ids[-1]

    graph._attach_forward_children(backward_id, [-0.2], [[2]])

    cousin_id = graph.nodes[backward_id].children_ids[-1]
    cousin = graph.nodes[cousin_id]
    assert cousin.level == 0
    assert cousin.direction is None
    assert cousin.center_id == cousin_id
    assert set(graph.nodes[backward_id].children_ids) == {anchor_id, cousin_id}
    assert cousin.parent_ids == [backward_id]
    assert f"net:{cousin_id}" in graph.hearts
    assert graph._region_pump_nodes()[f"net:{cousin_id}"] == cousin_id


def test_edge_records_the_seed_at_formation_time_permanently():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    fwd_child_id = next(c for c in graph.nodes[anchor_id].children_ids if graph.nodes[c].direction is Direction.FORWARD)
    edge = graph.edges[(anchor_id, fwd_child_id)]
    assert edge.seed_id_at_formation == anchor_id

    # Re-root away from anchor_id -- the edge's own record must not change,
    # even though anchor_id is no longer graph.anchor_id.
    graph._reroot(fwd_child_id)
    assert graph.anchor_id != anchor_id
    assert graph.edges[(anchor_id, fwd_child_id)].seed_id_at_formation == anchor_id


def test_edge_conductance_matches_directional_conductance_at_creation():
    graph = _build_graph(branch_factor=1)
    graph.config.return_conductance_scale = 0.4
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    child_id = graph.nodes[anchor_id].children_ids[0]
    edge = graph.edges[(anchor_id, child_id)]

    assert math.isclose(edge.forward.conductance, graph._directional_conductance(child_id, anchor_id))
    assert math.isclose(edge.reverse.conductance, graph._directional_conductance(anchor_id, child_id))
    # forward (delivery) is unscaled, reverse (return) is scaled down --
    # they should differ given return_conductance_scale != 1.0.
    assert edge.forward.conductance != edge.reverse.conductance


def test_edge_channels_default_to_bidirectional_with_no_filters():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    child_id = graph.nodes[anchor_id].children_ids[0]
    edge = graph.edges[(anchor_id, child_id)]

    assert edge.forward.style == "bidirectional"
    assert edge.reverse.style == "bidirectional"
    assert edge.forward.whitelist is None
    assert edge.forward.blacklist == set()


def test_graph_auditor_disabled_by_default_does_not_run_during_tick():
    graph = _build_graph(branch_factor=1, compute_budget=1)
    assert graph.config.graph_auditor_enabled is False
    graph.seed([0])
    graph.spawn_first_children()

    graph.tick()

    assert graph.traversals == {}


def test_graph_auditor_does_not_create_edges():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.5, 2)
    edge_count_before = len(graph.edges)

    graph._run_graph_auditor()

    assert len(graph.edges) == edge_count_before  # unchanged -- the auditor only ever touches traversals
    assert len(graph.traversals) > 0


def test_graph_auditor_enumerates_every_ancestor_descendant_pair():
    # Not just direct edges -- every (ancestor, descendant) pair anywhere
    # in the chain, since a traversal is any causal path, not just an
    # adjacent one.
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.5, 2)
    c_id = _add_hand_node(graph, b_id, 3, -0.5, 3)

    graph._run_graph_auditor()

    expected_keys = {
        (seed_id, a_id), (seed_id, b_id), (seed_id, c_id),
        (a_id, b_id), (a_id, c_id),
        (b_id, c_id),
    }
    assert set(graph.traversals.keys()) == expected_keys
    assert graph.traversals[(seed_id, c_id)].node_ids == [seed_id, a_id, b_id, c_id]
    assert graph.traversals[(a_id, c_id)].node_ids == [a_id, b_id, c_id]


def test_graph_auditor_never_reruns_an_already_recorded_traversal():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)

    graph._run_graph_auditor()
    existing = graph.traversals[(seed_id, a_id)]

    graph._run_graph_auditor()  # a second pass with nothing new

    assert graph.traversals[(seed_id, a_id)] is existing  # exact same object, never rebuilt


def test_traversal_mean_score_for_a_seed_rooted_path_matches_path_mean():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.3, 2)

    graph._run_graph_auditor()

    assert math.isclose(graph.traversals[(seed_id, b_id)].mean_score, graph.nodes[b_id].path_mean)


def test_traversal_mean_score_for_an_internal_path_is_the_sub_span_average():
    # (a, c)'s score must be exactly the mean evidence of JUST the a->c
    # sub-span, not the whole seed->c path -- genuinely derivable from
    # the two endpoints' own cumulative_evidence, not re-scored from
    # scratch.
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.5, 2)
    c_id = _add_hand_node(graph, b_id, 3, -0.5, 3)

    graph._run_graph_auditor()

    assert math.isclose(graph.traversals[(a_id, c_id)].mean_score, -0.5)


def test_burn_does_not_create_traversals():
    # Burn never *creates* traversals -- only the graph auditor populates
    # them, enumerating causal paths among whatever is currently live.
    # (Burn does prune already-recorded ones that reference the newly-dead
    # node -- see test_burn_prunes_traversals_touching_the_burned_node.)
    graph = _build_graph()
    seed_id = graph.seed([0])
    leaf_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)

    graph._burn(leaf_id)

    assert graph.traversals == {}


def test_burn_prunes_traversals_touching_the_burned_node():
    # self.traversals must not grow forever: a Traversal is recorded once
    # and never re-derived, so without pruning it would accumulate one
    # entry per (ancestor, descendant) pair ever seen across the whole
    # session, not just the ones still live -- a real unbounded-memory
    # risk given audit_edge_influence walks every recorded traversal on
    # every single tick.
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.5, 2)

    graph._run_graph_auditor()
    assert set(graph.traversals.keys()) == {(seed_id, a_id), (seed_id, b_id), (a_id, b_id)}

    graph._burn(a_id)  # cascades: a_id and its live child b_id both burn

    # Every traversal touching a_id or b_id is gone -- nothing references
    # a dead node anymore.
    assert graph.traversals == {}


def test_burn_leaves_traversals_among_still_live_nodes_alone():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, seed_id, 2, -0.5, 1)  # sibling of a_id, not a descendant

    graph._run_graph_auditor()
    assert (seed_id, b_id) in graph.traversals

    graph._burn(a_id)

    # a_id is gone, but the unrelated (seed_id, b_id) traversal survives.
    assert (seed_id, b_id) in graph.traversals
    assert (seed_id, a_id) not in graph.traversals


def test_burn_reaps_a_farther_backward_component_cut_off_from_the_seed():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    graph._attach_backward_parents(seed_id, [-0.1], [[1]])
    connector_id = graph.nodes[seed_id].parent_ids[-1]
    graph._attach_backward_parents(connector_id, [-0.2], [[2]])
    far_id = graph.nodes[connector_id].parent_ids[-1]
    graph.nodes[far_id].solubles["stored-ion"] = 2.0
    graph._run_graph_auditor()

    graph._burn(connector_id)

    assert graph.nodes[connector_id].burned is True
    assert graph.nodes[far_id].burned is True
    assert graph.bath["stored-ion"] == 2.0
    assert all(far_id not in edge_key for edge_key in graph.edges)
    assert all(
        far_id not in traversal.node_ids
        for traversal in graph.traversals.values()
    )


def test_connect_rejects_a_direct_backward_to_forward_level_jump():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    graph._attach_backward_parents(seed_id, [-0.1], [[1]])
    graph._attach_forward_children(seed_id, [-0.2], [[2]])
    backward_id = graph.nodes[seed_id].parent_ids[-1]
    forward_id = graph.nodes[seed_id].children_ids[-1]

    try:
        graph._connect(backward_id, forward_id)
        assert False, "expected a non-adjacent signed-level edge to be rejected"
    except ValueError as error:
        assert "must advance one signed level" in str(error)

def test_burn_preserves_a_backward_cousin_with_an_alternate_live_route():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    graph._attach_backward_parents(seed_id, [-0.1], [[1]])
    connector_id = graph.nodes[seed_id].parent_ids[-1]
    graph._attach_backward_parents(connector_id, [-0.2], [[2]])
    far_id = graph.nodes[connector_id].parent_ids[-1]
    graph._attach_backward_parents(seed_id, [-0.3], [[3]])
    alternate_id = graph.nodes[seed_id].parent_ids[-1]
    graph._connect(far_id, alternate_id)

    graph._burn(connector_id)

    assert graph.nodes[connector_id].burned is True
    assert graph.nodes[far_id].burned is False
    assert alternate_id in graph.nodes[far_id].children_ids


def test_survivor_of_a_burned_cousin_center_is_rehomed_to_a_live_center():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    graph._attach_forward_children(seed_id, [-0.1], [[1]])
    forward_id = graph.nodes[seed_id].children_ids[-1]
    graph._attach_backward_parents(forward_id, [-0.2], [[2]])
    cousin_id = [
        parent_id for parent_id in graph.nodes[forward_id].parent_ids
        if parent_id != seed_id
    ][0]
    graph._attach_forward_children(cousin_id, [-0.3], [[3]])
    survivor_id = graph.nodes[cousin_id].children_ids[-1]
    graph._connect(seed_id, survivor_id)
    assert graph.nodes[survivor_id].center_id == cousin_id

    graph._burn(cousin_id)

    assert graph.nodes[survivor_id].burned is False
    assert graph.nodes[survivor_id].center_id == seed_id
    assert cousin_id not in graph.orthogonal_network_roots().values()


def test_graph_auditor_keeps_causal_edges_unchanged_when_focus_moves():
    graph = _build_graph()
    seed_id = graph.seed([99])
    f1_id = _add_node(graph, seed_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2_id = _add_node(graph, f1_id, 2, -0.1, 2, direction=Direction.FORWARD, pressure=3.0)
    graph._reroot(f2_id)

    leaf_id = _add_hand_node(graph, seed_id, 3, -0.1, graph.nodes[seed_id].depth + 1)
    graph._run_graph_auditor()

    assert graph.traversals[(seed_id, f2_id)].node_ids == [seed_id, f1_id, f2_id]
    assert graph.traversals[(seed_id, leaf_id)].node_ids == [seed_id, leaf_id]
    assert (f2_id, leaf_id) not in graph.traversals


def test_evaluator_is_a_direct_view_over_traversals():
    graph = _build_graph()
    seed_id = graph.seed([0])
    fwd_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    graph.nodes[fwd_id].direction = Direction.FORWARD
    bwd_id = _add_hand_node(graph, seed_id, 2, -0.5, 1)
    graph.nodes[bwd_id].direction = Direction.BACKWARD

    graph._run_graph_auditor()

    assert graph.evaluator() is graph.traversals
    assert len(graph.evaluator()) == 2  # (seed,fwd) and (seed,bwd)


def test_audit_edge_influence_runs_the_auditor_itself():
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    assert graph.traversals == {}

    graph.audit_edge_influence()

    assert (seed_id, a_id) in graph.traversals  # no need to call _run_graph_auditor separately


def test_audit_edge_influence_sums_traversal_quality_per_real_edge():
    # seed->a->b: three traversals exist -- (seed,a), (a,b), and the
    # longer (seed,b) -- and (seed,b)'s path crosses both real edges, so
    # each real edge should see contributions from two traversals, not
    # just its own direct one.
    graph = _build_graph(branch_factor=1)
    seed_id = graph.seed([0])
    a_id = _add_hand_node(graph, seed_id, 1, -0.5, 1)
    b_id = _add_hand_node(graph, a_id, 2, -0.5, 2)

    influence = graph.audit_edge_influence()

    assert set(influence.keys()) == {(seed_id, a_id), (a_id, b_id)}
    quality = math.exp(-0.5)  # every traversal here has mean_score == -0.5
    for entry in influence.values():
        assert entry["count"] == 2
        assert math.isclose(entry["total"], 2 * quality, rel_tol=1e-6)


def test_meta_edges_bundles_edges_by_region():
    graph = _build_graph(branch_factor=1)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()

    metas = graph.meta_edges()

    assert ("main", "main") in metas
    meta = metas[("main", "main")]
    assert meta.from_region == "main"
    assert meta.to_region == "main"
    assert meta.member_edge_keys == set(graph.edges.keys())  # nothing orthogonal yet


def test_meta_edges_separates_an_orthogonal_network_into_its_own_region():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    cousin_id = _add_node(graph, f1_id, 10, -0.1, 2, direction=Direction.BACKWARD, pressure=1.0)
    assert cousin_id in graph.orthogonal_node_ids()
    graph.edges[(f1_id, cousin_id)] = Edge(
        from_id=f1_id, to_id=cousin_id,
        forward=Channel(conductance=0.5), reverse=Channel(conductance=0.5),
        formation="postfix_beam", seed_id_at_formation=anchor_id, created_tick=0,
    )

    metas = graph.meta_edges()

    net_region = f"net:{cousin_id}"
    assert ("main", net_region) in metas
    assert metas[("main", net_region)].member_edge_keys == {(f1_id, cousin_id)}


def test_tick_keeps_pressures_finite():
    graph = _build_graph()
    graph.seed([0])
    graph.spawn_first_children()
    for _ in range(5):
        graph.tick()
    for node in graph.nodes.values():
        assert math.isfinite(node.pressure)


def test_plan_expand_chunks_matches_flat_chunking_when_no_budget_set():
    graph = _build_graph()
    graph.config.expand_batch_chunk_size = 5
    row_lens = [3] * 13
    chunks = graph._plan_expand_chunks(row_lens, vocab_size=100)
    assert chunks == [(0, 5), (5, 10), (10, 13)]


def test_plan_expand_chunks_shrinks_rows_as_row_len_grows_under_a_budget():
    graph = _build_graph()
    graph.config.expand_batch_chunk_size = 100
    graph.config.max_expand_elements = 1000
    row_lens = [10] * 5 + [50] * 5  # vocab=10: budget/row_len/vocab = 10 rows, then 2 rows
    chunks = graph._plan_expand_chunks(row_lens, vocab_size=10)
    for start, end in chunks:
        segment = row_lens[start:end]
        count = end - start
        assert count == 1 or count * max(segment) * 10 <= 1000


def test_plan_expand_chunks_still_serves_a_single_row_that_alone_exceeds_budget():
    graph = _build_graph()
    graph.config.expand_batch_chunk_size = 100
    graph.config.max_expand_elements = 10
    chunks = graph._plan_expand_chunks([1000], vocab_size=10)
    assert chunks == [(0, 1)]  # served, not dropped or raising


def test_best_path_includes_anchor_and_grows_with_ticks():
    graph = _build_graph()
    graph.seed([2])
    graph.spawn_first_children()
    tokens_before, _ = graph.best_path()
    assert 2 in tokens_before

    for _ in range(4):
        graph.tick()
    tokens_after, score_after = graph.best_path()
    assert len(tokens_after) >= len(tokens_before)
    assert math.isfinite(score_after)


def test_starvation_burns_a_low_evidence_leaf_with_no_support():
    graph = _build_graph()
    anchor_id = graph.seed([0])

    # Hand-build one healthy leaf (high local_evidence, so it keeps getting
    # support) and one starved leaf (near-zero local_evidence, isolated) --
    # bypass real expansion so the starvation mechanism is tested directly.
    healthy_id = graph._alloc_id()
    graph.nodes[healthy_id] = FluxNode(
        id=healthy_id, tokens=[1], direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=0.0, pressure=1.0, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(healthy_id)

    starved_id = graph._alloc_id()
    graph.nodes[starved_id] = FluxNode(
        id=starved_id, tokens=[2], direction=Direction.BACKWARD, parent_id=anchor_id,
        depth=1, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(starved_id)

    for _ in range(graph.config.burn_after_ticks + 2):
        graph._update_pressures()
        graph._starve_and_burn()

    assert graph.nodes[starved_id].burned is True
    assert graph.nodes[healthy_id].burned is False
    assert starved_id not in graph.nodes[anchor_id].children_ids
    assert healthy_id in graph.nodes[anchor_id].children_ids


def test_starved_internal_branch_burns_without_leaving_orphans():
    # Starvation applies to every non-anchor node. When an internal node
    # burns, descendants that have no other causal parent burn with it.
    graph = _build_graph()
    anchor_id = graph.seed([0])

    weak_parent_id = graph._alloc_id()
    graph.nodes[weak_parent_id] = FluxNode(
        id=weak_parent_id, tokens=[1], direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(weak_parent_id)

    child_id = graph._alloc_id()
    graph.nodes[child_id] = FluxNode(
        id=child_id, tokens=[2], direction=Direction.FORWARD, parent_id=weak_parent_id,
        depth=2, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[weak_parent_id].children_ids.append(child_id)

    for _ in range(graph.config.burn_after_ticks + 2):
        graph._update_pressures()
        graph._starve_and_burn()

    assert graph.nodes[weak_parent_id].burned is True
    assert graph.nodes[child_id].burned is True


def test_a_node_whose_children_all_burn_becomes_expandable_again():
    # Eligibility must be based on current leaf status, not "has this node
    # ever been expanded" -- otherwise a node whose children all burn off
    # becomes a permanent dead end that can win best_path() forever
    # without ever being able to grow past it.
    graph = _build_graph()
    anchor_id = graph.seed([0])

    parent_id = graph._alloc_id()
    graph.nodes[parent_id] = FluxNode(
        id=parent_id, tokens=[1], direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=0.0, pressure=1.0, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(parent_id)

    child_id = graph._alloc_id()
    graph.nodes[child_id] = FluxNode(
        id=child_id, tokens=[2], direction=Direction.FORWARD, parent_id=parent_id,
        depth=2, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[parent_id].children_ids.append(child_id)

    assert parent_id not in [n.id for n in graph._expandable_nodes()]

    for _ in range(graph.config.burn_after_ticks + 2):
        graph._update_pressures()
        graph._starve_and_burn()

    assert graph.nodes[child_id].burned is True
    assert parent_id in [n.id for n in graph._expandable_nodes()]


def _build_chain(graph, length, local_evidence, direction=Direction.FORWARD):
    """Hand-build a straight chain of ``length`` nodes off the anchor."""
    prev = graph.anchor_id
    chain = []
    for depth in range(1, length + 1):
        nid = graph._alloc_id()
        cumulative = graph.nodes[prev].cumulative_evidence + local_evidence
        graph.nodes[nid] = FluxNode(
            id=nid, tokens=[depth], direction=direction, parent_id=prev,
            depth=depth, local_evidence=local_evidence, pressure=1.0, expanded=True,
            cumulative_evidence=cumulative, rollup_mean=cumulative / depth,
        )
        graph.nodes[prev].children_ids.append(nid)
        chain.append(nid)
        prev = nid
    return chain


def test_settle_circuit_does_more_than_one_relaxation_step():
    graph = _build_graph()
    graph.seed([0])
    _build_chain(graph, length=10, local_evidence=-1.0)

    graph._update_pressures()
    single_step = {nid: n.pressure for nid, n in graph.nodes.items()}

    graph2 = _build_graph()
    graph2.seed([0])
    _build_chain(graph2, length=10, local_evidence=-1.0)
    iterations = graph2._settle_circuit()

    assert iterations > 1
    settled = {nid: n.pressure for nid, n in graph2.nodes.items()}
    # Settling changes the far end of the chain relative to a single sweep --
    # multi-hop support hasn't had time to arrive after just one update.
    assert any(
        abs(settled[nid] - single_step[nid]) > 1e-6
        for nid in settled if nid != graph.anchor_id
    )


def test_settle_circuit_is_a_no_op_once_already_converged():
    graph = _build_graph()
    graph.seed([0])
    _build_chain(graph, length=6, local_evidence=-1.0)
    graph._settle_circuit()
    # Already settled -- a second call should converge almost immediately.
    iterations = graph._settle_circuit()
    assert iterations <= 2


def test_digest_propagates_a_deep_strong_discovery_to_shallow_ancestors():
    graph = _build_graph()
    graph.seed([0])

    # A weak first hop, then a much stronger run deeper in -- an ancestor's
    # own path_mean alone looks worse than what's reachable through it.
    weak_id = _build_chain(graph, length=1, local_evidence=-4.0)[0]
    prev = weak_id
    strong_chain = []
    for depth in range(2, 6):
        nid = graph._alloc_id()
        cumulative = graph.nodes[prev].cumulative_evidence + 0.0
        graph.nodes[nid] = FluxNode(
            id=nid, tokens=[depth], direction=Direction.FORWARD, parent_id=prev,
            depth=depth, local_evidence=0.0, pressure=1.0, expanded=True,
            cumulative_evidence=cumulative, rollup_mean=cumulative / depth,
        )
        graph.nodes[prev].children_ids.append(nid)
        strong_chain.append(nid)
        prev = nid

    graph._digest()

    deepest = graph.nodes[strong_chain[-1]]
    ancestor = graph.nodes[weak_id]
    # The deepest node's own path_mean is much better than the weak
    # ancestor's own path_mean...
    assert deepest.path_mean > ancestor.path_mean
    # ...and digestion should have carried that back: the ancestor's
    # rollup_mean reflects the best thing reachable through it, not just
    # its own (worse) local path_mean.
    assert ancestor.rollup_mean > ancestor.path_mean
    assert math.isclose(ancestor.rollup_mean, deepest.rollup_mean, rel_tol=1e-6)


def test_expansion_priority_gives_a_long_waiting_node_a_boost():
    graph = _build_graph()
    graph.seed([0])

    # Two candidate leaves with nearly the same pressure, but one has been
    # sitting eligible for many ticks while the other just appeared.
    old_id = graph._alloc_id()
    graph.nodes[old_id] = FluxNode(
        id=old_id, tokens=[1], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=-1.0, pressure=0.5, created_tick=0,
    )
    graph.nodes[graph.anchor_id].children_ids.append(old_id)

    new_id = graph._alloc_id()
    graph.nodes[new_id] = FluxNode(
        id=new_id, tokens=[2], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=-1.0, pressure=0.51, created_tick=0,
    )
    graph.nodes[graph.anchor_id].children_ids.append(new_id)

    graph.tick_count = 0
    # At tick 0, both have waited 0 ticks -- the tiny pressure edge wins.
    assert graph._expansion_priority(graph.nodes[new_id]) > graph._expansion_priority(graph.nodes[old_id])

    # Advance time without ever expanding old_id -- its wait-time bonus
    # should eventually overcome new_id's tiny pressure edge.
    graph.nodes[new_id].created_tick = 20  # new_id "arrives" much later
    graph.tick_count = 20
    assert graph._expansion_priority(graph.nodes[old_id]) > graph._expansion_priority(graph.nodes[new_id])


def test_hot_loop_depth_one_matches_current_single_level_behavior():
    graph = _build_graph(branch_factor=2, compute_budget=2)
    graph.config.hot_loop_depth = 1
    graph.seed([0])
    graph.spawn_first_children()

    graph._expand_top_pressure_nodes()

    live_depths = [n.depth for n in graph.nodes.values() if not n.burned]
    assert max(live_depths) == 2  # anchor(0) -> spawn_first_children(1) -> one round(2)


def test_hot_loop_depth_grows_several_levels_in_one_call():
    graph = _build_graph(branch_factor=2, compute_budget=2)
    graph.config.hot_loop_depth = 4
    graph.seed([0])
    graph.spawn_first_children()

    graph._expand_top_pressure_nodes()

    live_depths = [n.depth for n in graph.nodes.values() if not n.burned]
    assert max(live_depths) == 5  # depth-1 leaves selected, then 4 hot-loop rounds on top


def test_hot_loop_depth_every_round_uses_the_same_branch_factor():
    graph = _build_graph(branch_factor=2, compute_budget=2)
    graph.config.hot_loop_depth = 3
    graph.seed([0])
    graph.spawn_first_children()

    graph._expand_top_pressure_nodes()

    by_depth = {}
    for n in graph.nodes.values():
        if n.burned:
            continue
        by_depth[n.depth] = by_depth.get(n.depth, 0) + 1
    # 2 selected depth-1 leaves (compute_budget=2), branch_factor=2 at every
    # round: depth 2 has 2*2=4, depth 3 has 4*2=8, depth 4 has 8*2=16.
    assert by_depth[2] == 4
    assert by_depth[3] == 8
    assert by_depth[4] == 16


def test_hot_loop_depth_handles_an_empty_frontier_without_erroring():
    # If some round ever produces zero children (e.g. no_repeat_ngram_size
    # blocks every candidate), the remaining rounds must no-op cleanly
    # instead of erroring on an empty frontier.
    graph = _build_graph(branch_factor=1, compute_budget=1)
    graph._expand_batch_hot_loop([], depth=5)  # should not raise on an empty starting frontier either

    graph.config.hot_loop_depth = 10
    graph.seed([0])
    graph.spawn_first_children()
    graph._expand_top_pressure_nodes()  # should not raise despite depth=10


def test_direction_reach_is_zero_when_a_side_has_no_live_nodes():
    graph = _build_graph()
    graph.seed([0])
    assert graph._direction_reach(Direction.FORWARD) == 0.0
    assert graph._direction_reach(Direction.BACKWARD) == 0.0


def test_direction_reach_tracks_the_furthest_live_node_on_that_side():
    graph = _build_graph()
    graph.seed([0])
    for depth in (1, 2, 5):
        nid = graph._alloc_id()
        graph.nodes[nid] = FluxNode(
            id=nid, tokens=[depth], direction=Direction.FORWARD, parent_id=graph.anchor_id,
            depth=depth, local_evidence=0.0, pressure=1.0,
        )
    assert graph._direction_reach(Direction.FORWARD) == 5.0
    assert graph._direction_reach(Direction.BACKWARD) == 0.0


def test_direction_reach_ignores_burned_nodes():
    graph = _build_graph()
    graph.seed([0])
    nid = graph._alloc_id()
    graph.nodes[nid] = FluxNode(
        id=nid, tokens=[1], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=7, local_evidence=0.0, pressure=1.0, burned=True,
    )
    assert graph._direction_reach(Direction.FORWARD) == 0.0


def test_balance_weight_zero_is_a_true_noop():
    graph = _build_graph()
    graph.config.balance_weight = 0.0
    graph.seed([0])
    fwd_id = graph._alloc_id()
    graph.nodes[fwd_id] = FluxNode(
        id=fwd_id, tokens=[1], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=0.0, pressure=0.5,
    )
    bwd_id = graph._alloc_id()
    graph.nodes[bwd_id] = FluxNode(
        id=bwd_id, tokens=[2], direction=Direction.BACKWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=0.0, pressure=0.5,
    )
    # Forward is way out ahead, backward barely started -- with the knob
    # off, that imbalance must not change either node's priority at all.
    plain = graph._expansion_priority(graph.nodes[bwd_id])
    boosted = graph._expansion_priority(graph.nodes[bwd_id], forward_reach=50.0, backward_reach=1.0)
    assert math.isclose(plain, boosted)


def test_balance_weight_boosts_only_the_trailing_sides_candidates():
    graph = _build_graph()
    graph.config.balance_weight = 1.0
    graph.seed([0])
    fwd_id = graph._alloc_id()
    graph.nodes[fwd_id] = FluxNode(
        id=fwd_id, tokens=[1], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=0.0, pressure=0.5,
    )
    bwd_id = graph._alloc_id()
    graph.nodes[bwd_id] = FluxNode(
        id=bwd_id, tokens=[2], direction=Direction.BACKWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=0.0, pressure=0.5,
    )
    # Forward has pulled ahead (reach 10 vs backward's 1): the lagging
    # backward candidate should get boosted...
    bwd_priority = graph._expansion_priority(graph.nodes[bwd_id], forward_reach=10.0, backward_reach=1.0)
    bwd_baseline = graph._expansion_priority(graph.nodes[bwd_id], forward_reach=1.0, backward_reach=1.0)
    assert bwd_priority > bwd_baseline
    assert math.isclose(bwd_priority, bwd_baseline + 1.0 * (10.0 - 1.0))
    # ...while the already-leading forward candidate gets nothing extra.
    fwd_priority = graph._expansion_priority(graph.nodes[fwd_id], forward_reach=10.0, backward_reach=1.0)
    fwd_baseline = graph._expansion_priority(graph.nodes[fwd_id], forward_reach=1.0, backward_reach=1.0)
    assert math.isclose(fwd_priority, fwd_baseline)


def test_balance_weight_shifts_which_node_expand_top_pressure_picks():
    # Integration-level: a lower-pressure backward leaf, once boosted for
    # trailing far behind an already-deep forward side, can outrank a
    # slightly-stronger forward leaf for the tick's single expansion slot.
    graph = _build_graph(branch_factor=1, compute_budget=1)
    graph.config.balance_weight = 5.0
    graph.seed([0])

    prev = graph.anchor_id
    for depth in range(1, 6):
        nid = graph._alloc_id()
        graph.nodes[nid] = FluxNode(
            id=nid, tokens=[depth], direction=Direction.FORWARD, parent_id=prev,
            depth=depth, local_evidence=0.0, pressure=1.0, expanded=(depth < 5),
        )
        graph.nodes[prev].children_ids.append(nid)
        prev = nid
    forward_leaf = prev

    backward_leaf = graph._alloc_id()
    graph.nodes[backward_leaf] = FluxNode(
        id=backward_leaf, tokens=[99], direction=Direction.BACKWARD, parent_id=None,
        children_ids=[graph.anchor_id], depth=1, level=-1, center_id=graph.anchor_id,
        local_evidence=0.0, pressure=0.9,
    )
    graph.nodes[graph.anchor_id].parent_ids.append(backward_leaf)
    graph.nodes[graph.anchor_id].parent_id = backward_leaf

    candidates = graph._expandable_nodes()
    assert {forward_leaf, backward_leaf} <= {n.id for n in candidates}
    # Without balance, the higher-pressure forward leaf wins outright.
    assert (
        graph._expansion_priority(graph.nodes[forward_leaf])
        > graph._expansion_priority(graph.nodes[backward_leaf])
    )

    forward_reach = graph._direction_reach(Direction.FORWARD)
    backward_reach = graph._direction_reach(Direction.BACKWARD)
    assert forward_reach > backward_reach
    # With balance active, the trailing backward leaf's boost (proportional
    # to the reach gap) overtakes the forward leaf's pressure edge.
    assert (
        graph._expansion_priority(graph.nodes[backward_leaf], forward_reach, backward_reach)
        > graph._expansion_priority(graph.nodes[forward_leaf], forward_reach, backward_reach)
    )


def test_repeated_ngram_tokens_blocks_exact_trigram_repeat_on_append():
    graph = _build_graph()
    graph.config.no_repeat_ngram_size = 3
    # "1 2" was already followed by 3 once; about to follow "1 2" again.
    context = [5, 1, 2, 3, 1, 2]
    assert graph._repeated_ngram_tokens(context, "append") == {3}


def test_repeated_ngram_tokens_blocks_exact_trigram_repeat_on_prepend():
    graph = _build_graph()
    graph.config.no_repeat_ngram_size = 3
    # Prepending goes at the very front, so the new n-gram is
    # [candidate] + context[:2]. "1 2" already exists preceded by "3"
    # later in the context ([..., 3, 1, 2, 5]) -- prepending "3" onto the
    # very front (which itself starts with "1 2") would recreate that
    # exact trigram.
    context = [1, 2, 3, 1, 2, 5]
    assert graph._repeated_ngram_tokens(context, "prepend") == {3}


def test_repeated_ngram_tokens_disabled_when_size_is_none():
    graph = _build_graph()
    graph.config.no_repeat_ngram_size = None
    assert graph._repeated_ngram_tokens([1, 2, 1, 2], "append") == set()


def test_repeated_ngram_tokens_ignores_non_matching_context():
    graph = _build_graph()
    graph.config.no_repeat_ngram_size = 3
    assert graph._repeated_ngram_tokens([1, 2, 3, 4, 5], "append") == set()


def test_expand_forward_blocks_a_token_that_would_recreate_a_seen_bigram():
    # An oscillating trap: token 0's only strong continuation is token 1,
    # and token 1's only strong continuation is token 0 -- exactly the
    # "I'm sorry, I'm sorry, ..." shape an induction head falls into.
    # Length-normalizing the score doesn't stop this (each step is
    # genuinely high-probability); no-repeat-ngram blocking should.
    class OscillatingDummyModel(AbstractModelWrapper):
        def forward(self, input_ids, attention_mask, **kwargs):
            table = torch.tensor(
                [
                    [0.0, 10.0, 0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            return {"logits": table[input_ids]}

        def get_device(self):
            return "cpu"

    model = OscillatingDummyModel()
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=1, compute_budget_per_tick=1, no_repeat_ngram_size=2)
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([0])

    graph._expand_forward(graph.anchor_id)
    first_child = graph.nodes[graph.anchor_id].children_ids[0]
    assert graph.nodes[first_child].tokens[0] == 1  # 0 -> 1 is the model's only strong pick

    graph._expand_forward(first_child)  # context [0, 1]: no repeat yet, greedy pick is 0
    second_child = graph.nodes[first_child].children_ids[0]
    assert graph.nodes[second_child].tokens[0] == 0

    graph._expand_forward(second_child)  # context [0, 1, 0]: greedy pick 1 would recreate bigram [0, 1]
    third_child = graph.nodes[second_child].children_ids[0]
    assert graph.nodes[third_child].tokens[0] != 1


def test_expand_backward_normalizes_by_suffix_length_not_raw_sum():
    # A uniform model: the log-prob of any target at any position is always
    # log(1/VOCAB), so the *total* suffix log-likelihood scales with suffix
    # length even though every token is equally "good". local_evidence must
    # be the per-token mean, not that raw total, or a backward candidate
    # scored against a longer suffix looks mechanically worse than the same
    # quality candidate scored against a short one -- exactly the bug that
    # was silently starving backward nodes as the graph grew.
    class UniformDummyModel(AbstractModelWrapper):
        def forward(self, input_ids, attention_mask, **kwargs):
            batch, row_len = input_ids.shape
            logits = torch.zeros(batch, row_len, VOCAB, dtype=torch.float32)
            return {"logits": logits}

        def get_device(self):
            return "cpu"

    model = UniformDummyModel()
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=1, compute_budget_per_tick=1, no_repeat_ngram_size=None)
    expected_per_token = -math.log(VOCAB)

    short_graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    short_graph.seed([0])
    short_graph._expand_backward(short_graph.anchor_id)
    short_parent = short_graph.nodes[short_graph.anchor_id].parent_ids[0]
    assert math.isclose(short_graph.nodes[short_parent].local_evidence, expected_per_token, rel_tol=1e-4)

    long_graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    long_graph.seed([0, 1, 2, 3, 4])
    long_graph._expand_backward(long_graph.anchor_id)
    long_parent = long_graph.nodes[long_graph.anchor_id].parent_ids[0]
    assert math.isclose(long_graph.nodes[long_parent].local_evidence, expected_per_token, rel_tol=1e-4)


def _cycle_table_textured(vocab=VOCAB, peak=10.0):
    table = []
    for i in range(vocab):
        row = [0.0] * vocab
        row[(i + 1) % vocab] = peak
        row[(i + 2) % vocab] = peak / 2
        table.append(row)
    return table


def test_expand_batch_matches_expand_backward_for_a_single_backward_node():
    # _expand_batch scores rows across every selected node in one shared,
    # padded, chunked pass instead of _expand_backward's own per-node call.
    # Scored alone (no forward node sharing the call), it must produce
    # bit-for-bit the same result as the old per-node path -- the batching
    # is purely about how many model calls happen, not what gets computed.
    table = _cycle_table_textured()
    model = BigramDummyModel(table)
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=2, compute_budget_per_tick=2, no_repeat_ngram_size=None)

    g_old = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    g_old.seed([0, 1])
    g_old.spawn_first_children()
    bwd_old = next(c for c in g_old.nodes[g_old.anchor_id].parent_ids
                   if g_old.nodes[c].direction is Direction.BACKWARD)
    g_old._expand_backward(bwd_old)
    old_result = sorted(
        (g_old.nodes[c].tokens[0], round(g_old.nodes[c].local_evidence, 6))
        for c in g_old.nodes[bwd_old].parent_ids
    )

    g_new = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    g_new.seed([0, 1])
    g_new.spawn_first_children()
    bwd_new = next(c for c in g_new.nodes[g_new.anchor_id].parent_ids
                   if g_new.nodes[c].direction is Direction.BACKWARD)
    g_new._expand_batch([g_new.nodes[bwd_new]])
    new_result = sorted(
        (g_new.nodes[c].tokens[0], round(g_new.nodes[c].local_evidence, 6))
        for c in g_new.nodes[bwd_new].parent_ids
    )

    assert old_result == new_result


def test_expand_batch_matches_expand_forward_for_a_single_forward_node():
    table = _cycle_table_textured()
    model = BigramDummyModel(table)
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=2, compute_budget_per_tick=2, no_repeat_ngram_size=None)

    g_old = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    g_old.seed([0, 1])
    g_old.spawn_first_children()
    fwd_old = next(c for c in g_old.nodes[g_old.anchor_id].children_ids
                   if g_old.nodes[c].direction is Direction.FORWARD)
    g_old._expand_forward(fwd_old)
    old_result = sorted(
        (g_old.nodes[c].tokens[0], round(g_old.nodes[c].local_evidence, 6))
        for c in g_old.nodes[fwd_old].children_ids
    )

    g_new = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    g_new.seed([0, 1])
    g_new.spawn_first_children()
    fwd_new = next(c for c in g_new.nodes[g_new.anchor_id].children_ids
                   if g_new.nodes[c].direction is Direction.FORWARD)
    g_new._expand_batch([g_new.nodes[fwd_new]])
    new_result = sorted(
        (g_new.nodes[c].tokens[0], round(g_new.nodes[c].local_evidence, 6))
        for c in g_new.nodes[fwd_new].children_ids
    )

    assert old_result == new_result


def test_expand_batch_handles_forward_and_backward_together_in_one_call():
    # The actual point of _expand_batch: one call, one shared padded/chunked
    # pipeline, both directions represented among the rows at once.
    table = _cycle_table_textured()
    model = BigramDummyModel(table)
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(branch_factor=2, compute_budget_per_tick=2, no_repeat_ngram_size=None)

    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([0, 1])
    graph.spawn_first_children()
    fwd = next(c for c in graph.nodes[graph.anchor_id].children_ids
               if graph.nodes[c].direction is Direction.FORWARD)
    bwd = next(c for c in graph.nodes[graph.anchor_id].parent_ids
               if graph.nodes[c].direction is Direction.BACKWARD)

    graph._expand_batch([graph.nodes[fwd], graph.nodes[bwd]])

    assert len(graph.nodes[fwd].children_ids) == config.branch_factor
    assert len(graph.nodes[bwd].parent_ids) == config.branch_factor
    for c in graph.nodes[fwd].children_ids:
        assert graph.nodes[c].direction is Direction.FORWARD
    for c in graph.nodes[bwd].parent_ids:
        assert graph.nodes[c].direction is Direction.BACKWARD


def test_expand_batch_chunking_matches_unchunked_for_backward_candidates():
    # A tiny chunk size forces multiple chunked model calls within one
    # _expand_batch call -- results must be identical to one big chunk.
    table = _cycle_table_textured()
    model = BigramDummyModel(table)
    tok = FakeTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)

    def run(chunk_size):
        config = FluxGraphConfig(
            branch_factor=2, compute_budget_per_tick=2,
            no_repeat_ngram_size=None, expand_batch_chunk_size=chunk_size,
        )
        graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
        graph.seed([0, 1])
        graph.spawn_first_children()
        bwd = next(c for c in graph.nodes[graph.anchor_id].parent_ids
                   if graph.nodes[c].direction is Direction.BACKWARD)
        graph._expand_batch([graph.nodes[bwd]])
        return sorted(
            (graph.nodes[c].tokens[0], round(graph.nodes[c].local_evidence, 6))
            for c in graph.nodes[bwd].parent_ids
        )

    assert run(chunk_size=2048) == run(chunk_size=1)


_POETIC_WORDS = ["cat", "hat", "dog", "run", "sun", "log"]


class FakeWordTokenizer:
    vocab_size = len(_POETIC_WORDS)

    def decode(self, ids):
        return " ".join(_POETIC_WORDS[i] for i in ids)


class FixedLogitModel(AbstractModelWrapper):
    """Same logits at every position -- isolates poetic re-ranking from context effects."""

    def __init__(self, logits):
        self.logits = logits

    def forward(self, input_ids, attention_mask, **kwargs):
        batch, row_len = input_ids.shape
        row = torch.tensor(self.logits, dtype=torch.float32)
        return {"logits": row.unsqueeze(0).unsqueeze(0).expand(batch, row_len, len(self.logits)).clone()}

    def get_device(self):
        return "cpu"


def _hand_log_softmax_at(row, idx):
    m = max(row)
    denom = math.log(sum(math.exp(v - m) for v in row))
    return (row[idx] - m) - denom


def _make_hat_leaf(graph):
    hat_id = graph._alloc_id()
    graph.nodes[hat_id] = FluxNode(
        id=hat_id, tokens=[1], direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=0.0, pressure=1.0, cumulative_evidence=0.0, rollup_mean=0.0,
    )
    graph.nodes[graph.anchor_id].children_ids.append(hat_id)
    return hat_id


def test_poetic_attractor_disabled_matches_plain_expand_forward_exactly():
    # cat, hat, dog, run, sun, log -- the model strongly prefers "dog".
    logits = [1.0, -50.0, 5.0, 0.0, 0.0, 0.0]
    model = FixedLogitModel(logits)
    tok = FakeWordTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    expected_dog = _hand_log_softmax_at(logits, 2)

    config = FluxGraphConfig(branch_factor=1, compute_budget_per_tick=1, no_repeat_ngram_size=None, poetic_attractor=None)
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([])
    hat = _make_hat_leaf(graph)
    graph._expand_batch([graph.nodes[hat]])

    picked = graph.nodes[hat].children_ids[0]
    assert graph.nodes[picked].tokens[0] == 2  # dog, the model's real favorite
    assert math.isclose(graph.nodes[picked].local_evidence, expected_dog, rel_tol=1e-4)


def test_poetic_attractor_can_override_the_models_top_pick_without_corrupting_the_score():
    # The model still strongly prefers "dog" over "cat" -- but "cat" rhymes
    # with the context's last word "hat" and "dog" doesn't. A strong enough
    # poetic attractor should flip the pick to "cat", while local_evidence
    # on the resulting node must stay "cat"'s own true (mediocre) score,
    # never the poetically-boosted ranking value -- the same honesty
    # guarantee local_evidence was just fixed to have for forward/backward.
    logits = [1.0, -50.0, 5.0, 0.0, 0.0, 0.0]
    model = FixedLogitModel(logits)
    tok = FakeWordTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    expected_cat = _hand_log_softmax_at(logits, 0)

    attractor = PoeticAttractor(rhyme_weight=5.0, slant_rhyme_weight=2.0, alliteration_weight=1.0, internal_rhyme_weight=1.0)
    config = FluxGraphConfig(
        branch_factor=1, compute_budget_per_tick=1, no_repeat_ngram_size=None,
        poetic_attractor=attractor, poetic_scale=3.0, poetic_shortlist_k=20,
    )
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([])
    hat = _make_hat_leaf(graph)
    graph._expand_batch([graph.nodes[hat]])

    picked = graph.nodes[hat].children_ids[0]
    assert graph.nodes[picked].tokens[0] == 0  # cat, not the model's raw favorite (dog)
    assert math.isclose(graph.nodes[picked].local_evidence, expected_cat, rel_tol=1e-4)


from speaktome.core.word_trie import WordTrie

_WG_WORDS = [" run", "ning", " cat", " dog"]


class WordGrowthTokenizer:
    vocab_size = len(_WG_WORDS)

    def decode(self, ids):
        return "".join(_WG_WORDS[i] for i in ids)


_WG_TABLE = [
    [0.0, 10.0, 0.0, 0.0],   # after " run" -> strongly prefer "ning"
    [0.0, 0.0, 10.0, 0.0],   # after "ning" -> strongly prefer " cat"
    [0.0, 0.0, 0.0, 10.0],   # after " cat" -> strongly prefer " dog"
    [0.0, 10.0, 0.0, 0.0],
]


class WordGrowthBigramModel(AbstractModelWrapper):
    def forward(self, input_ids, attention_mask, **kwargs):
        t = torch.tensor(_WG_TABLE, dtype=torch.float32)
        return {"logits": t[input_ids]}

    def get_device(self):
        return "cpu"


def _word_growth_graph(
    word_trie, branch_factor=2, max_subword_steps=6, anchor_tokens=None,
    backward_word_trie=None,
):
    model = WordGrowthBigramModel()
    tok = WordGrowthTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(
        branch_factor=branch_factor, compute_budget_per_tick=1, no_repeat_ngram_size=None,
        word_trie=word_trie, backward_word_trie=backward_word_trie,
        max_subword_steps=max_subword_steps,
    )
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed(anchor_tokens if anchor_tokens is not None else [])
    return graph


def test_grow_forward_word_joins_continuation_pieces_into_one_word():
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie)
    result = graph._grow_forward_word(graph.nodes[graph.anchor_id], [(0, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert (0, 1) in spans  # "run" + "ning" -> "running", joined into one span


def test_grow_forward_word_stops_before_a_fresh_word_candidate():
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie)
    result = graph._grow_forward_word(graph.nodes[graph.anchor_id], [(0, -0.1)])
    spans = [tokens for tokens, _ in result]
    # "cat" (token 2) must never be folded into the "running" span -- it's
    # the *next* word's first token, not this one's third.
    for span in spans:
        assert 2 not in span


def test_grow_forward_word_prunes_a_continuation_the_trie_rejects():
    # No "running" in this dictionary -- "run" alone is valid, but growing
    # it into "run"+"ning" is not, so that continuation must be dropped.
    trie = WordTrie(["run", "cat", "dog"])
    graph = _word_growth_graph(trie)
    result = graph._grow_forward_word(graph.nodes[graph.anchor_id], [(0, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert (0, 1) not in spans
    assert (0,) in spans


def test_grow_forward_word_has_no_duplicate_spans():
    # Regression: multiple tied fresh-word candidates in the same round
    # must finalize a beam once, not once per triggering candidate.
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie, branch_factor=2)
    result = graph._grow_forward_word(graph.nodes[graph.anchor_id], [(0, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert len(spans) == len(set(spans))


def test_grow_forward_word_respects_max_subword_steps():
    words = [" a", "b", "c", "d", "e"]

    class NeverStopsTokenizer:
        vocab_size = len(words)

        def decode(self, ids):
            return "".join(words[i] for i in ids)

    table = [
        [0, 10, 0, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 0, 10, 0],
        [0, 0, 0, 0, 10],
        [0, 0, 0, 0, 10],
    ]

    class NeverStopsModel(AbstractModelWrapper):
        def forward(self, input_ids, attention_mask, **kwargs):
            t = torch.tensor(table, dtype=torch.float32)
            return {"logits": t[input_ids]}

        def get_device(self):
            return "cpu"

    model = NeverStopsModel()
    tok = NeverStopsTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(
        branch_factor=1, compute_budget_per_tick=1, no_repeat_ngram_size=None,
        word_trie=None, max_subword_steps=4,
    )
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([])
    result = graph._grow_forward_word(graph.nodes[graph.anchor_id], [(0, -0.1)])
    assert len(result[0][0]) <= 4


def test_grow_backward_word_prepends_until_a_fresh_word_token():
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie)
    # Seed with the continuation piece "ning" alone -- growth must prepend
    # " run" before it to complete "running".
    result = graph._grow_backward_word(graph.nodes[graph.anchor_id], [(1, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert (0, 1) in spans


def test_grow_backward_word_finalizes_immediately_on_a_fresh_word_round0_candidate():
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie)
    # Seed with " cat" (already starts fresh) -- must finalize as a single-token word.
    result = graph._grow_backward_word(graph.nodes[graph.anchor_id], [(2, -0.05)])
    assert result == [([2], -0.05)]


def test_grow_backward_word_with_reversed_trie_produces_running():
    fwd = WordTrie(["running", "cat", "dog"])
    bwd = WordTrie(["running", "cat", "dog"], reverse=True)
    graph = _word_growth_graph(fwd, backward_word_trie=bwd)
    result = graph._grow_backward_word(graph.nodes[graph.anchor_id], [(1, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert (0, 1) in spans


def test_grow_backward_word_never_loops_on_a_repeated_dictionary_fragment():
    """Regression: a real GPT-2 run produced garbage like "AbAbAballah"
    because each individually-dictionary-valid token ("Ab" alone passes a
    flat per-token filter) got prepended repeatedly with no check that the
    *accumulating* span still headed toward a real word. The reversed
    trie's stateful walk must reject this even when the model itself
    strongly prefers repeating the fragment.
    """
    words = [" run", "ning", "ab"]  # "ab" is a short, tempting, real-looking fragment

    class RepeatBaitTokenizer:
        vocab_size = len(words)

        def decode(self, ids):
            return "".join(words[i] for i in ids)

    # After "ning", the model strongly prefers repeating "ab" forever
    # rather than ever producing " run".
    table = [
        [0.0, 10.0, 0.0],   # after " run" -> "ning"
        [0.0, 0.0, 10.0],   # after "ning" -> "ab" (repeat bait, model's real favorite)
        [0.0, 0.0, 10.0],   # after "ab" -> "ab" again
    ]

    class RepeatBaitModel(AbstractModelWrapper):
        def forward(self, input_ids, attention_mask, **kwargs):
            t = torch.tensor(table, dtype=torch.float32)
            return {"logits": t[input_ids]}

        def get_device(self):
            return "cpu"

    model = RepeatBaitModel()
    tok = RepeatBaitTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    fwd = WordTrie(["running"])
    bwd = WordTrie(["running"], reverse=True)
    config = FluxGraphConfig(
        branch_factor=2, compute_budget_per_tick=1, no_repeat_ngram_size=None,
        word_trie=fwd, backward_word_trie=bwd, max_subword_steps=6,
    )
    graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    graph.seed([])

    result = graph._grow_backward_word(graph.nodes[graph.anchor_id], [(1, -0.1)])
    spans = [tuple(tokens) for tokens, _ in result]
    assert (0, 1) in spans, f"expected running (0,1) among {spans}"
    assert not any(2 in span for span in spans), f"'ab' (token 2) leaked into a span: {spans}"


def test_expand_batch_with_word_trie_produces_multi_token_nodes():
    trie = WordTrie(["running", "cat", "dog"])
    graph = _word_growth_graph(trie, anchor_tokens=[0])
    graph.spawn_first_children()
    graph.tick()

    multi_token_nodes = [n for n in graph.nodes.values() if len(n.tokens) > 1]
    assert multi_token_nodes, "expected at least one multi-token word node after a tick with word_trie set"

    tokens, score = graph.best_path()
    assert math.isfinite(score)

    # path_tokens must flatten a multi-token node's own span in its own
    # internal order, not scrambled by the leaf-to-anchor walk that
    # assembles the full path -- check directly against every multi-token
    # node rather than best_path() (which may not have selected any of
    # their leaves). Where the node's own span lands depends on
    # direction: FORWARD reverses the walked spans (so the queried
    # leaf's own span, appended first in the raw walk, ends up last),
    # BACKWARD does not (leaf-first is already reading order -- see
    # path_tokens's own docstring), so the leaf's span stays first.
    for node in multi_token_nodes:
        path, direction = graph.path_tokens(node.id)
        span = node.tokens
        if direction is Direction.FORWARD:
            assert path[-len(span):] == span
        else:
            assert path[:len(span)] == span


def test_expand_batch_without_word_trie_keeps_single_token_nodes():
    graph = _word_growth_graph(word_trie=None, anchor_tokens=[0])
    graph.spawn_first_children()
    graph.tick()
    for n in graph.nodes.values():
        if n.id != graph.anchor_id:
            assert len(n.tokens) == 1


def _add_hand_node(graph, parent_id, tok_id, evidence, depth):
    nid = graph._alloc_id()
    parent = graph.nodes[parent_id]
    cum = parent.cumulative_evidence + evidence
    graph.nodes[nid] = FluxNode(
        id=nid, tokens=[tok_id], direction=Direction.FORWARD, parent_id=parent_id,
        depth=depth, local_evidence=evidence, pressure=1.0,
        cumulative_evidence=cum, rollup_mean=cum / depth,
    )
    parent.children_ids.append(nid)
    return nid


def test_auxin_disabled_by_default_is_a_true_noop():
    graph = _build_graph()
    graph.config.auxin_suppression = 0.0
    anchor_id = graph.seed([0])
    leaf_id = _add_hand_node(graph, anchor_id, 1, -5.0, 1)
    graph._diffuse_auxin()
    assert graph.nodes[leaf_id].auxin_level == 0.0
    assert graph._effective_branch_factor(graph.nodes[leaf_id]) == graph.config.branch_factor


def test_auxin_uses_centerward_geometry_for_parentless_backward_tips():
    """Regression for the radar's ``KeyError(None)`` on its first tick.

    A backward-farthest node has no causal parent by definition. Auxin's
    radial lineage must therefore reach centerward through its causal
    child rather than indexing the legacy primary parent.
    """
    graph = _build_graph(branch_factor=2)
    graph.config.auxin_suppression = 0.05
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    backward_tips = [
        graph.nodes[parent_id] for parent_id in graph.nodes[anchor_id].parent_ids
    ]
    assert backward_tips
    assert all(node.parent_id is None for node in backward_tips)

    graph._digest()
    graph._diffuse_auxin()

    assert all(graph._centerward_neighbors(node.id) == [anchor_id] for node in backward_tips)
    assert all(math.isfinite(node.auxin_level) for node in backward_tips)


def test_first_fluid_tick_with_auxin_and_bidirectional_seed_does_not_null_lookup():
    graph = _build_graph(branch_factor=2, compute_budget=2)
    graph.config.auxin_suppression = 0.05
    graph.config.graph_auditor_enabled = True
    graph.seed([0])
    graph.spawn_first_children()

    graph.tick()

    assert graph.tick_count == 1
    assert graph.nodes[graph.anchor_id].burned is False


def test_auxin_suppresses_a_weak_branch_more_than_its_strong_sibling():
    # A strong, near-certain tip should barely suppress itself (real apical
    # dominance doesn't stunt the leader shoot), while its weak sibling
    # branch -- competing for the same joint -- gets suppressed hard.
    graph = _build_graph(branch_factor=4)
    graph.config.auxin_suppression = 2.0
    graph.config.auxin_decay = 0.6
    anchor_id = graph.seed([0])

    weak_id = _add_hand_node(graph, anchor_id, 1, -5.0, 1)
    strong_id = _add_hand_node(graph, anchor_id, 2, -0.01, 1)

    graph._digest()
    graph._diffuse_auxin()

    weak = graph.nodes[weak_id]
    strong = graph.nodes[strong_id]
    assert strong.auxin_level < weak.auxin_level

    eff_weak = graph._effective_branch_factor(weak)
    eff_strong = graph._effective_branch_factor(strong)
    assert eff_strong == graph.config.branch_factor  # unsuppressed -- it's the dominant tip
    assert eff_weak < eff_strong  # suppressed by the strong sibling


def test_auxin_suppression_decays_with_distance():
    # A strong tip's suppression reaches branches beyond its own direct
    # siblings (the requested "and so does everything downstream of it"),
    # but weaker the further away they are.
    graph = _build_graph(branch_factor=4)
    graph.config.auxin_suppression = 2.0
    graph.config.auxin_decay = 0.6
    anchor_id = graph.seed([0])

    # anchor -> A -> A1 (a distant cousin of the strong tip)
    a_id = _add_hand_node(graph, anchor_id, 1, -1.0, 1)
    a1_id = _add_hand_node(graph, a_id, 1, -1.0, 2)

    # anchor -> B -> {B1 (strong tip), B2 (B1's direct sibling)}
    b_id = _add_hand_node(graph, anchor_id, 2, -1.0, 1)
    _add_hand_node(graph, b_id, 2, -0.01, 2)  # the strong tip itself
    b2_id = _add_hand_node(graph, b_id, 3, -1.0, 2)

    graph._digest()
    graph._diffuse_auxin()

    a1 = graph.nodes[a1_id]
    b2 = graph.nodes[b2_id]
    assert a1.auxin_level > 0.0  # suppression still reaches the distant branch
    assert b2.auxin_level > a1.auxin_level  # but more weakly than the close one


def test_height_is_signed_depth_by_direction():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    assert graph.nodes[anchor_id].height == 0.0

    fwd_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph.nodes[fwd_id].direction = Direction.FORWARD
    assert graph.nodes[fwd_id].height == 1.0

    bwd_id = _add_hand_node(graph, anchor_id, 2, -0.5, 3)
    graph.nodes[bwd_id].direction = Direction.BACKWARD
    graph.nodes[bwd_id].level = -3
    assert graph.nodes[bwd_id].height == -3.0


def test_head_pressure_disabled_by_default_is_a_true_noop():
    # With the coefficient at 0, distance from the anchor must make no
    # pressure difference at all -- a near and a far node with identical
    # local_evidence should end up with identical pressure.
    graph = _build_graph()
    graph.config.head_pressure_coefficient = 0.0
    anchor_id = graph.seed([0])

    near_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph.nodes[near_id].direction = Direction.FORWARD
    far_id = _add_hand_node(graph, anchor_id, 1, -0.5, 5)
    graph.nodes[far_id].direction = Direction.FORWARD

    graph._settle_circuit()
    assert math.isclose(graph.nodes[near_id].pressure, graph.nodes[far_id].pressure, rel_tol=1e-9)


def test_head_pressure_costs_forward_and_backward_equally_at_equal_distance():
    graph = _build_graph()
    graph.config.head_pressure_coefficient = 0.05
    anchor_id = graph.seed([0])

    fwd_id = _add_hand_node(graph, anchor_id, 1, -0.5, 3)
    graph.nodes[fwd_id].direction = Direction.FORWARD
    bwd_id = _add_hand_node(graph, anchor_id, 2, -0.5, 3)
    graph.nodes[bwd_id].direction = Direction.BACKWARD

    graph._settle_circuit()
    assert math.isclose(graph.nodes[fwd_id].pressure, graph.nodes[bwd_id].pressure, rel_tol=1e-9)


def test_head_pressure_increases_with_distance_from_anchor():
    graph = _build_graph()
    graph.config.head_pressure_coefficient = 0.05
    anchor_id = graph.seed([0])

    near_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph.nodes[near_id].direction = Direction.FORWARD
    far_id = _add_hand_node(graph, anchor_id, 1, -0.5, 5)
    graph.nodes[far_id].direction = Direction.FORWARD

    graph._settle_circuit()
    assert graph.nodes[near_id].pressure > graph.nodes[far_id].pressure


def test_head_pressure_never_drives_pressure_negative():
    graph = _build_graph()
    graph.config.head_pressure_coefficient = 1000.0
    anchor_id = graph.seed([0])
    node_id = _add_hand_node(graph, anchor_id, 1, -0.5, 5)
    graph.nodes[node_id].direction = Direction.FORWARD
    graph._settle_circuit()
    assert graph.nodes[node_id].pressure >= 0.0


def test_direction_branch_factor_falls_back_to_shared_default():
    graph = _build_graph(branch_factor=5)
    assert graph._direction_branch_factor(Direction.FORWARD) == 5
    assert graph._direction_branch_factor(Direction.BACKWARD) == 5


def test_direction_branch_factor_override_applies_to_only_that_direction():
    graph = _build_graph(branch_factor=5)
    graph.config.forward_branch_factor = 7
    assert graph._direction_branch_factor(Direction.FORWARD) == 7
    assert graph._direction_branch_factor(Direction.BACKWARD) == 5


def test_direction_hot_loop_depth_falls_back_to_shared_default():
    graph = _build_graph()
    graph.config.hot_loop_depth = 3
    assert graph._direction_hot_loop_depth(Direction.FORWARD) == 3
    assert graph._direction_hot_loop_depth(Direction.BACKWARD) == 3


def test_direction_hot_loop_depth_override_applies_to_only_that_direction():
    graph = _build_graph()
    graph.config.hot_loop_depth = 1
    graph.config.backward_hot_loop_depth = 4
    assert graph._direction_hot_loop_depth(Direction.FORWARD) == 1
    assert graph._direction_hot_loop_depth(Direction.BACKWARD) == 4


def test_top_p_keep_count_stops_once_cumulative_mass_is_reached():
    graph = _build_graph()
    scores = [math.log(p) for p in (0.5, 0.3, 0.15, 0.05)]
    assert graph._top_p_keep_count(scores, top_p=0.5) == 1
    assert graph._top_p_keep_count(scores, top_p=0.7) == 2
    assert graph._top_p_keep_count(scores, top_p=0.79) == 2
    assert graph._top_p_keep_count(scores, top_p=0.81) == 3
    assert graph._top_p_keep_count(scores, top_p=0.99) == 4


def test_top_p_keep_count_keeps_at_least_one_even_if_it_alone_exceeds_top_p():
    graph = _build_graph()
    scores = [math.log(0.95), math.log(0.05)]
    assert graph._top_p_keep_count(scores, top_p=0.5) == 1


def test_top_p_keep_count_empty_input_keeps_nothing():
    graph = _build_graph()
    assert graph._top_p_keep_count([], top_p=0.9) == 0


def test_resolve_keep_count_topk_mode_matches_effective_branch_factor():
    graph = _build_graph(branch_factor=3)
    anchor_id = graph.seed([0])
    node_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    node = graph.nodes[node_id]
    spans = [([i], math.log(0.4)) for i in range(5)]
    assert graph._resolve_keep_count(node, spans) == graph._effective_branch_factor(node) == 3


def test_resolve_keep_count_topp_mode_ignores_branch_factor_and_uses_top_p():
    graph = _build_graph(branch_factor=1)  # deliberately tiny, to prove topp isn't capped by it
    graph.config.forward_top_p = 0.81
    graph.config.forward_selection_mode = "topp"
    anchor_id = graph.seed([0])
    node_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    node = graph.nodes[node_id]
    node.direction = Direction.FORWARD
    spans = [([i], s) for i, s in enumerate(math.log(p) for p in (0.5, 0.3, 0.15, 0.05))]
    assert graph._resolve_keep_count(node, spans) == 3  # would be 1 under topk with branch_factor=1


def test_resolve_keep_count_topp_mode_capped_by_auxin_suppression_when_on():
    graph = _build_graph(branch_factor=10)
    graph.config.forward_selection_mode = "topp"
    graph.config.forward_top_p = 0.999  # would otherwise keep everything
    graph.config.auxin_suppression = 1.0
    anchor_id = graph.seed([0])
    node_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    node = graph.nodes[node_id]
    node.direction = Direction.FORWARD
    node.auxin_level = 100.0  # heavy ambient suppression -> effective_branch_factor collapses to 1
    spans = [([i], math.log(0.1)) for i in range(10)]
    assert graph._resolve_keep_count(node, spans) == graph._effective_branch_factor(node) == 1


def test_expand_top_pressure_nodes_explicit_per_direction_budget_is_a_hard_cap_no_redistribution():
    # Forward gets a hard cap of 1 even though backward has room left in
    # compute_budget_per_tick and forward has more eligible candidates --
    # explicit per-direction budgets don't redistribute leftover the way
    # the default floor-split does.
    graph = _build_graph(branch_factor=1, compute_budget=10)
    graph.config.forward_budget_per_tick = 1
    graph.config.backward_budget_per_tick = 0
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    forward_leaves_before = [n for n in graph.nodes.values() if not n.burned and n.direction is Direction.FORWARD]
    assert len(forward_leaves_before) >= 1

    graph._expand_top_pressure_nodes()

    live_depths_by_dir = {}
    for n in graph.nodes.values():
        if n.burned or n.direction is None:
            continue
        live_depths_by_dir.setdefault(n.direction, []).append(n.depth)
    # Only forward should have grown (backward_budget_per_tick=0 means no
    # backward candidates were selected at all this tick).
    assert max(live_depths_by_dir.get(Direction.FORWARD, [0])) == 2
    assert max(live_depths_by_dir.get(Direction.BACKWARD, [0])) == 1  # unchanged from spawn_first_children


def test_expand_top_pressure_nodes_reseeds_anchor_after_a_total_stall():
    """Real dead-end bug: _expandable_nodes() never includes the anchor
    (no direction, so _expand_batch can't score it), and nothing else
    ever re-triggers growth on it after spawn_first_children()'s one-time
    call. If every forward/backward node ever burns away, leaving only
    the anchor, the graph must be able to recover -- not sit there
    forever with both candidate pools permanently empty.
    """
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    for child_id in list(graph.nodes[anchor_id].children_ids):
        graph._burn(child_id)
    for parent_id in list(graph.nodes[anchor_id].parent_ids):
        graph._burn(parent_id)
    assert graph._live_children(anchor_id) == []
    assert graph._live_parents(anchor_id) == []

    graph._expand_top_pressure_nodes()

    live_children = graph._live_children(anchor_id)
    live_parents = graph._live_parents(anchor_id)
    assert live_children and live_parents
    assert {graph.nodes[c].direction for c in live_children} == {Direction.FORWARD}
    assert {graph.nodes[p].direction for p in live_parents} == {Direction.BACKWARD}


def test_anchor_urgently_regrows_a_missing_backward_circulatory_side():
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    for parent_id in list(graph.nodes[anchor_id].parent_ids):
        graph._burn(parent_id)
    forward_ids_before = {
        n.id for n in graph.nodes.values()
        if not n.burned and n.direction is Direction.FORWARD
    }
    assert graph._live_children(anchor_id)
    assert graph._live_parents(anchor_id) == []

    graph._expand_top_pressure_nodes()

    assert graph._live_parents(anchor_id)
    assert {
        n.id for n in graph.nodes.values()
        if not n.burned and n.direction is Direction.FORWARD
    } == forward_ids_before


def test_anchor_urgently_regrows_a_missing_forward_circulatory_side():
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    for child_id in list(graph.nodes[anchor_id].children_ids):
        graph._burn(child_id)
    backward_ids_before = {
        n.id for n in graph.nodes.values()
        if not n.burned and n.direction is Direction.BACKWARD
    }
    assert graph._live_parents(anchor_id)
    assert graph._live_children(anchor_id) == []

    graph._expand_top_pressure_nodes()

    assert graph._live_children(anchor_id)
    assert {
        n.id for n in graph.nodes.values()
        if not n.burned and n.direction is Direction.BACKWARD
    } == backward_ids_before


def test_expand_top_pressure_nodes_does_not_reseed_when_both_anchor_sides_are_intact():
    graph = _build_graph(branch_factor=2)
    anchor_id = graph.seed([0])
    graph.spawn_first_children()
    children_before = set(graph.nodes[anchor_id].children_ids)
    parents_before = set(graph.nodes[anchor_id].parent_ids)
    assert children_before and parents_before

    graph._expand_top_pressure_nodes()

    # The anchor's direct connections are untouched by an ordinary tick;
    # new growth happens farther out, not by re-seeding either side.
    assert set(graph.nodes[anchor_id].children_ids) == children_before
    assert set(graph.nodes[anchor_id].parent_ids) == parents_before


def test_return_conductance_scale_default_matches_edge_conductance_exactly():
    graph = _build_graph()
    assert graph.config.return_conductance_scale == 1.0
    anchor_id = graph.seed([0])
    child_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph.nodes[child_id].direction = Direction.FORWARD

    assert graph._directional_conductance(anchor_id, child_id) == graph._edge_conductance(anchor_id, child_id)
    assert graph._directional_conductance(child_id, anchor_id) == graph._edge_conductance(child_id, anchor_id)


def test_return_conductance_scale_only_dampens_the_return_leg():
    graph = _build_graph()
    graph.config.return_conductance_scale = 0.25
    anchor_id = graph.seed([0])
    child_id = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph.nodes[child_id].direction = Direction.FORWARD

    # Delivery: parent (anchor) flowing out to child -- child's own
    # inflow-from-parent term must stay exactly the base value.
    delivery = graph._directional_conductance(child_id, anchor_id)
    assert delivery == graph._edge_conductance(child_id, anchor_id)

    # Return: child flowing back to parent -- the anchor's inflow-from-
    # child term must be scaled down.
    ret = graph._directional_conductance(anchor_id, child_id)
    assert math.isclose(ret, graph._edge_conductance(anchor_id, child_id) * 0.25)


def test_return_conductance_scale_dampens_how_much_a_strong_child_inflates_its_parent():
    strong_child_evidence = -0.05  # near-zero evidence -> local_value close to 1.0, a strong claim

    full_return = _build_graph()
    full_anchor = full_return.seed([0])
    weak_parent_id = _add_hand_node(full_return, full_anchor, 1, -3.0, 1)
    full_return.nodes[weak_parent_id].direction = Direction.FORWARD
    strong_child_id = _add_hand_node(full_return, weak_parent_id, 2, strong_child_evidence, 2)
    full_return.nodes[strong_child_id].direction = Direction.FORWARD
    full_return._settle_circuit()

    damped_return = _build_graph()
    damped_return.config.return_conductance_scale = 0.1
    damped_anchor = damped_return.seed([0])
    damped_weak_parent_id = _add_hand_node(damped_return, damped_anchor, 1, -3.0, 1)
    damped_return.nodes[damped_weak_parent_id].direction = Direction.FORWARD
    damped_strong_child_id = _add_hand_node(damped_return, damped_weak_parent_id, 2, strong_child_evidence, 2)
    damped_return.nodes[damped_strong_child_id].direction = Direction.FORWARD
    damped_return._settle_circuit()

    # Same weak parent, same strong child -- but with the return leg
    # dampened, the parent's settled pressure ends up lower: the child's
    # strength reports back less than it would through a symmetric edge.
    assert full_return.nodes[weak_parent_id].pressure > damped_return.nodes[damped_weak_parent_id].pressure


def test_inflow_is_weighted_by_conductance_not_diluted_by_raw_neighbor_count():
    """Pins down the actual bug: inflow used to normalize by (1 + neighbor
    count), so adding a neighbor diluted the average purely by existing,
    regardless of how weak its connection was. A near-zero-conductance
    extra neighbor must barely move inflow at all now -- a thin, almost-
    disconnected wire shouldn't meaningfully dilute a strong one, the way
    it would in any real resistor network.
    """
    solo = _build_graph()
    anchor_id = solo.seed([0])
    hub_id = _add_hand_node(solo, anchor_id, 1, -0.5, 1)
    strong_child_id = _add_hand_node(solo, hub_id, 2, -0.1, 2)
    solo.nodes[strong_child_id].pressure = 2.0
    solo._update_pressures()
    solo_pressure = solo.nodes[hub_id].pressure

    plus_weak = _build_graph()
    anchor_id2 = plus_weak.seed([0])
    hub_id2 = _add_hand_node(plus_weak, anchor_id2, 1, -0.5, 1)
    strong_child_id2 = _add_hand_node(plus_weak, hub_id2, 2, -0.1, 2)
    plus_weak.nodes[strong_child_id2].pressure = 2.0
    # A near-zero-evidence -- wait, near *negative-infinity* evidence --
    # child: local_value = exp(evidence) is essentially 0, so its
    # conductance is essentially 0 too.
    weak_child_id = _add_hand_node(plus_weak, hub_id2, 3, -50.0, 2)
    plus_weak.nodes[weak_child_id].pressure = 2.0
    plus_weak._update_pressures()
    plus_weak_pressure = plus_weak.nodes[hub_id2].pressure

    assert math.isclose(solo_pressure, plus_weak_pressure, rel_tol=1e-3)


def test_inflow_rewards_more_strong_connections_not_just_one():
    # A hub with three well-supported neighbors ends up with more settled
    # pressure than an otherwise-identical node with just one -- real
    # connectivity is a genuine advantage now, not something that gets
    # diluted the more neighbors a node happens to have.
    solo = _build_graph()
    anchor_id = solo.seed([0])
    hub_id = _add_hand_node(solo, anchor_id, 1, -0.5, 1)
    child_id = _add_hand_node(solo, hub_id, 2, -0.1, 2)
    solo.nodes[child_id].pressure = 2.0
    solo._update_pressures()
    solo_pressure = solo.nodes[hub_id].pressure

    trio = _build_graph()
    anchor_id2 = trio.seed([0])
    hub_id2 = _add_hand_node(trio, anchor_id2, 1, -0.5, 1)
    for tok in (10, 11, 12):
        cid = _add_hand_node(trio, hub_id2, tok, -0.1, 2)
        trio.nodes[cid].pressure = 2.0
    trio._update_pressures()
    trio_pressure = trio.nodes[hub_id2].pressure

    assert trio_pressure > solo_pressure


def test_balance_weight_disabled_by_default_is_a_true_noop_on_settled_pressure():
    graph = _build_graph()
    assert graph.config.balance_weight == 0.0
    anchor_id = graph.seed([0])

    far_fwd_id = _add_hand_node(graph, anchor_id, 1, -0.5, 8)
    graph.nodes[far_fwd_id].direction = Direction.FORWARD
    near_bwd_id = _add_hand_node(graph, anchor_id, 2, -0.5, 1)
    graph.nodes[near_bwd_id].direction = Direction.BACKWARD

    baseline = _build_graph()
    baseline_anchor = baseline.seed([0])
    baseline_bwd_id = _add_hand_node(baseline, baseline_anchor, 2, -0.5, 1)
    baseline.nodes[baseline_bwd_id].direction = Direction.BACKWARD

    graph._settle_circuit()
    baseline._settle_circuit()
    # Forward being way out ahead must change nothing about the backward
    # node's settled pressure while the knob is off.
    assert math.isclose(graph.nodes[near_bwd_id].pressure, baseline.nodes[baseline_bwd_id].pressure, rel_tol=1e-9)


def test_balance_weight_raises_settled_pressure_of_the_lagging_side():
    graph = _build_graph()
    graph.config.balance_weight = 0.1
    anchor_id = graph.seed([0])

    far_fwd_id = _add_hand_node(graph, anchor_id, 1, -0.5, 8)
    graph.nodes[far_fwd_id].direction = Direction.FORWARD
    near_bwd_id = _add_hand_node(graph, anchor_id, 2, -0.5, 1)
    graph.nodes[near_bwd_id].direction = Direction.BACKWARD

    baseline = _build_graph()
    baseline_anchor = baseline.seed([0])
    baseline_bwd_id = _add_hand_node(baseline, baseline_anchor, 2, -0.5, 1)
    baseline.nodes[baseline_bwd_id].direction = Direction.BACKWARD

    graph._settle_circuit()
    baseline._settle_circuit()
    # Same backward node, but this time forward is way out ahead with the
    # knob on -- its settled pressure should come out higher than the
    # exact same node settled with nothing to lag behind.
    assert graph.nodes[near_bwd_id].pressure > baseline.nodes[baseline_bwd_id].pressure
    # The far-ahead forward node itself is already leading -- it gets no
    # gain from its own knob.
    far_baseline = _build_graph()
    far_baseline_anchor = far_baseline.seed([0])
    far_baseline_fwd_id = _add_hand_node(far_baseline, far_baseline_anchor, 1, -0.5, 8)
    far_baseline.nodes[far_baseline_fwd_id].direction = Direction.FORWARD
    far_baseline._settle_circuit()
    assert math.isclose(graph.nodes[far_fwd_id].pressure, far_baseline.nodes[far_baseline_fwd_id].pressure, rel_tol=1e-9)


def test_balance_weight_never_drives_pressure_negative():
    graph = _build_graph()
    graph.config.balance_weight = 1000.0
    graph.config.head_pressure_coefficient = 1000.0
    anchor_id = graph.seed([0])
    far_fwd_id = _add_hand_node(graph, anchor_id, 1, -0.5, 8)
    graph.nodes[far_fwd_id].direction = Direction.FORWARD
    near_bwd_id = _add_hand_node(graph, anchor_id, 2, -0.5, 1)
    graph.nodes[near_bwd_id].direction = Direction.BACKWARD
    graph._settle_circuit()
    assert graph.nodes[far_fwd_id].pressure >= 0.0
    assert graph.nodes[near_bwd_id].pressure >= 0.0


# ---------------------------------------------------------------------------
# Root displacement / re-rooting
# ---------------------------------------------------------------------------

def _add_node(graph, parent_id, tok_id, evidence, depth, direction=Direction.FORWARD, pressure=1.0):
    nid = graph._alloc_id()
    parent = graph.nodes[parent_id]
    cum = parent.cumulative_evidence + evidence
    graph.nodes[nid] = FluxNode(
        id=nid, tokens=[tok_id], direction=direction, parent_id=parent_id,
        depth=depth, level=depth if direction is Direction.FORWARD else -depth,
        center_id=parent.center_id, local_evidence=evidence, pressure=pressure,
        cumulative_evidence=cum, rollup_mean=cum / depth if depth else 0.0,
    )
    parent.children_ids.append(nid)
    return nid


def test_reroot_simple_one_hop_moves_focus_without_reversing_edge():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=2.0)

    graph._reroot(f1_id)

    assert graph.anchor_id == f1_id
    f1 = graph.nodes[f1_id]
    assert f1.direction is None
    assert f1.parent_id == anchor_id
    assert f1.depth == 0
    assert f1.local_evidence == 0.0
    assert f1.tokens == []
    assert graph.anchor_tokens == [1]

    old_anchor = graph.nodes[anchor_id]
    assert old_anchor.parent_id is None
    assert old_anchor.direction is Direction.BACKWARD
    assert old_anchor.depth == 1
    assert old_anchor.tokens == [99]
    assert f1_id in old_anchor.children_ids
    assert anchor_id not in f1.children_ids


def test_reroot_multi_hop_rebases_levels_without_reversing_the_path():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2_id = _add_node(graph, f1_id, 2, -0.1, 2, direction=Direction.FORWARD, pressure=1.0)
    f3_id = _add_node(graph, f2_id, 3, -0.1, 3, direction=Direction.FORWARD, pressure=3.0)

    graph._reroot(f3_id)

    assert graph.anchor_id == f3_id
    f3 = graph.nodes[f3_id]
    assert f3.parent_id == f2_id and f3.direction is None and f3.depth == 0

    f2 = graph.nodes[f2_id]
    assert f2.parent_id == f1_id and f2.direction is Direction.BACKWARD and f2.depth == 1

    f1 = graph.nodes[f1_id]
    assert f1.parent_id == anchor_id and f1.direction is Direction.BACKWARD and f1.depth == 2

    old_anchor = graph.nodes[anchor_id]
    assert old_anchor.parent_id is None and old_anchor.direction is Direction.BACKWARD
    assert old_anchor.depth == 3
    assert old_anchor.tokens == [99]

    assert f1_id in old_anchor.children_ids
    assert f2_id in f1.children_ids
    assert f3_id in f2.children_ids


def test_reroot_keeps_orthogonal_sibling_at_intermediate_ancestor_fully_live():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2_id = _add_node(graph, f1_id, 2, -0.1, 2, direction=Direction.FORWARD, pressure=3.0)
    f2b_id = _add_node(graph, f1_id, 20, -0.1, 2, direction=Direction.FORWARD, pressure=0.5)

    graph._reroot(f2_id)

    # f2b stays attached under f1. Since it is one causal step forward of
    # f1 just like the new focus f2, it occupies the same seed layer.
    assert f2b_id in graph.nodes[f1_id].children_ids
    f2b = graph.nodes[f2b_id]
    assert f2b.parent_id == f1_id
    assert f2b.direction is None
    assert f2b.tokens == [20]
    assert f2b_id in graph.nodes

    assert f2b_id in graph.orthogonal_node_ids()


def test_restore_rebases_stale_leaf_levels_with_their_backward_ancestor():
    source = _build_graph()
    anchor_id = source.seed([99])
    f1_id = _add_node(
        source, anchor_id, 1, -0.1, 1,
        direction=Direction.FORWARD, pressure=1.0,
    )
    leaf_id = _add_node(
        source, f1_id, 20, -0.1, 2,
        direction=Direction.FORWARD, pressure=0.5,
    )
    f2_id = _add_node(
        source, f1_id, 2, -0.1, 2,
        direction=Direction.FORWARD, pressure=1.0,
    )
    f3_id = _add_node(
        source, f2_id, 3, -0.1, 3,
        direction=Direction.FORWARD, pressure=3.0,
    )
    source._reroot(f3_id)

    # A stale save can contain the pre-reroot display coordinates even
    # though causal ownership is already the current legal topology.
    source.nodes[f1_id].level = 1
    source.nodes[f1_id].depth = 1
    source.nodes[f1_id].direction = Direction.FORWARD
    source.nodes[leaf_id].level = 2
    source.nodes[leaf_id].depth = 2
    source.nodes[leaf_id].direction = Direction.FORWARD

    restored = _build_graph()
    restored.restore_state(
        source.nodes, f3_id, source.anchor_tokens, source.tick_count,
        anchor_local_evidence=source.anchor_local_evidence,
    )

    assert restored.nodes[f1_id].level == -2
    assert restored.nodes[f1_id].direction is Direction.BACKWARD
    assert restored.nodes[leaf_id].level == -1
    assert restored.nodes[leaf_id].depth == 1
    assert restored.nodes[leaf_id].direction is Direction.BACKWARD
    assert (f1_id, leaf_id) in restored.edges


def test_restore_rejects_contradictory_causal_levels_without_pruning_an_edge():
    source = _build_graph()
    anchor_id = source.seed([99])
    first_id = _add_node(
        source, anchor_id, 1, -0.1, 1,
        direction=Direction.FORWARD, pressure=1.0,
    )
    second_id = _add_node(
        source, first_id, 2, -0.1, 2,
        direction=Direction.FORWARD, pressure=1.0,
    )
    # This direct shortcut says second is simultaneously one and two causal
    # hops from the seed. Current _connect cannot create it; emulate a
    # damaged legacy save and require restoration to reject it wholesale.
    source.nodes[anchor_id].children_ids.append(second_id)
    source.nodes[second_id].parent_ids.append(anchor_id)

    restored = _build_graph()
    try:
        restored.restore_state(
            source.nodes, anchor_id, [99], source.tick_count,
        )
        assert False, "expected contradictory saved topology to be rejected"
    except ValueError as error:
        assert "inconsistent causal levels" in str(error)

    assert second_id in source.nodes[anchor_id].children_ids
    assert anchor_id in source.nodes[second_id].parent_ids


def test_restore_rebuilds_backward_pipes_and_audits_words_in_reading_order():
    source = _build_graph()
    anchor_id = source.seed([0])
    source._attach_backward_parents(anchor_id, [-0.1], [[10]])
    near_id = source.nodes[anchor_id].parent_ids[-1]
    source._attach_backward_parents(near_id, [-0.2], [[20]])
    far_id = source.nodes[near_id].parent_ids[-1]

    restored = _build_graph()
    restored.restore_state(
        source.nodes, anchor_id, [0], source.tick_count,
    )
    restored._run_graph_auditor()

    assert restored.edges[(far_id, near_id)].formation == "prefix_beam"
    assert restored.edges[(near_id, anchor_id)].formation == "prefix_beam"
    assert restored.traversals[(far_id, anchor_id)].node_ids == [
        far_id, near_id, anchor_id,
    ]
    tokens, direction = restored.path_tokens(far_id)
    assert direction is Direction.BACKWARD
    assert tokens == [20, 10]


def test_soft_traversal_physiology_learns_structures_and_subedge_archetype():
    graph = _build_graph(branch_factor=2)
    graph.config.physiology_learning_enabled = True
    graph.config.physiology_learning_rate = 0.2
    graph.config.physiology_resource_cost = 0.1
    graph.config.physiology_initial_opening = 0.6
    graph.seed([0])
    graph.spawn_first_children()
    graph._run_graph_auditor()
    graph._ensure_physiology_parameters()
    before = {
        key: float(parameter.detach())
        for key, parameter in graph.physiology_parameters.items()
    }

    graph._learn_physiology()

    after = {
        key: float(parameter.detach())
        for key, parameter in graph.physiology_parameters.items()
    }
    assert {"edge", "node", "heart", "archetype"} <= {
        key.split(":")[0] for key in before
    }
    assert not any(key.startswith("traversal:") for key in before)
    changed = {
        key for key in before
        if after[key] != before[key]
    }
    assert any(key.startswith("archetype:subedge:") for key in changed)
    assert any(not key.startswith("archetype:") for key in changed)
    assert any(
        key.startswith("archetype:subedge:")
        and not key.endswith(":bias")
        and after[key] != 0.0
        for key in after
    )
    assert graph.physiology_steps == 1
    assert graph.physiology_last_loss is not None
    assert graph.physiology_best_traversal is not None
    state = graph.physiology_state()
    assert state["parameter_count"] == len(before)
    assert state["archetype_parameter_count"] == 16


def test_physiology_logits_round_trip_with_fluid_state():
    graph = _build_graph(branch_factor=1)
    graph.config.physiology_learning_enabled = True
    graph.seed([0])
    graph.spawn_first_children()
    graph._run_graph_auditor()
    graph._learn_physiology()
    state = graph.export_fluid_state()
    expected = {
        key: float(parameter.detach())
        for key, parameter in graph.physiology_parameters.items()
    }

    restored = _build_graph(branch_factor=1)
    restored.config.physiology_learning_enabled = True
    restored.seed([0])
    restored.import_fluid_state(state)

    assert {
        key: float(parameter.detach())
        for key, parameter in restored.physiology_parameters.items()
    } == expected
    assert restored.physiology_steps == graph.physiology_steps
    assert restored.physiology_best_traversal == graph.physiology_best_traversal


def test_pytorch_model_wrapper_can_enable_gradient_tracked_forwards():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 3)
            self.projection = torch.nn.Linear(3, 5)

        def forward(self, input_ids, attention_mask, **kwargs):
            return {"logits": self.projection(self.embedding(input_ids))}

    wrapper = PyTorchModelWrapper(TinyModel())
    tokens = torch.tensor([[1, 2]])
    mask = torch.ones_like(tokens)
    assert not wrapper.forward(tokens, mask)["logits"].requires_grad

    wrapper.set_gradient_tracking(True)
    logits = wrapper.forward(tokens, mask)["logits"]

    assert logits.requires_grad
    assert logits.grad_fn is not None


def test_reroot_preserves_global_backward_to_forward_ownership_on_both_sides():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=3.0)
    f2_id = _add_node(graph, anchor_id, 2, -0.1, 1, direction=Direction.FORWARD, pressure=0.5)
    graph._attach_backward_parents(anchor_id, [-0.1], [[3]])
    b1_id = graph.nodes[anchor_id].parent_ids[-1]

    graph._reroot(f1_id)

    orthogonal = graph.orthogonal_node_ids()
    assert f2_id in orthogonal
    assert b1_id in orthogonal

    b1 = graph.nodes[b1_id]
    assert b1.parent_id is None
    assert b1.direction is Direction.BACKWARD
    assert b1.depth == 2  # anchor shifted to depth 1, b1 one further
    old_anchor = graph.nodes[anchor_id]
    assert b1_id in old_anchor.parent_ids
    assert f2_id in old_anchor.children_ids  # still attached, just orthogonal now


def test_orthogonal_network_roots_are_the_shallowest_diverging_node():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    grandchild_id = _add_node(graph, f1_id, 2, -0.1, 2, direction=Direction.BACKWARD, pressure=1.0)
    great_grandchild_id = _add_node(graph, grandchild_id, 3, -0.1, 3, direction=Direction.BACKWARD, pressure=1.0)

    roots = graph.orthogonal_network_roots()

    assert roots[grandchild_id] == grandchild_id  # it's the divergence point itself
    assert roots[great_grandchild_id] == grandchild_id  # inherits its network's root
    assert f1_id not in roots  # never orthogonal -- direct anchor child


def test_orthogonal_network_roots_keeps_unrelated_branches_distinct():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2_id = _add_node(graph, anchor_id, 2, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    branch_a_id = _add_node(graph, f1_id, 10, -0.1, 2, direction=Direction.BACKWARD, pressure=1.0)
    branch_b_id = _add_node(graph, f2_id, 20, -0.1, 2, direction=Direction.BACKWARD, pressure=1.0)

    roots = graph.orthogonal_network_roots()

    assert roots[branch_a_id] == branch_a_id
    assert roots[branch_b_id] == branch_b_id
    assert roots[branch_a_id] != roots[branch_b_id]  # two separate networks, not merged


def test_orthogonal_network_roots_empty_when_nothing_orthogonal():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    _add_node(graph, anchor_id, 2, -0.1, 1, direction=Direction.BACKWARD, pressure=1.0)

    assert graph.orthogonal_network_roots() == {}


def test_orthogonal_network_roots_reuses_supplied_orthogonal_set():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(
        graph,
        anchor_id,
        1,
        -0.1,
        1,
        direction=Direction.FORWARD,
        pressure=1.0,
    )
    branch_id = _add_node(
        graph,
        f1_id,
        10,
        -0.1,
        2,
        direction=Direction.BACKWARD,
        pressure=1.0,
    )
    orthogonal = graph.orthogonal_node_ids()

    def fail_on_second_scan():
        raise AssertionError("orthogonal nodes were scanned twice")

    graph.orthogonal_node_ids = fail_on_second_scan

    assert graph.orthogonal_network_roots(orthogonal)[branch_id] == branch_id

def test_reroot_demoted_anchor_becomes_vulnerable_to_starvation():
    graph = _build_graph()
    graph.config.starvation_floor = 0.5
    graph.config.burn_after_ticks = 2
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=3.0)
    graph._reroot(f1_id)

    old_anchor = graph.nodes[anchor_id]
    old_anchor.pressure = 0.0
    graph._starve_and_burn()
    assert old_anchor.low_pressure_ticks == 1
    graph._starve_and_burn()
    assert old_anchor.burned is True


def test_reroot_restores_a_demoted_nodes_real_local_evidence():
    # _reset_to_anchor_invariants zeroes local_evidence for whichever node
    # is being promoted (an anchor is treated as free/certain) -- but that
    # must come back once the node is later demoted, or its real score is
    # gone forever and every future cumulative_evidence sum through it is
    # silently wrong. Needs two reroots: the first promotion's demoted
    # node (the original seed) always had local_evidence==0.0 anyway, so
    # only a *second* reroot -- demoting a node that had a real non-zero
    # score before its own promotion -- actually exercises the bug.
    graph = _build_graph()
    seed_id = graph.seed([99])
    a_id = _add_node(graph, seed_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    b_id = _add_node(graph, a_id, 2, -0.4, 2, direction=Direction.FORWARD, pressure=1.0)

    graph._reroot(b_id)  # promotes b -- its real -0.4 must be stashed
    assert graph.nodes[b_id].local_evidence == 0.0  # zeroed while it holds anchor status
    assert graph.anchor_local_evidence == -0.4

    graph._reroot(a_id)  # demotes b -- its real -0.4 must come back
    assert graph.nodes[b_id].local_evidence == -0.4


def test_demoted_anchor_pressure_is_now_recomputed_not_frozen():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=3.0)
    graph._reroot(f1_id)

    old_anchor = graph.nodes[anchor_id]
    old_anchor.pressure = 999.0
    graph._update_pressures()
    assert old_anchor.pressure != 999.0


def test_maybe_reroot_triggers_when_a_node_exceeds_anchor_pressure():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    weak_id = _add_node(graph, anchor_id, 1, -5.0, 1, direction=Direction.FORWARD, pressure=0.5)
    strong_id = _add_node(graph, anchor_id, 2, -0.01, 1, direction=Direction.FORWARD, pressure=5.0)

    assert graph.anchor_id == anchor_id
    graph._maybe_reroot()

    assert graph.anchor_id == strong_id
    # weak_id is orthogonal now (its direction no longer matches its
    # parent's post-flip direction) but it's still a fully live node in
    # the same graph -- nothing was removed, and it remains eligible to
    # become anchor itself if its own pressure ever earns it.
    assert weak_id in graph.nodes
    assert weak_id in graph.orthogonal_node_ids()


def test_maybe_reroot_does_nothing_when_anchor_pressure_is_highest():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=0.5)
    graph._maybe_reroot()
    assert graph.anchor_id == anchor_id


def test_previously_orthogonal_node_can_itself_become_the_next_anchor():
    """Displacing the root doesn't remove anything from the graph -- an
    orthogonal node is still fully live and can be re-rooted to again."""
    graph = _build_graph()
    anchor_id = graph.seed([99])
    weak_id = _add_node(graph, anchor_id, 1, -5.0, 1, direction=Direction.FORWARD, pressure=0.5)
    _add_node(graph, anchor_id, 2, -0.01, 1, direction=Direction.FORWARD, pressure=5.0)

    graph._maybe_reroot()
    assert weak_id in graph.orthogonal_node_ids()

    graph.nodes[weak_id].pressure = 100.0
    graph._maybe_reroot()

    assert graph.anchor_id == weak_id
    assert weak_id not in graph.orthogonal_node_ids()  # it's the anchor now


def test_path_tokens_correct_after_reroot_includes_demoted_anchor_seed():
    graph = _build_graph()
    anchor_id = graph.seed([99])
    f1_id = _add_node(graph, anchor_id, 1, -0.1, 1, direction=Direction.FORWARD, pressure=3.0)
    f2_id = _add_node(graph, f1_id, 2, -0.1, 2, direction=Direction.FORWARD, pressure=1.0)

    graph._reroot(f1_id)

    tokens, direction = graph.path_tokens(f2_id)
    assert tokens == [2]
    assert direction is Direction.FORWARD

    old_tokens, old_direction = graph.path_tokens(anchor_id)
    assert old_tokens == [99]
    assert old_direction is Direction.BACKWARD

    seq = graph.full_sequence(forward_leaf=f2_id, backward_leaf=anchor_id)
    assert seq == [99, 1, 2]


def test_path_tokens_multi_hop_forward_reads_anchor_outward_in_append_order():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    f1 = _add_node(graph, anchor_id, 10, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2 = _add_node(graph, f1, 20, -0.1, 2, direction=Direction.FORWARD, pressure=1.0)
    f3 = _add_node(graph, f2, 30, -0.1, 3, direction=Direction.FORWARD, pressure=1.0)

    tokens, direction = graph.path_tokens(f3)
    assert direction is Direction.FORWARD
    assert tokens == [10, 20, 30]  # append order: closest to anchor first, reads left to right


def test_path_tokens_multi_hop_backward_reads_furthest_from_anchor_first():
    """Regression test for a real bug: backward growth prepends (each new
    node is grown further *away* from the anchor, extending leftward), so
    the deepest node is the leftmost word and must come first when read
    left to right -- the opposite of forward's append order. path_tokens
    used to reverse both directions identically, which silently scrambled
    any backward chain deeper than one hop (e.g. "near situated the
    ocean" instead of the intended "situated near the ocean") -- not just
    display, since this same text is what _expand_backward/_expand_batch
    feed the model as real scoring context.
    """
    graph = _build_graph()
    anchor_id = graph.seed([0])
    graph._attach_backward_parents(anchor_id, [-0.1], [[10]])
    b1 = graph.nodes[anchor_id].parent_ids[-1]
    graph._attach_backward_parents(b1, [-0.1], [[20]])
    b2 = graph.nodes[b1].parent_ids[-1]
    graph._attach_backward_parents(b2, [-0.1], [[30]])
    b3 = graph.nodes[b2].parent_ids[-1]

    tokens, direction = graph.path_tokens(b3)
    assert direction is Direction.BACKWARD
    assert tokens == [30, 20, 10]  # furthest from anchor first, reads left to right into the anchor


def test_full_sequence_reads_correctly_with_multi_hop_backward_and_forward():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    graph._attach_backward_parents(anchor_id, [-0.1], [[10]])
    b1 = graph.nodes[anchor_id].parent_ids[-1]
    graph._attach_backward_parents(b1, [-0.1], [[20]])
    b2 = graph.nodes[b1].parent_ids[-1]
    f1 = _add_node(graph, anchor_id, 40, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    f2 = _add_node(graph, f1, 50, -0.1, 2, direction=Direction.FORWARD, pressure=1.0)

    seq = graph.full_sequence(forward_leaf=f2, backward_leaf=b2)
    assert seq == [20, 10, 0, 40, 50]  # backward (far-to-near) + anchor + forward (near-to-far)


# ---------------------------------------------------------------------------
# Shared token pressure: redundant tokens weaken all instances
# ---------------------------------------------------------------------------

def test_attach_children_spawns_a_duplicate_token_at_zero_pressure():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)

    graph._attach_children(anchor_id, Direction.FORWARD, [-0.2], [[5]])
    new_id = max(graph.nodes)
    assert graph.nodes[new_id].tokens == [5]
    assert graph.nodes[new_id].pressure == 0.0


def test_attach_children_spawns_a_novel_token_at_normal_pressure():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)

    graph._attach_children(anchor_id, Direction.FORWARD, [-0.2], [[6]])
    new_id = max(graph.nodes)
    assert graph.nodes[new_id].tokens == [6]
    assert graph.nodes[new_id].pressure > 0.0


def test_attach_children_duplicate_siblings_in_the_same_batch_are_also_zeroed():
    graph = _build_graph()
    anchor_id = graph.seed([0])

    graph._attach_children(anchor_id, Direction.FORWARD, [-0.1, -0.2], [[7], [7]])
    children = [graph.nodes[c] for c in graph.nodes[anchor_id].children_ids]
    pressures = sorted(c.pressure for c in children)
    assert pressures[0] == 0.0  # the second [7] in the same batch
    assert pressures[1] > 0.0   # the first one, nothing lived at [7] yet


def test_shared_token_pressure_disabled_keeps_normal_spawn_pressure():
    graph = _build_graph()
    graph.config.shared_token_pressure_enabled = False
    anchor_id = graph.seed([0])
    _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)

    graph._attach_children(anchor_id, Direction.FORWARD, [-0.2], [[5]])
    new_id = max(graph.nodes)
    assert graph.nodes[new_id].pressure > 0.0


def test_shared_token_intrinsic_divides_claim_by_live_instance_count():
    """Pins down the actual algorithm: each member's own ordinary
    intrinsic claim (found_bonus + local_value) is divided by how many
    live instances of that token span exist. A stronger claim still ends
    up with a bigger override than a weaker one, but neither keeps its
    full original claim once there's anyone to share with -- that's the
    "regulating effect": redundancy costs something for every member,
    including the best one.
    """
    graph = _build_graph()
    anchor_id = graph.seed([0])
    strong = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    weak = _add_node(graph, anchor_id, 5, -2.0, 1, direction=Direction.FORWARD, pressure=1.0)

    overrides = graph._shared_token_intrinsic()

    strong_claim = graph.config.found_bonus + graph.nodes[strong].local_value
    weak_claim = graph.config.found_bonus + graph.nodes[weak].local_value

    assert math.isclose(overrides[strong], strong_claim / 2)
    assert math.isclose(overrides[weak], weak_claim / 2)
    assert overrides[strong] > overrides[weak]
    assert overrides[strong] < strong_claim


def test_shared_token_intrinsic_divides_evenly_across_identical_claims():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    a = _add_node(graph, anchor_id, 5, -0.5, 1, direction=Direction.FORWARD, pressure=1.0)
    b = _add_node(graph, anchor_id, 5, -0.5, 1, direction=Direction.FORWARD, pressure=1.0)
    c = _add_node(graph, anchor_id, 5, -0.5, 1, direction=Direction.FORWARD, pressure=1.0)

    overrides = graph._shared_token_intrinsic()
    claim = graph.config.found_bonus + graph.nodes[a].local_value

    assert overrides[a] == overrides[b] == overrides[c]
    assert math.isclose(overrides[a], claim / 3)


def test_shared_token_intrinsic_ignores_lone_instances():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    solo = _add_node(graph, anchor_id, 9, -0.3, 1, direction=Direction.FORWARD, pressure=1.0)

    overrides = graph._shared_token_intrinsic()

    assert solo not in overrides  # nothing to trade against -- keeps its own ordinary intrinsic


def test_shared_token_pressure_divides_settled_pressure_by_group_size():
    solo = _build_graph()
    anchor_id = solo.seed([0])
    solo_leaf = _add_node(solo, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    solo._settle_circuit()
    solo_pressure = solo.nodes[solo_leaf].pressure

    duo = _build_graph()
    anchor_id2 = duo.seed([0])
    a = _add_node(duo, anchor_id2, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    b = _add_node(duo, anchor_id2, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    duo._settle_circuit()

    assert duo.nodes[a].pressure < solo_pressure
    assert duo.nodes[b].pressure < solo_pressure
    assert abs(duo.nodes[a].pressure - duo.nodes[b].pressure) < 1e-9  # identical siblings, identical share


def test_shared_token_pressure_persists_across_multiple_settles():
    """Regression test for a real bug: an earlier design pooled shared
    pressure once, as a post-processing step *after* _settle_circuit
    reached equilibrium. But relaxation is a fixed-point iteration that
    converges to the same equilibrium regardless of its starting value --
    so the very next tick's from-scratch re-settle silently undid the
    pooling before it could have any lasting effect. Folding the division
    directly into _update_pressures (so it's part of the equilibrium
    itself, recomputed fresh every sweep from current structure) fixes
    this: repeated settling must NOT drift back toward the undivided,
    lone-instance value.
    """
    graph = _build_graph()
    anchor_id = graph.seed([0])
    a = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    graph._settle_circuit()
    first = graph.nodes[a].pressure

    graph._settle_circuit()
    second = graph.nodes[a].pressure

    assert abs(second - first) < 1e-6


def test_shared_token_pressure_ignores_lone_instances():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    solo = _add_node(graph, anchor_id, 9, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    graph._settle_circuit()

    baseline = _build_graph()
    anchor_id2 = baseline.seed([0])
    solo2 = _add_node(baseline, anchor_id2, 9, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    baseline._settle_circuit()

    assert graph.nodes[solo].pressure == baseline.nodes[solo2].pressure


def test_shared_token_pressure_ignores_burned_duplicates():
    graph = _build_graph()
    anchor_id = graph.seed([0])
    live = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    burned = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    graph.nodes[burned].burned = True
    graph._settle_circuit()

    baseline = _build_graph()
    anchor_id2 = baseline.seed([0])
    solo_leaf = _add_node(baseline, anchor_id2, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    baseline._settle_circuit()

    assert graph.nodes[live].pressure == baseline.nodes[solo_leaf].pressure  # a burned sibling doesn't count


def test_shared_token_pressure_disabled_is_a_true_noop():
    graph = _build_graph()
    graph.config.shared_token_pressure_enabled = False
    anchor_id = graph.seed([0])
    a = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    b = _add_node(graph, anchor_id, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    graph._settle_circuit()

    baseline = _build_graph()
    anchor_id2 = baseline.seed([0])
    solo_leaf = _add_node(baseline, anchor_id2, 5, -0.1, 1, direction=Direction.FORWARD, pressure=1.0)
    baseline._settle_circuit()

    assert graph.nodes[a].pressure == baseline.nodes[solo_leaf].pressure
    assert graph.nodes[b].pressure == baseline.nodes[solo_leaf].pressure


# ---------------------------------------------------------------------------
# decay_rate / population_target
# ---------------------------------------------------------------------------

def test_decay_rate_zero_is_a_true_noop():
    graph = _build_graph()
    graph.config.decay_rate = 0.0
    anchor_id = graph.seed([0])
    leaf = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph._settle_circuit()
    without_decay = graph.nodes[leaf].pressure

    graph2 = _build_graph()
    graph2.config.decay_rate = 0.0
    anchor_id2 = graph2.seed([0])
    leaf2 = _add_hand_node(graph2, anchor_id2, 1, -0.5, 1)
    graph2._settle_circuit()

    assert graph2.nodes[leaf2].pressure == without_decay


def test_decay_rate_lowers_the_settled_pressure_of_an_unsupported_node():
    baseline = _build_graph()
    anchor_id = baseline.seed([0])
    leaf = _add_hand_node(baseline, anchor_id, 1, -0.5, 1)
    baseline._settle_circuit()
    baseline_pressure = baseline.nodes[leaf].pressure

    decaying = _build_graph()
    decaying.config.decay_rate = 0.5
    anchor_id2 = decaying.seed([0])
    leaf2 = _add_hand_node(decaying, anchor_id2, 1, -0.5, 1)
    decaying._settle_circuit()

    assert decaying.nodes[leaf2].pressure < baseline_pressure


def test_population_target_scales_up_decay_when_over_target():
    under_target = _build_graph()
    under_target.config.decay_rate = 0.5
    under_target.config.population_target = 100  # nowhere near the live count below
    anchor_id = under_target.seed([0])
    leaf = _add_hand_node(under_target, anchor_id, 1, -0.5, 1)
    _add_hand_node(under_target, anchor_id, 2, -0.5, 1)
    under_target._settle_circuit()
    under_target_pressure = under_target.nodes[leaf].pressure

    over_target = _build_graph()
    over_target.config.decay_rate = 0.5
    over_target.config.population_target = 1  # both nodes below already exceed this
    anchor_id2 = over_target.seed([0])
    leaf2 = _add_hand_node(over_target, anchor_id2, 1, -0.5, 1)
    _add_hand_node(over_target, anchor_id2, 2, -0.5, 1)
    over_target._settle_circuit()

    assert over_target.nodes[leaf2].pressure < under_target_pressure


def test_population_target_alone_drives_decay_without_decay_rate():
    """Regression test for a real bug: an earlier version gated the whole
    population_target mechanism behind decay_rate > 0 (`if cfg.decay_rate
    > 0 and cfg.population_target`), so leaving decay_rate at its default
    of 0 -- the natural thing to do if population_target is the only
    control you actually touched -- made population_target a silent
    no-op no matter how far over target the graph was. It must now be
    additive: population_target drives real decay on its own.
    """
    under_target = _build_graph()
    under_target.config.decay_rate = 0.0  # deliberately left at its default
    under_target.config.population_target = 100
    anchor_id = under_target.seed([0])
    leaf = _add_hand_node(under_target, anchor_id, 1, -0.5, 1)
    _add_hand_node(under_target, anchor_id, 2, -0.5, 1)
    under_target._settle_circuit()
    under_target_pressure = under_target.nodes[leaf].pressure

    over_target = _build_graph()
    over_target.config.decay_rate = 0.0  # deliberately left at its default
    over_target.config.population_target = 1
    anchor_id2 = over_target.seed([0])
    leaf2 = _add_hand_node(over_target, anchor_id2, 1, -0.5, 1)
    _add_hand_node(over_target, anchor_id2, 2, -0.5, 1)
    over_target._settle_circuit()

    assert over_target.nodes[leaf2].pressure < under_target_pressure


def test_population_target_none_leaves_decay_rate_unscaled():
    graph = _build_graph()
    graph.config.decay_rate = 0.5
    graph.config.population_target = None
    anchor_id = graph.seed([0])
    leaf = _add_hand_node(graph, anchor_id, 1, -0.5, 1)
    graph._settle_circuit()
    with_none = graph.nodes[leaf].pressure

    graph2 = _build_graph()
    graph2.config.decay_rate = 0.5
    graph2.config.population_target = 10_000  # far above live count either way
    anchor_id2 = graph2.seed([0])
    leaf2 = _add_hand_node(graph2, anchor_id2, 1, -0.5, 1)
    graph2._settle_circuit()

    assert graph2.nodes[leaf2].pressure == with_none

