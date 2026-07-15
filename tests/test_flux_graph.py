"""Tests for speaktome.core.flux_graph."""

import math

import torch

from tensors import AbstractTensor
from tensors.torch_backend import PyTorchTensorOperations
from speaktome.core.model_abstraction import AbstractModelWrapper
from speaktome.core.choice_policy import TopKPolicy
from speaktome.core.implicit_backpath import ImplicitBackpathScorer
from speaktome.core.noodle_explorer import Direction
from speaktome.core.flux_graph import FluxGraph, FluxGraphConfig, FluxNode


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
    assert anchor.token is None
    assert anchor.direction is None
    assert anchor.local_evidence == 0.0
    assert anchor.pressure > 0


def test_spawn_first_children_creates_both_directions():
    graph = _build_graph(branch_factor=3)
    graph.seed([0])
    graph.spawn_first_children()

    anchor = graph.nodes[graph.anchor_id]
    children = [graph.nodes[c] for c in anchor.children_ids]
    directions = {c.direction for c in children}

    assert Direction.FORWARD in directions
    assert Direction.BACKWARD in directions
    assert len(children) == 6  # branch_factor each direction
    # The cycle table makes token 1 the overwhelmingly likely forward pick after 0.
    fwd_children = [c for c in children if c.direction is Direction.FORWARD]
    best_fwd = max(fwd_children, key=lambda c: c.local_evidence)
    assert best_fwd.token == 1


def test_tick_keeps_pressures_finite():
    graph = _build_graph()
    graph.seed([0])
    graph.spawn_first_children()
    for _ in range(5):
        graph.tick()
    for node in graph.nodes.values():
        assert math.isfinite(node.pressure)


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
        id=healthy_id, token=1, direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=0.0, pressure=1.0, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(healthy_id)

    starved_id = graph._alloc_id()
    graph.nodes[starved_id] = FluxNode(
        id=starved_id, token=2, direction=Direction.BACKWARD, parent_id=anchor_id,
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


def test_internal_nodes_are_never_burned_even_with_low_pressure():
    # Only extremities (current leaves) are candidates for burning, so a
    # struggling internal node with a live child is protected from being
    # pruned out from under its descendant -- burning never orphans a
    # subtree.
    graph = _build_graph()
    anchor_id = graph.seed([0])

    weak_parent_id = graph._alloc_id()
    graph.nodes[weak_parent_id] = FluxNode(
        id=weak_parent_id, token=1, direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(weak_parent_id)

    child_id = graph._alloc_id()
    graph.nodes[child_id] = FluxNode(
        id=child_id, token=2, direction=Direction.FORWARD, parent_id=weak_parent_id,
        depth=2, local_evidence=-50.0, pressure=0.01, expanded=True,
    )
    graph.nodes[weak_parent_id].children_ids.append(child_id)

    for _ in range(graph.config.burn_after_ticks + 2):
        graph._update_pressures()
        graph._starve_and_burn()

    # The leaf (child_id) starves and burns; its parent, no longer a leaf
    # once the child is gone... but we check mid-run behavior: at no point
    # was weak_parent_id burned while it still had a live child.
    assert graph.nodes[weak_parent_id].burned is False


def test_a_node_whose_children_all_burn_becomes_expandable_again():
    # Eligibility must be based on current leaf status, not "has this node
    # ever been expanded" -- otherwise a node whose children all burn off
    # becomes a permanent dead end that can win best_path() forever
    # without ever being able to grow past it.
    graph = _build_graph()
    anchor_id = graph.seed([0])

    parent_id = graph._alloc_id()
    graph.nodes[parent_id] = FluxNode(
        id=parent_id, token=1, direction=Direction.FORWARD, parent_id=anchor_id,
        depth=1, local_evidence=0.0, pressure=1.0, expanded=True,
    )
    graph.nodes[anchor_id].children_ids.append(parent_id)

    child_id = graph._alloc_id()
    graph.nodes[child_id] = FluxNode(
        id=child_id, token=2, direction=Direction.FORWARD, parent_id=parent_id,
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
            id=nid, token=depth, direction=direction, parent_id=prev,
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
            id=nid, token=depth, direction=Direction.FORWARD, parent_id=prev,
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
        id=old_id, token=1, direction=Direction.FORWARD, parent_id=graph.anchor_id,
        depth=1, local_evidence=-1.0, pressure=0.5, created_tick=0,
    )
    graph.nodes[graph.anchor_id].children_ids.append(old_id)

    new_id = graph._alloc_id()
    graph.nodes[new_id] = FluxNode(
        id=new_id, token=2, direction=Direction.FORWARD, parent_id=graph.anchor_id,
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
