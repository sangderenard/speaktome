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
    assert best_fwd.tokens[0] == 1


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


def test_internal_nodes_are_never_burned_even_with_low_pressure():
    # Only extremities (current leaves) are candidates for burning, so a
    # struggling internal node with a live child is protected from being
    # pruned out from under its descendant -- burning never orphans a
    # subtree.
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
    short_child = short_graph.nodes[short_graph.anchor_id].children_ids[0]
    assert math.isclose(short_graph.nodes[short_child].local_evidence, expected_per_token, rel_tol=1e-4)

    long_graph = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    long_graph.seed([0, 1, 2, 3, 4])
    long_graph._expand_backward(long_graph.anchor_id)
    long_child = long_graph.nodes[long_graph.anchor_id].children_ids[0]
    assert math.isclose(long_graph.nodes[long_child].local_evidence, expected_per_token, rel_tol=1e-4)


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
    bwd_old = next(c for c in g_old.nodes[g_old.anchor_id].children_ids
                   if g_old.nodes[c].direction is Direction.BACKWARD)
    g_old._expand_backward(bwd_old)
    old_result = sorted(
        (g_old.nodes[c].tokens[0], round(g_old.nodes[c].local_evidence, 6))
        for c in g_old.nodes[bwd_old].children_ids
    )

    g_new = FluxGraph(model, backpath, TopKPolicy(), ops, config=config)
    g_new.seed([0, 1])
    g_new.spawn_first_children()
    bwd_new = next(c for c in g_new.nodes[g_new.anchor_id].children_ids
                   if g_new.nodes[c].direction is Direction.BACKWARD)
    g_new._expand_batch([g_new.nodes[bwd_new]])
    new_result = sorted(
        (g_new.nodes[c].tokens[0], round(g_new.nodes[c].local_evidence, 6))
        for c in g_new.nodes[bwd_new].children_ids
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
    bwd = next(c for c in graph.nodes[graph.anchor_id].children_ids
               if graph.nodes[c].direction is Direction.BACKWARD)

    graph._expand_batch([graph.nodes[fwd], graph.nodes[bwd]])

    assert len(graph.nodes[fwd].children_ids) == config.branch_factor
    assert len(graph.nodes[bwd].children_ids) == config.branch_factor
    for c in graph.nodes[fwd].children_ids:
        assert graph.nodes[c].direction is Direction.FORWARD
    for c in graph.nodes[bwd].children_ids:
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
        bwd = next(c for c in graph.nodes[graph.anchor_id].children_ids
                   if graph.nodes[c].direction is Direction.BACKWARD)
        graph._expand_batch([graph.nodes[bwd]])
        return sorted(
            (graph.nodes[c].tokens[0], round(graph.nodes[c].local_evidence, 6))
            for c in graph.nodes[bwd].children_ids
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


def _word_growth_graph(word_trie, branch_factor=2, max_subword_steps=6, anchor_tokens=None):
    model = WordGrowthBigramModel()
    tok = WordGrowthTokenizer()
    backpath = ImplicitBackpathScorer(model, tok, writing_filter=None)
    ops = PyTorchTensorOperations(track_time=False)
    config = FluxGraphConfig(
        branch_factor=branch_factor, compute_budget_per_tick=1, no_repeat_ngram_size=None,
        word_trie=word_trie, max_subword_steps=max_subword_steps,
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
    # internal order, not reversed by the leaf-to-anchor walk that
    # assembles the full path -- check directly against one multi-token
    # node rather than best_path() (which may not have selected that
    # exact node's leaf).
    node = multi_token_nodes[0]
    path, _ = graph.path_tokens(node.id)
    span = node.tokens
    assert path[-len(span):] == span


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
