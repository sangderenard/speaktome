"""Tests for speaktome.core.implicit_backpath."""

import math

import torch

from tensors import AbstractTensor
from speaktome.core.model_abstraction import AbstractModelWrapper
from speaktome.core.implicit_backpath import ImplicitBackpathScorer
from speaktome.core.writing_token_filter import WritingTokenFilter


class BigramDummyModel(AbstractModelWrapper):
    """logits at each position depend only on the token at that position
    (a Markov-order-1 lookup table), so scores genuinely vary with which
    candidate got prepended -- unlike a position-independent dummy model,
    which would make prepend-and-rescore a no-op."""

    def __init__(self, table):
        self.table = table  # nested list, [vocab_size][vocab_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        table_t = torch.tensor(self.table, dtype=torch.float32)
        return {"logits": table_t[input_ids]}

    def get_device(self):
        return "cpu"


# A 3-token cycle: after 0 comes 1, after 1 comes 2, after 2 comes 0.
CYCLE_TABLE = [
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    [10.0, 0.0, 0.0],
]


def _log_softmax_at(row, idx):
    m = max(row)
    denom = math.log(sum(math.exp(v - m) for v in row))
    return (row[idx] - m) - denom


def test_score_candidates_matches_hand_computed_single_step():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel(CYCLE_TABLE)
    scorer = ImplicitBackpathScorer(model, tokenizer=None)

    suffix = ops.tensor([1], dtype=ops.long_dtype)
    candidates = ops.tensor([0, 1, 2], dtype=ops.long_dtype)

    scores = scorer.score_candidates(suffix, candidates)
    got = scores.tolist()
    expected = [_log_softmax_at(CYCLE_TABLE[c], 1) for c in (0, 1, 2)]

    for e, g in zip(expected, got):
        assert math.isclose(e, g, rel_tol=1e-4, abs_tol=1e-4)

    # Candidate 0 -> 1 is the only edge the cycle actually encodes.
    assert got[0] > got[1]
    assert got[0] > got[2]


def test_score_candidates_sums_log_probs_over_a_longer_suffix():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel(CYCLE_TABLE)
    scorer = ImplicitBackpathScorer(model, tokenizer=None)

    # 0 -> 1 -> 2 is exactly the cycle's path; 1 -> 1 -> 2 is not.
    suffix = ops.tensor([1, 2], dtype=ops.long_dtype)
    candidates = ops.tensor([0, 1], dtype=ops.long_dtype)

    scores = scorer.score_candidates(suffix, candidates)
    got = scores.tolist()

    # row = [candidate] + suffix; position p's logits come from table[row[p]].
    # candidate=0 -> row [0,1,2]: table[0] predicts suffix[0]=1, table[1] predicts suffix[1]=2.
    expected_0 = _log_softmax_at(CYCLE_TABLE[0], 1) + _log_softmax_at(CYCLE_TABLE[1], 2)
    # candidate=1 -> row [1,1,2]: table[1] predicts suffix[0]=1, table[1] again predicts suffix[1]=2.
    expected_1 = _log_softmax_at(CYCLE_TABLE[1], 1) + _log_softmax_at(CYCLE_TABLE[1], 2)

    assert math.isclose(got[0], expected_0, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(got[1], expected_1, rel_tol=1e-4, abs_tol=1e-4)
    assert got[0] > got[1]


def test_score_candidates_chunking_matches_unchunked():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel(CYCLE_TABLE)
    scorer = ImplicitBackpathScorer(model, tokenizer=None)

    suffix = ops.tensor([1, 2], dtype=ops.long_dtype)
    candidates = ops.tensor([0, 1, 2, 0, 1, 2, 0], dtype=ops.long_dtype)

    unchunked = scorer.score_candidates(suffix, candidates, max_batch_size=None)
    chunked = scorer.score_candidates(suffix, candidates, max_batch_size=3)

    assert unchunked.shape[0] == chunked.shape[0] == 7
    for a, b in zip(unchunked.tolist(), chunked.tolist()):
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


def test_score_candidates_handles_empty_pool():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel(CYCLE_TABLE)
    scorer = ImplicitBackpathScorer(model, tokenizer=None)

    suffix = ops.tensor([1], dtype=ops.long_dtype)
    candidates = ops.tensor([], dtype=ops.long_dtype)

    scores = scorer.score_candidates(suffix, candidates)
    assert scores.shape[0] == 0


class FakeTokenizer:
    VOCAB = ["a", "b", "\x00junk", "c"]

    def decode(self, ids):
        return self.VOCAB[ids[0]]


def test_candidate_pool_applies_writing_filter():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel([[0.0] * 4] * 4)
    tok = FakeTokenizer()
    wf = WritingTokenFilter(tok)
    scorer = ImplicitBackpathScorer(model, tokenizer=tok, writing_filter=wf)

    pool = scorer.candidate_pool(ops, vocab_size=len(tok.VOCAB), device="cpu")

    assert pool.tolist() == [0, 1, 3]  # index 2 ("\x00junk") filtered out


def test_candidate_pool_without_filter_returns_full_vocab():
    ops = AbstractTensor.get_tensor()
    model = BigramDummyModel([[0.0] * 4] * 4)
    scorer = ImplicitBackpathScorer(model, tokenizer=None)

    pool = scorer.candidate_pool(ops, vocab_size=4, device="cpu")

    assert pool.tolist() == [0, 1, 2, 3]
