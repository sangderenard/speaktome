"""Tests for speaktome.core.choice_policy."""

import math

import pytest

pytest.importorskip("torch", reason="choice policy tests need a real tensor backend")

from tensors import AbstractTensor
from speaktome.core.choice_policy import AlphaBetaPolicy, TopKPolicy


def _ops():
    return AbstractTensor.get_tensor()


def test_topk_policy_matches_manual_log_softmax_topk():
    ops = _ops()
    logits = ops.tensor_from_list(
        [[0.0, 5.0, 1.0, -3.0]], dtype=ops.float_dtype, device="cpu"
    )
    policy = TopKPolicy()

    scores, indices = policy.choose(logits, k=2)

    assert indices.tolist() == [[1, 2]]

    row = logits.tolist()[0]
    denom = math.log(sum(math.exp(v) for v in row))
    expected = [row[1] - denom, row[2] - denom]
    got = scores.tolist()[0]
    for e, g in zip(expected, got):
        assert math.isclose(e, g, rel_tol=1e-5, abs_tol=1e-6)


def test_topk_policy_respects_batch_dimension():
    ops = _ops()
    logits = ops.tensor_from_list(
        [[0.0, 5.0, 1.0], [9.0, 0.0, 0.0]], dtype=ops.float_dtype, device="cpu"
    )
    policy = TopKPolicy()

    _, indices = policy.choose(logits, k=1)

    assert indices.tolist() == [[1], [0]]


def test_alpha_beta_policy_rejects_bad_parameters():
    with pytest.raises(ValueError):
        AlphaBetaPolicy(alpha=1.5)
    with pytest.raises(ValueError):
        AlphaBetaPolicy(alpha=0.5, beta=0.0)


def test_alpha_beta_policy_alpha_one_is_near_deterministic_for_sharp_distribution():
    ops = _ops()
    # One logit towers over the rest; softmax puts ~all mass on index 1.
    logits = ops.tensor_from_list(
        [[0.0, 20.0, 0.0]], dtype=ops.float_dtype, device="cpu"
    )
    policy = AlphaBetaPolicy(alpha=1.0, beta=1.0, seed=0)

    _, indices = policy.choose(logits, k=1)

    assert indices.tolist() == [[1]]


def test_alpha_beta_policy_alpha_zero_is_roughly_uniform():
    ops = _ops()
    vocab = 4
    trials = 200
    # One dominant logit per row -- if alpha weighting leaked through, index 1
    # would be picked far more than its 1/vocab share.
    row = [0.0, 20.0, 0.0, 0.0]
    logits = ops.tensor_from_list(
        [row for _ in range(trials)], dtype=ops.float_dtype, device="cpu"
    )
    policy = AlphaBetaPolicy(alpha=0.0, beta=1.0, seed=1234)

    _, indices = policy.choose(logits, k=1)

    counts = [0] * vocab
    for row_idx in indices.tolist():
        counts[row_idx[0]] += 1

    expected = trials / vocab
    for c in counts:
        assert abs(c - expected) < expected  # generous band, just rules out collapse


def test_alpha_beta_policy_scores_are_true_model_logprob_not_sampling_weight():
    ops = _ops()
    logits = ops.tensor_from_list(
        [[0.0, 20.0, 0.0]], dtype=ops.float_dtype, device="cpu"
    )
    row = logits.tolist()[0]
    denom = math.log(sum(math.exp(v) for v in row))

    policy = AlphaBetaPolicy(alpha=0.0, beta=1.0, seed=7)
    scores, indices = policy.choose(logits, k=3)

    for idx, score in zip(indices.tolist()[0], scores.tolist()[0]):
        expected = row[idx] - denom
        assert math.isclose(expected, score, rel_tol=1e-5, abs_tol=1e-6)
