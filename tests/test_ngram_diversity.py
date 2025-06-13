#!/usr/bin/env python3
"""Scorer n-gram diversity tests."""

from __future__ import annotations

try:
    import os
    import pytest
    pytestmark = pytest.mark.requires_torch

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    torch = pytest.importorskip("torch", reason="requires PyTorch")
    from speaktome.core.scorer import Scorer
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def slow_ngram_diversity(beams, lengths, n=2, penalty=-1.0):
    penalties = torch.zeros(beams.shape[0])
    for i in range(beams.shape[0]):
        l = int(lengths[i])
        tokens = beams[i, :l].tolist()
        ngrams = set()
        count = 0
        for j in range(l - n + 1):
            ng = tuple(tokens[j:j+n])
            if ng in ngrams:
                count += 1
            else:
                ngrams.add(ng)
        penalties[i] = penalty * count
    return -penalties


@pytest.mark.parametrize("n", [2, 3])
def test_ngram_diversity_matches_slow(n):
    beams = torch.tensor([
        [1, 2, 3, 4, 1, 2],
        [5, 5, 5, 5, 5, 5],
        [1, 1, 1, 1, 1, 1],
    ])
    lengths = torch.tensor([6, 6, 6])
    tok = type('T', (), {'pad_token_id':0, 'vocab_size':10})()
    fast = Scorer.ngram_diversity_score(beams=beams, lengths=lengths, tokenizer=tok, n=n)
    slow = slow_ngram_diversity(beams, lengths, n=n)
    assert torch.allclose(fast, slow)
