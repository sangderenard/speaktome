import pytest

torch = pytest.importorskip("torch", reason="requires PyTorch")
from speaktome.core.scorer import Scorer


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


def test_ngram_diversity_matches_slow():
    beams = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
    ])
    lengths = torch.tensor([5, 5])
    tok = type('T', (), {'pad_token_id':0, 'vocab_size':10})()
    fast = Scorer.ngram_diversity_score(beams=beams, lengths=lengths, tokenizer=tok, n=2)
    slow = slow_ngram_diversity(beams, lengths, n=2)
    assert torch.allclose(fast, slow)
