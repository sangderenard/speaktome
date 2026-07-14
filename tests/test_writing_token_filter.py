"""Tests for speaktome.core.writing_token_filter."""

from tensors import AbstractTensor
from speaktome.core.writing_token_filter import WritingTokenFilter


class FakeTokenizer:
    """Decodes a small fixed vocabulary of ordinary and junk tokens."""

    VOCAB = [
        "hello",
        " world",
        "don't",
        "3.14",
        "\x00\x01",       # control characters
        "�",         # unicode replacement character
        "café",           # non-ascii
        "",                # empty decode (e.g. some special tokens)
        "!?.,",
    ]

    def __init__(self):
        self.decode_calls = 0

    def decode(self, ids):
        self.decode_calls += 1
        return self.VOCAB[ids[0]]


EXPECTED = [True, True, True, True, False, False, False, False, True]


def test_is_ordinary_writing_classifies_individual_strings():
    f = WritingTokenFilter(FakeTokenizer())
    assert f.is_ordinary_writing("hello world") is True
    assert f.is_ordinary_writing("don't stop, 3.14!") is True
    assert f.is_ordinary_writing("") is False
    assert f.is_ordinary_writing("\x00control") is False
    assert f.is_ordinary_writing("�") is False
    assert f.is_ordinary_writing("café") is False


def test_mask_as_list_matches_manual_classification():
    tok = FakeTokenizer()
    f = WritingTokenFilter(tok)
    result = f.mask_as_list(len(tok.VOCAB))
    assert result == EXPECTED


def test_mask_as_list_is_cached():
    tok = FakeTokenizer()
    f = WritingTokenFilter(tok)
    f.mask_as_list(len(tok.VOCAB))
    calls_after_first = tok.decode_calls
    f.mask_as_list(len(tok.VOCAB))
    assert tok.decode_calls == calls_after_first  # no re-decoding


def test_extra_allowed_chars_widens_the_filter():
    tok = FakeTokenizer()
    f = WritingTokenFilter(tok, extra_allowed_chars="é")
    assert f.is_ordinary_writing("café") is True


def test_build_mask_returns_tensor_matching_backend():
    tok = FakeTokenizer()
    f = WritingTokenFilter(tok)
    ops = AbstractTensor.get_tensor()

    mask = f.build_mask(ops, vocab_size=len(tok.VOCAB), device="cpu")

    assert type(mask) is type(ops)
    assert mask.tolist() == EXPECTED
