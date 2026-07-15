"""Tests for speaktome.core.token_filters."""

from speaktome.core.token_filters import DictionaryTokenFilter, CombinedTokenFilter
from speaktome.core.writing_token_filter import WritingTokenFilter


class FakeTokenizer:
    VOCAB = ["cat", " dog", "Run", "\x00junk", "xyzzy", " "]

    def decode(self, ids):
        return self.VOCAB[ids[0]]


def test_dictionary_filter_keeps_only_known_words():
    tok = FakeTokenizer()
    filt = DictionaryTokenFilter(tok, {"cat", "dog", "run"})
    mask = filt.mask_as_list(len(tok.VOCAB))
    assert mask == [True, True, True, False, False, False]


def test_dictionary_filter_normalizes_leading_space_and_case():
    tok = FakeTokenizer()
    filt = DictionaryTokenFilter(tok, {"Dog "})  # dictionary entries get normalized too
    assert filt.is_dictionary_word(" dog")
    assert filt.is_dictionary_word("DOG")


def test_dictionary_filter_caches_mask():
    tok = FakeTokenizer()
    filt = DictionaryTokenFilter(tok, {"cat"})
    first = filt.mask_as_list(len(tok.VOCAB))
    second = filt.mask_as_list(len(tok.VOCAB))
    assert first is second


def test_combined_filter_ands_masks_together():
    tok = FakeTokenizer()
    dictionary = DictionaryTokenFilter(tok, {"cat", "dog", "run", "xyzzy"})
    writing = WritingTokenFilter(tok)
    combined = CombinedTokenFilter([dictionary, writing])
    mask = combined.mask_as_list(len(tok.VOCAB))
    # "xyzzy" passes the dictionary but WritingTokenFilter should pass it too
    # (it's ordinary ASCII) -- only tokens passing *both* filters survive.
    assert mask == [True, True, True, False, True, False]


def test_combined_filter_with_no_filters_passes_everything():
    combined = CombinedTokenFilter([])
    assert combined.mask_as_list(4) == [True, True, True, True]


def test_dictionary_filter_from_file(tmp_path):
    word_file = tmp_path / "words.txt"
    word_file.write_text("cat\ndog\nrun\n", encoding="utf-8")
    tok = FakeTokenizer()
    filt = DictionaryTokenFilter.from_file(tok, str(word_file))
    mask = filt.mask_as_list(len(tok.VOCAB))
    assert mask == [True, True, True, False, False, False]
