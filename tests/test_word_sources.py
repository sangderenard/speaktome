"""Tests for speaktome.core.word_sources."""

from unittest.mock import patch

from speaktome.core.word_sources import curated_english_wordlist


def test_curated_wordlist_keeps_only_real_dictionary_words():
    # "rok" and "sainsbury's" are wordfreq-ranked strings that are not real
    # dictionary words -- they must be dropped even though they outrank
    # some real words in the mocked frequency order.
    with patch("speaktome.core.word_sources._load_nltk_dictionary_words", return_value={"cat", "dog", "run"}):
        with patch("wordfreq.top_n_list", return_value=["rok", "cat", "sainsbury's", "dog", "xyz", "run"]) as m:
            result = curated_english_wordlist(n=20, pool_size=6)
            m.assert_called_once_with("en", 6)
    assert result == ["cat", "dog", "run"]


def test_curated_wordlist_respects_n():
    with patch("speaktome.core.word_sources._load_nltk_dictionary_words", return_value={"cat", "dog", "run", "sun"}):
        with patch("wordfreq.top_n_list", return_value=["cat", "dog", "run", "sun"]):
            result = curated_english_wordlist(n=2, pool_size=10)
    assert result == ["cat", "dog"]


def test_curated_wordlist_preserves_popularity_order():
    with patch("speaktome.core.word_sources._load_nltk_dictionary_words", return_value={"cat", "dog", "run"}):
        with patch("wordfreq.top_n_list", return_value=["run", "cat", "dog"]):
            result = curated_english_wordlist(n=10, pool_size=10)
    assert result == ["run", "cat", "dog"]
