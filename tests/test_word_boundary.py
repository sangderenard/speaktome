"""Tests for speaktome.core.word_boundary."""

from speaktome.core.word_boundary import starts_new_word


def test_leading_space_starts_new_word():
    assert starts_new_word(" running")


def test_no_leading_space_alnum_is_a_continuation():
    assert not starts_new_word("ning")


def test_apostrophe_and_hyphen_continuations_are_not_new_words():
    assert not starts_new_word("'t")  # e.g. completing "don" + "'t"
    assert not starts_new_word("-and")


def test_punctuation_counts_as_a_boundary():
    assert starts_new_word(".")
    assert starts_new_word(",")
    assert starts_new_word("\n")


def test_empty_text_counts_as_a_boundary():
    assert starts_new_word("")
