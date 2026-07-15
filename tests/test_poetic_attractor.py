"""Tests for speaktome.core.poetic_attractor."""

import math

from speaktome.core.poetic_attractor import PoeticAttractor


def test_full_rhyme_scores_higher_than_no_rhyme():
    attractor = PoeticAttractor()
    rhyming = attractor.score_word("cat", ["hat"])
    non_rhyming = attractor.score_word("dog", ["hat"])
    assert rhyming > non_rhyming


def test_exact_repeat_scores_at_least_as_high_as_partial_rhyme():
    # A word rhymes perfectly with itself -- not a special case, just the
    # natural result of full suffix overlap.
    attractor = PoeticAttractor()
    assert attractor.score_word("hat", ["hat"]) >= attractor.score_word("cat", ["hat"])


def test_alliteration_contributes_even_without_rhyme():
    attractor = PoeticAttractor(rhyme_weight=0.0, slant_rhyme_weight=0.0, internal_rhyme_weight=0.0)
    alliterative = attractor.score_word("sun", ["silver"])
    plain = attractor.score_word("moon", ["silver"])
    assert alliterative > plain
    assert math.isclose(plain, 0.0)


def test_internal_rhyme_reaches_words_before_the_line_end():
    attractor = PoeticAttractor(rhyme_weight=0.0, slant_rhyme_weight=0.0, alliteration_weight=0.0)
    # "light" doesn't rhyme with the line-end word "day" but does rhyme
    # with "night" earlier in the line.
    bonus = attractor.score_word("light", ["the", "night", "was", "long", "today"])
    assert bonus > 0.0


def test_zero_weights_disable_that_component():
    attractor = PoeticAttractor(
        rhyme_weight=0.0, slant_rhyme_weight=0.0, alliteration_weight=0.0, internal_rhyme_weight=0.0
    )
    assert attractor.score_word("cat", ["hat", "sat", "mat"]) == 0.0


def test_empty_context_scores_zero():
    attractor = PoeticAttractor()
    assert attractor.score_word("cat", []) == 0.0


def test_empty_candidate_word_scores_zero():
    attractor = PoeticAttractor()
    assert attractor.score_word("", ["hat"]) == 0.0


def test_short_words_below_min_word_len_do_not_rhyme():
    attractor = PoeticAttractor(min_word_len=3)
    # "a" and "at" are too short to count as a rhyme match.
    assert attractor.score_word("a", ["hat"]) == 0.0
