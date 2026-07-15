#!/usr/bin/env python3
"""A rudimentary poetic-quality bonus over candidate words.

Not a phonetic model -- there's no CMUdict/phoneme lookup here, just cheap
orthographic heuristics over decoded token text: trailing-letter overlap for
rhyme, a looser trailing-letter overlap for slant rhyme, leading-letter match
for alliteration, and the same rhyme check swept across the whole context
window (not just its last word) for internal rhyme. Crude on purpose --
"rudimentary actions of checking for rhymes", not a linguistics engine.

This class only ever turns (candidate word, context words) into a small
float bonus. It knows nothing about tokens, models, or scores -- callers
own the honesty question of how that bonus is allowed to touch a real score
(see FluxGraph._apply_poetic_rerank, which uses this to *rank* candidates
without ever overwriting a node's true model-evidence score).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

VOWELS = set("aeiouy")
# --- END HEADER ---


def _clean(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha())


def _consonant_skeleton(word: str) -> str:
    return "".join(ch for ch in word if ch not in VOWELS)


def _suffix_overlap(a: str, b: str, max_len: int = 4) -> float:
    """Fraction of matching trailing characters, up to ``max_len``, in [0, 1]."""
    if not a or not b:
        return 0.0
    n = 0
    limit = min(len(a), len(b), max_len)
    while n < limit and a[-1 - n] == b[-1 - n]:
        n += 1
    return n / max_len


@dataclass
class PoeticAttractor:
    """Coefficients for a rudimentary poetic-quality bonus, plus the scorer itself.

    Each weight is independently settable; 0.0 turns that component off
    without disabling the others. ``score_word`` returns a single combined
    bonus meant to be added (at some external scale) to a real evidence
    score for *ranking* purposes only.
    """

    rhyme_weight: float = 0.6
    slant_rhyme_weight: float = 0.3
    alliteration_weight: float = 0.2
    internal_rhyme_weight: float = 0.4
    min_word_len: int = 2

    def _rhyme(self, a: str, b: str) -> float:
        if len(a) < self.min_word_len or len(b) < self.min_word_len:
            return 0.0
        return _suffix_overlap(a, b)

    def _slant_rhyme(self, a: str, b: str) -> float:
        if len(a) < self.min_word_len or len(b) < self.min_word_len:
            return 0.0
        # Same trailing-consonant skeleton but not the same trailing letters
        # outright -- "shape"/"grape" rhyme outright; "shape"/"laugh" don't
        # rhyme but share a trailing "consonant feel". Weaker signal than a
        # full rhyme, so it's scored on the consonant skeleton only.
        return _suffix_overlap(_consonant_skeleton(a), _consonant_skeleton(b), max_len=3)

    def _alliteration(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return 1.0 if a[0] == b[0] else 0.0

    def score_word(self, candidate_word: str, context_words: List[str]) -> float:
        """Combined poetic bonus for ``candidate_word`` given recent ``context_words``.

        ``context_words`` is read in the order the underlying text actually
        reads (oldest first); the last entry is treated as the current
        "line end" for rhyme/alliteration, and the rest are swept for
        internal rhyme.
        """
        word = _clean(candidate_word)
        if not word:
            return 0.0
        cleaned = [_clean(w) for w in context_words]
        cleaned = [w for w in cleaned if w]
        if not cleaned:
            return 0.0

        line_end = cleaned[-1]
        nearest = cleaned[-1]

        bonus = 0.0
        bonus += self.rhyme_weight * self._rhyme(word, line_end)
        bonus += self.slant_rhyme_weight * self._slant_rhyme(word, line_end)
        bonus += self.alliteration_weight * self._alliteration(word, nearest)

        if len(cleaned) > 1 and self.internal_rhyme_weight:
            best_internal = max(self._rhyme(word, w) for w in cleaned[:-1])
            bonus += self.internal_rhyme_weight * best_internal

        return bonus
