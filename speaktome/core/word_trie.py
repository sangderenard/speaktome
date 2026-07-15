#!/usr/bin/env python3
"""A minimal prefix trie over a word list.

A flat set can only answer "is this a complete word". Growing a word one
BPE subtoken at a time needs a second question along the way: "is what
I've accumulated so far still capable of becoming a real word" -- that's
what a trie gives you for the same storage, and it's what lets a
branching subword search prune dead paths before they ever reach the
model again.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable
# --- END HEADER ---


class WordTrie:
    """Case-insensitive prefix trie: ``is_word`` and ``is_prefix`` over a word list."""

    _END = "$"

    def __init__(self, words: Iterable[str]):
        self._root: Dict[str, Any] = {}
        for word in words:
            word = word.strip().lower()
            if not word:
                continue
            node = self._root
            for ch in word:
                node = node.setdefault(ch, {})
            node[self._END] = True

    def _walk(self, s: str):
        node = self._root
        for ch in s.lower():
            node = node.get(ch)
            if node is None:
                return None
        return node

    def is_word(self, s: str) -> bool:
        node = self._walk(s)
        return node is not None and self._END in node

    def is_prefix(self, s: str) -> bool:
        """True if ``s`` is a (possibly-empty) prefix of some word in the trie.

        The empty string is always a valid prefix -- a word-in-progress with
        no characters accumulated yet hasn't failed to match anything.
        """
        if s == "":
            return True
        return self._walk(s) is not None

    @classmethod
    def from_file(cls, path: str) -> "WordTrie":
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        return cls(words)

    @classmethod
    def from_curated_wordlist(cls, n: int = 20000) -> "WordTrie":
        """Build from a real dictionary narrowed to its n most common words.

        See word_sources.curated_english_wordlist -- cross-references
        nltk's actual dictionary against wordfreq's popularity ranking, so
        the result is real English words only, ordered/limited by how
        common they are. Requires the optional ``nltk`` and ``wordfreq``
        packages, imported lazily so the rest of this module has no hard
        dependency on either.
        """
        from .word_sources import curated_english_wordlist
        return cls(curated_english_wordlist(n))
