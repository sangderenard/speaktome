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

from typing import Any, Dict, Iterable, List, Tuple
# --- END HEADER ---


class WordTrie:
    """Case-insensitive prefix trie: ``is_word`` and ``is_prefix`` over a word list.

    ``reverse=True`` indexes each word backwards (``word[::-1]``) instead of
    as-is -- built for backward word growth, which discovers a word from its
    end toward its start (each new token gets *prepended*, so the trie needs
    to answer "is this a valid suffix-so-far of some real word", not prefix).
    Querying a reversed trie with a reversed string asks exactly that
    question, reusing the same walk logic either way.
    """

    _END = "$"

    def __init__(self, words: Iterable[str], reverse: bool = False):
        self._root: Dict[str, Any] = {}
        self.reverse = reverse
        for word in words:
            word = word.strip().lower()
            if not word:
                continue
            if reverse:
                word = word[::-1]
            node = self._root
            for ch in word:
                node = node.setdefault(ch, {})
            node[self._END] = True

    def root_node(self) -> Dict[str, Any]:
        """The trie's entry point -- the walk state before any characters are known."""
        return self._root

    def is_end(self, node: Any) -> bool:
        """True if ``node`` (a state returned by walk_from/root_node) is a complete word."""
        return node is not None and self._END in node

    def walk_from(self, node: Any, s: str):
        """Walk ``s`` from an existing state, not necessarily the root.

        Returns the resulting state, or ``None`` if ``s`` isn't a valid
        continuation from ``node`` at some character. ``node=None`` in
        (already failed elsewhere) short-circuits to ``None`` rather than
        raising, so callers can chain walks without checking after every step.
        """
        for ch in s.lower():
            if node is None:
                return None
            node = node.get(ch)
        return node

    def _walk(self, s: str):
        return self.walk_from(self._root, s)

    def is_word(self, s: str) -> bool:
        node = self._walk(s)
        return self.is_end(node)

    def is_prefix(self, s: str) -> bool:
        """True if ``s`` is a (possibly-empty) prefix of some word in the trie.

        The empty string is always a valid prefix -- a word-in-progress with
        no characters accumulated yet hasn't failed to match anything.
        """
        if s == "":
            return True
        return self._walk(s) is not None

    @classmethod
    def from_file(cls, path: str, reverse: bool = False) -> "WordTrie":
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        return cls(words, reverse=reverse)

    @classmethod
    def from_curated_wordlist(cls, n: int = 20000, reverse: bool = False) -> "WordTrie":
        """Build from a real dictionary narrowed to its n most common words.

        See word_sources.curated_english_wordlist -- cross-references
        nltk's actual dictionary against wordfreq's popularity ranking, so
        the result is real English words only, ordered/limited by how
        common they are. Requires the optional ``nltk`` and ``wordfreq``
        packages, imported lazily so the rest of this module has no hard
        dependency on either.
        """
        from .word_sources import curated_english_wordlist
        return cls(curated_english_wordlist(n), reverse=reverse)


class TrieGate:
    """Caches, per trie node, which candidates from a fixed vocabulary pool validly continue from there.

    A trie's own fan-out (children per node) is small and bounded by the
    alphabet, but the question word growth actually needs answered is
    "which of my ~20k-word-dictionary-filtered *vocabulary token ids*
    decode to text that keeps me inside the trie" -- one BPE token can
    span several characters, so this is a real per-candidate walk, not a
    single-character lookup. Doing that walk over the whole pool is a
    fixed, tokenizer-only (no GPU, no model call) cost; caching it by trie
    node means it only happens once per unique node ever reached across a
    whole session, not once per (node, tick) revisit -- the same node
    (e.g. the root, or a common short prefix like "th") gets revisited
    constantly as different graph nodes grow different words.

    When ``trie.reverse`` is set, candidate text is reversed before
    walking, not just the trie's own indexing: prepending in normal-text
    space (``cand_text + span_so_far``) is *appending* in the reversed
    representation the trie's state already tracks (``reverse(span) +
    reverse(cand_text)``) -- walking the candidate's un-reversed text from
    a reversed-trie state would ask the wrong question at every step past
    the first character.
    """

    def __init__(self, trie: WordTrie, tokenizer: Any, candidate_ids: Iterable[int]):
        self._trie = trie
        self._ids: List[int] = list(candidate_ids)
        # Decode once, up front -- every subsequent node query reuses this,
        # rather than re-decoding the same ~24k candidates per call.
        self._texts: List[str] = []
        for tid in self._ids:
            try:
                text = tokenizer.decode([int(tid)]).strip()
            except Exception:
                text = ""
            if trie.reverse:
                text = text[::-1]
            self._texts.append(text)
        self._cache: Dict[int, List[Tuple[int, Any]]] = {}

    def continuations(self, node: Any) -> List[Tuple[int, Any]]:
        """Return ``[(token_id, next_node), ...]`` for pool candidates valid from ``node``.

        Every returned candidate is guaranteed to keep the walk inside the
        trie -- a token whose decoded text doesn't survive the walk (or
        decodes to nothing after stripping) is simply absent, not flagged.
        Cached by node identity: trie nodes are plain dicts created once at
        build time and never mutated afterward, so the same logical state
        always maps to the same object and this cache never goes stale.
        """
        key = id(node)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result: List[Tuple[int, Any]] = []
        for tid, text in zip(self._ids, self._texts):
            if not text:
                continue
            next_node = self._trie.walk_from(node, text)
            if next_node is not None:
                result.append((tid, next_node))
        self._cache[key] = result
        return result
