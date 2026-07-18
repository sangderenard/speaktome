#!/usr/bin/env python3
"""A plain dict/set-backed word filter, plus a combinator for stacking filters.

WritingTokenFilter only screens out junk (control characters, byte-fallback
fragments) -- it still leaves the full space of "looks like writing" BPE
tokens in play, which in practice is nearly the whole vocabulary. The actual
backward-expansion bottleneck is the size of that candidate pool (every
backward expansion scores it in full), so the lever that matters is cutting
the pool down to recognizable words, not just non-junk fragments.

This is deliberately dumb: a token passes if its decoded text, once
GPT-2-style leading-space/casing is normalized away, is literally a member
of a caller-supplied set of words. No stemming, no subword awareness. BPE
routinely splits a real word into fragments ("running" -> "runn" + "ing"),
and this filter has no way to know that -- a fragment that isn't itself a
whole dictionary entry gets dropped even when it's a perfectly ordinary
continuation. That's a real tradeoff, not an oversight: this filter is for
callers who want the pool biased toward whole recognizable words and are
willing to lose mid-word BPE pieces to get there. It composes with
WritingTokenFilter (or anything sharing the same interface) via
CombinedTokenFilter rather than replacing it outright.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence
# --- END HEADER ---


class DictionaryTokenFilter:
    """Classifies which vocabulary ids decode to a member of a given word set."""

    def __init__(self, tokenizer: Any, dictionary: Iterable[str]):
        self.tokenizer = tokenizer
        self.dictionary = {w.strip().lower() for w in dictionary if w.strip()}
        self._mask_cache: Optional[List[bool]] = None

    def is_dictionary_word(self, text: str) -> bool:
        return text.strip().lower() in self.dictionary

    def _build_bool_list(self, vocab_size: int) -> List[bool]:
        flags = []
        for token_id in range(vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
            except Exception:
                flags.append(False)
                continue
            flags.append(self.is_dictionary_word(text))
        return flags

    def mask_as_list(self, vocab_size: int) -> List[bool]:
        if self._mask_cache is None or len(self._mask_cache) != vocab_size:
            self._mask_cache = self._build_bool_list(vocab_size)
        return self._mask_cache

    def build_mask(self, tensor_ops: Any, vocab_size: int, device: Any = None):
        flags = self.mask_as_list(vocab_size)
        backend_cls = type(tensor_ops)
        return backend_cls.tensor(flags, dtype=tensor_ops.bool_dtype, device=device)

    @classmethod
    def from_file(cls, tokenizer: Any, path: str) -> "DictionaryTokenFilter":
        """Build from a newline-separated word list file (e.g. a system dictionary)."""
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        return cls(tokenizer, words)

    @classmethod
    def from_curated_wordlist(
        cls,
        tokenizer: Any,
        n: int = 20000,
        min_word_len: int = 2,
        max_word_len: Optional[int] = None,
    ) -> "DictionaryTokenFilter":
        """Build from a real dictionary narrowed to its n most common words.

        See word_sources.curated_english_wordlist -- cross-references
        nltk's actual dictionary against wordfreq's popularity ranking, so
        the result is real English words only, ordered/limited by how
        common they are. Requires the optional ``nltk`` and ``wordfreq``
        packages.
        """
        from .word_sources import curated_english_wordlist
        return cls(tokenizer, curated_english_wordlist(n, min_word_len=min_word_len, max_word_len=max_word_len))


class CombinedTokenFilter:
    """ANDs together any number of filters sharing this module's mask interface.

    Every member filter must expose ``mask_as_list(vocab_size)``, the same
    contract WritingTokenFilter and DictionaryTokenFilter both already
    implement -- a token id passes only if every filter agrees it should.
    """

    def __init__(self, filters: Sequence[Any]):
        self.filters = list(filters)
        self._mask_cache: Optional[List[bool]] = None

    def mask_as_list(self, vocab_size: int) -> List[bool]:
        if self._mask_cache is None or len(self._mask_cache) != vocab_size:
            if not self.filters:
                self._mask_cache = [True] * vocab_size
            else:
                per_filter = [f.mask_as_list(vocab_size) for f in self.filters]
                self._mask_cache = [all(flags) for flags in zip(*per_filter)]
        return self._mask_cache

    def build_mask(self, tensor_ops: Any, vocab_size: int, device: Any = None):
        flags = self.mask_as_list(vocab_size)
        backend_cls = type(tensor_ops)
        return backend_cls.tensor(flags, dtype=tensor_ops.bool_dtype, device=device)
