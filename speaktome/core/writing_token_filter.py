#!/usr/bin/env python3
"""Filter a tokenizer's vocabulary down to ordinary prose tokens.

BPE vocabularies (GPT-2 and similar) contain a lot of tokens that never
show up in ordinary writing: raw byte-fallback fragments, control
characters, isolated combining marks, and so on. Sweeping the full
(filtered) vocabulary in a single batched forward pass -- the "implicit
backpath" landscape probe -- is only worth doing if that sweep isn't
polluted by junk tokens the model would never actually choose. This module
builds that filter once and caches the expensive part (decoding and
classifying every token id) as plain Python, so the tensor it hands back
can be constructed fresh against whatever backend the caller is using.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

# --- END HEADER ---

# Printable ASCII (space through tilde) plus ordinary whitespace. Reject
# control characters, the unicode replacement character, and byte-fallback
# fragments that decode to non-printable or non-ASCII symbol soup.
_ORDINARY_WRITING_RE = re.compile(r"^[ -~\s]+$")


class WritingTokenFilter:
    """Classifies which vocabulary ids decode to ordinary prose."""

    def __init__(self, tokenizer: Any, extra_allowed_chars: str = ""):
        self.tokenizer = tokenizer
        self.extra_allowed_chars = extra_allowed_chars
        self._mask_cache: Optional[List[bool]] = None

    def is_ordinary_writing(self, text: str) -> bool:
        """Return True if ``text`` looks like ordinary prose."""
        if not text:
            return False
        if self.extra_allowed_chars:
            text = "".join(c for c in text if c not in self.extra_allowed_chars)
            if not text:
                return True
        return bool(_ORDINARY_WRITING_RE.match(text))

    def _build_bool_list(self, vocab_size: int) -> List[bool]:
        flags = []
        for token_id in range(vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
            except Exception:
                flags.append(False)
                continue
            flags.append(self.is_ordinary_writing(text))
        return flags

    def mask_as_list(self, vocab_size: int) -> List[bool]:
        """Return the (cached) classification as a plain Python list."""
        if self._mask_cache is None or len(self._mask_cache) != vocab_size:
            self._mask_cache = self._build_bool_list(vocab_size)
        return self._mask_cache

    def build_mask(self, tensor_ops: Any, vocab_size: int, device: Any = None):
        """Return the classification as an ``AbstractTensor`` of bools.

        ``tensor_ops`` is any existing tensor of the caller's working
        backend (an instance, not a class) -- the returned mask is built
        to match it via ``type(tensor_ops)``, same as every other tensor
        construction in this codebase.
        """
        flags = self.mask_as_list(vocab_size)
        backend_cls = type(tensor_ops)
        return backend_cls.tensor(flags, dtype=tensor_ops.bool_dtype, device=device)
