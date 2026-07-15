#!/usr/bin/env python3
"""Word-boundary detection for GPT-2-style BPE token text.

GPT-2's tokenizer marks the start of a new word by decoding it with a
literal leading space (the underlying byte-level "Ġ" marker); a token with
no leading space is a continuation piece glued onto whatever came before it
with no space in between. That's the actual signal the model was trained
on for "did the previous word just end" -- more faithful than trying to
re-derive word boundaries purely from a dictionary, which is why this is a
separate, dictionary-free check from WordTrie.
"""
from __future__ import annotations
# --- END HEADER ---


def starts_new_word(token_text: str) -> bool:
    """True if decoded ``token_text`` begins a new word (or isn't a word at all).

    Whitespace and punctuation-led tokens count as starting fresh -- a
    continuation piece is specifically an alphanumeric (or apostrophe/
    hyphen, for contractions and compounds) fragment with no leading space.
    """
    if not token_text:
        return True
    first = token_text[0]
    if first.isspace():
        return True
    return not (first.isalnum() or first in "'-")
