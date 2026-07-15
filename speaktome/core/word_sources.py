#!/usr/bin/env python3
"""Build a real, moderately-sized English vocabulary automatically.

Two different data sources, neither sufficient alone:

- nltk's ``words`` corpus is an actual dictionary (no frequency
  information at all) -- but it's enormous (~236k entries) and full of
  obscure/archaic terms nobody uses.
- wordfreq's frequency lists are ranked by real-world usage -- but they
  are not a dictionary at all, just frequent strings from web/subtitle/news
  text, so they carry proper nouns, possessives, and misspellings
  ("sainsbury's", "rouhani", "rok" all show up well inside the top 50k).

Cross-referencing them -- keep only words that are in the real dictionary,
ordered/limited by how common they actually are -- gives a real,
moderately-sized, ordinary-English vocabulary with neither problem. This
list is meant purely as an input to WordTrie/DictionaryTokenFilter's own
hard membership filtering, not as a scoring signal by itself.
"""
from __future__ import annotations

from typing import List
# --- END HEADER ---


def _load_nltk_dictionary_words() -> set:
    import nltk
    try:
        from nltk.corpus import words as nltk_words
        return {w.lower() for w in nltk_words.words()}
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        return {w.lower() for w in nltk_words.words()}


def curated_english_wordlist(n: int = 20000, pool_size: int = 200000, min_word_len: int = 2) -> List[str]:
    """Real dictionary words, narrowed to the ``n`` most common among them.

    ``pool_size`` is how far down wordfreq's ranking to search for real
    dictionary words before stopping -- needs to be well above ``n`` since
    a lot of the frequency ranking is proper nouns/junk that gets filtered
    out along the way.

    ``min_word_len`` matters more than it looks: nltk's ``words`` corpus
    genuinely includes single letters ("a", "b", "c", ...) as headwords,
    and single letters are extremely common in ordinary text (bullets,
    grades, initials) -- common enough that wordfreq's popularity ranking
    doesn't filter them out either, so without this they survive both
    checks and dominate the front of the list despite not being what
    anyone means by "a real word." Default excludes anything shorter than
    2 characters; raise it further (3+) to also drop two-letter entries
    like "ok"/"hi" if those aren't wanted either.

    Requires the optional ``nltk`` (with its ``words`` corpus, downloaded
    on first use) and ``wordfreq`` packages.
    """
    real_words = {w for w in _load_nltk_dictionary_words() if len(w) >= min_word_len}
    from wordfreq import top_n_list
    ranked = top_n_list("en", pool_size)
    curated = [w for w in ranked if len(w) >= min_word_len and w in real_words]
    return curated[:n]
