"""Tests for speaktome.core.word_trie."""

from speaktome.core.word_trie import WordTrie, TrieGate


def test_is_word_true_for_exact_member():
    trie = WordTrie(["cat", "dog", "running"])
    assert trie.is_word("cat")
    assert trie.is_word("CAT")  # case-insensitive


def test_is_word_false_for_non_member():
    trie = WordTrie(["cat", "dog"])
    assert not trie.is_word("ca")
    assert not trie.is_word("caterpillar")


def test_is_prefix_true_for_partial_match():
    trie = WordTrie(["running"])
    assert trie.is_prefix("run")
    assert trie.is_prefix("runn")
    assert trie.is_prefix("running")


def test_is_prefix_false_for_dead_end():
    trie = WordTrie(["cat", "dog"])
    assert not trie.is_prefix("ca-nope")
    assert not trie.is_prefix("xyz")


def test_empty_string_is_always_a_valid_prefix():
    trie = WordTrie(["cat"])
    assert trie.is_prefix("")


def test_a_complete_word_can_also_be_a_prefix_of_a_longer_word():
    trie = WordTrie(["run", "running"])
    assert trie.is_word("run")
    assert trie.is_prefix("run")
    assert trie.is_word("running")


def test_from_file(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("cat\ndog\n", encoding="utf-8")
    trie = WordTrie.from_file(str(p))
    assert trie.is_word("cat")
    assert not trie.is_word("dog2")


def test_walk_from_extends_an_existing_state_incrementally():
    trie = WordTrie(["cat", "catalog", "car", "dog"])
    root = trie.root_node()
    ca = trie.walk_from(root, "ca")
    assert ca is not None
    assert not trie.is_end(ca)
    cat = trie.walk_from(ca, "t")
    assert cat is not None
    assert trie.is_end(cat)  # "cat" is a complete word
    catalog = trie.walk_from(cat, "alog")
    assert catalog is not None
    assert trie.is_end(catalog)  # "catalog" also completes


def test_walk_from_fails_on_a_dead_end_without_raising():
    trie = WordTrie(["cat", "dog"])
    root = trie.root_node()
    assert trie.walk_from(root, "xyz") is None
    # None propagates through further walks rather than raising.
    assert trie.walk_from(None, "anything") is None


def test_reverse_trie_indexes_words_backwards():
    trie = WordTrie(["cat"], reverse=True)
    assert trie.is_word("tac")  # "cat" reversed
    assert not trie.is_word("cat")


def test_reverse_trie_prefix_matches_a_words_suffix():
    # A word-in-progress built backward (e.g. "ly" while growing "quickly")
    # is a *suffix* of the eventual word -- querying the reversed trie with
    # the reversed in-progress span asks that question correctly.
    trie = WordTrie(["quickly"], reverse=True)
    assert trie.is_prefix("ly"[::-1])  # "ly" reversed is a valid prefix of "ylkciuq"
    assert trie.is_word("quickly"[::-1])  # the full word, reversed, is a complete entry
    assert not trie.is_prefix("zz")


class _PoolTokenizer:
    """token id i decodes to POOL[i]; used to test TrieGate in isolation."""

    def __init__(self, pool):
        self.pool = pool

    def decode(self, ids):
        return self.pool[ids[0]]


def test_trie_gate_narrows_to_valid_continuations_only():
    pool = [" cat", " car", "a", "t", "alog", "xyz", " dog"]
    trie = WordTrie(["cat", "catalog", "car", "dog"])
    gate = TrieGate(trie, _PoolTokenizer(pool), range(len(pool)))

    root = trie.root_node()
    root_hits = {tid for tid, _ in gate.continuations(root)}
    # From the root, only whole first-step matches survive: "cat", "car", "dog".
    assert root_hits == {0, 1, 6}

    ca_node = trie.walk_from(root, "ca")
    ca_hits = {tid for tid, _ in gate.continuations(ca_node)}
    # From "ca", only "t" (-> cat, or catalog if grown further) is valid.
    assert ca_hits == {3}


def test_trie_gate_reversed_mode_reverses_candidate_text_before_walking():
    # Growth prepends: new_beam_text = cand_text + beam_text_so_far, so in
    # reversed-representation space that's an *append* of reverse(cand_text)
    # -- TrieGate must reverse candidate text too when the trie is reversed,
    # not just its own indexing, or every step past the first character
    # would ask the wrong question.
    pool = [" run", "ning"]
    fwd_trie = WordTrie(["running"])
    rev_trie = WordTrie(["running"], reverse=True)

    # Seed state: after prepending "ning" (backward growth's first step),
    # the accumulated span is "ning" -- walk that (reversed) from the
    # reversed trie's root to get the state a real _grow_backward_word call
    # would be in before trying to prepend " run".
    state = rev_trie.walk_from(rev_trie.root_node(), "ning"[::-1])
    assert state is not None

    gate = TrieGate(rev_trie, _PoolTokenizer(pool), range(len(pool)))
    hits = {tid for tid, _ in gate.continuations(state)}
    assert hits == {0}  # only " run" (id 0) can validly prepend onto "ning"

    for tid, next_node in gate.continuations(state):
        if tid == 0:
            assert rev_trie.is_end(next_node)  # "run" + "ning" = "running", complete
