"""Tests for speaktome.core.word_trie."""

from speaktome.core.word_trie import WordTrie


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
