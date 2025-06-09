import logging
import pytest

pytest.importorskip("torch")

from speaktome.core.compressed_beam_tree import CompressedBeamTree

# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_snap_beam_path_root_prefix_reuse():
    """Paths sharing a root prefix should reuse existing nodes."""
    tree = CompressedBeamTree(device="cpu")
    tree.snap_beam_path([1, 2, 3], [0.1, 0.2, 0.3])
    tree.snap_beam_path([5, 6], [0.5, 0.6])
    tree.snap_beam_path([1, 2, 4], [0.1, 0.2, 0.4])

    root_one = [idx for idx, n in enumerate(tree.nodes) if n.parent_node_idx is None and n.token_tensor.item() == 1]
    assert len(root_one) == 1
    child_twos = [idx for idx in tree.nodes[root_one[0]].children_node_indices if tree.nodes[idx].token_tensor.item() == 2]
    assert len(child_twos) == 1


def test_snap_beam_path_insert_under_leaf():
    """New paths can attach under an existing leaf."""
    tree = CompressedBeamTree(device="cpu")
    idx = tree.snap_beam_path([1, 2], [0.1, 0.2])
    tree.snap_beam_path([3, 4], [0.3, 0.4], insert_under_beam_idx=idx)
    root_one = [i for i, n in enumerate(tree.nodes) if n.parent_node_idx is None and n.token_tensor.item() == 1]
    assert len(root_one) == 1
    leaf_len = tree.nodes[tree.leaf_node_indices[idx]].depth
    new_leaf = tree.leaf_node_indices[idx + 1]
    assert tree.nodes[new_leaf].depth == leaf_len + 2
