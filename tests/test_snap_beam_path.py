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
