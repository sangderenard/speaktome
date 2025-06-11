import pytest

pytest.importorskip("torch", reason="CompressedBeamTree tests require torch")

import torch
from speaktome.core.compressed_beam_tree import CompressedBeamTree


def test_extend_leaves_batch_multiple():
    tree = CompressedBeamTree(device="cpu")
    idx0 = tree.add_root_beam([1, 2], [0.1, 0.2])
    idx1 = tree.add_root_beam([5], [0.5])

    parents = torch.tensor([idx0, idx1], dtype=torch.long)
    tokens = torch.tensor([[1, 2, 3], [5, 6, 7]], dtype=torch.long)
    scores = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]], dtype=torch.float)
    lengths = torch.tensor([3, 3], dtype=torch.long)

    new_ids = tree.extend_leaves_batch(parents, tokens, scores, lengths)
    assert len(new_ids) == 2

    t0, _, l0 = tree.get_beam_tensors_by_beam_idx(int(new_ids[0]), 3)
    assert t0.tolist() == [1, 2, 3] and l0 == 3

    t1, _, l1 = tree.get_beam_tensors_by_beam_idx(int(new_ids[1]), 3)
    assert t1.tolist() == [5, 6, 7] and l1 == 3
