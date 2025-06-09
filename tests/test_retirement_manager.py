"""BeamRetirementManager pruning tests."""

import logging
import types
import pytest

pytest.importorskip('torch', reason='Retirement manager requires torch for tensors')
import torch
from speaktome.core.beam_retirement_manager import BeamRetirementManager

# --- END HEADER ---

logger = logging.getLogger(__name__)


def make_manager():
    tree = types.SimpleNamespace(device=torch.device("cpu"), operator=types.SimpleNamespace())
    return BeamRetirementManager(tree)


def test_remove_used_prunes_buckets():
    logger.info("test_remove_used_prunes_buckets start")
    mgr = make_manager()
    mgr._bucket = {1: [1, 2, 3], 2: [2, 4]}
    mgr.remove_used([2, 3])
    assert mgr._bucket == {1: [1], 2: [4]}
    logger.info("test_remove_used_prunes_buckets end")


def test_garbage_collect_limits_buckets():
    mgr = make_manager()
    # create several buckets with many entries
    for i in range(30):
        mgr._bucket.setdefault(i % 3, []).append(i)

    mgr.garbage_collect(limit_per_bucket=5, total_limit=8)
    assert all(len(g) <= 5 for g in mgr._bucket.values())
    total = sum(len(g) for g in mgr._bucket.values())
    assert total <= 8
