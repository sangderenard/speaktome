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
