"""Tests for the pure-Python tensor operations."""

import pytest
import logging
from speaktome.core.tensor_abstraction import PurePythonTensorOperations
# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_stack_dim0_and_dim1() -> None:
    """Stack tensors along the first two dimensions."""
    logger.info("test_stack_dim0_and_dim1 start")
    ops = PurePythonTensorOperations()
    t = [[1, 2], [3, 4]]
    stacked0 = ops.stack([t, t], dim=0)
    assert stacked0 == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    stacked1 = ops.stack([t, t], dim=1)
    assert stacked1 == [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]
    logger.info('test_stack_dim0_and_dim1 end')


def test_pad_2d() -> None:
    """Pad a 2-D tensor and verify the result."""
    logger.info("test_pad_2d start")
    ops = PurePythonTensorOperations()
    t = [[1, 2], [3, 4]]
    padded = ops.pad(t, (1, 1, 1, 1), value=0)
    assert padded == [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]
    logger.info('test_pad_2d end')


def test_topk_and_dtypes() -> None:
    """Check ``topk`` output and exposed dtypes."""
    logger.info("test_topk_and_dtypes start")
    ops = PurePythonTensorOperations()
    values, indices = ops.topk([1, 3, 2, 4], k=2, dim=-1)
    assert values == [4, 3]
    assert indices == [3, 1]
    assert ops.long_dtype is int
    assert ops.bool_dtype is bool
    logger.info('test_topk_and_dtypes end')
