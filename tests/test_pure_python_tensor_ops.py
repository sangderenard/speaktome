"""Tests for the pure-Python tensor operations."""

import pytest
import logging
import importlib.util
from speaktome.tensors import (
    PurePythonTensorOperations,
    PyTorchTensorOperations,
)
# --- END HEADER ---

logger = logging.getLogger(__name__)

BACKENDS = [PurePythonTensorOperations]
if importlib.util.find_spec("torch") is not None:
    BACKENDS.append(PyTorchTensorOperations)


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_stack_dim0_and_dim1(backend_cls) -> None:
    """Stack tensors along the first two dimensions."""
    logger.info("test_stack_dim0_and_dim1 start")
    ops = backend_cls(track_time=True)
    t = [[1, 2], [3, 4]]
    stacked0 = ops.benchmark(lambda: ops.stack([t, t], dim=0))
    assert ops.last_op_time is not None
    assert stacked0 == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    stacked1 = ops.benchmark(lambda: ops.stack([t, t], dim=1))
    assert stacked1 == [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]
    logger.info('test_stack_dim0_and_dim1 end')


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_pad_2d(backend_cls) -> None:
    """Pad a 2-D tensor and verify the result."""
    logger.info("test_pad_2d start")
    ops = backend_cls(track_time=True)
    t = [[1, 2], [3, 4]]
    padded = ops.benchmark(lambda: ops.pad(t, (1, 1, 1, 1), value=0))
    assert ops.last_op_time is not None
    assert padded == [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]
    logger.info('test_pad_2d end')


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_topk_and_dtypes(backend_cls) -> None:
    """Check ``topk`` output and exposed dtypes."""
    logger.info("test_topk_and_dtypes start")
    ops = backend_cls(track_time=True)
    values, indices = ops.benchmark(lambda: ops.topk([1, 3, 2, 4], k=2, dim=-1))
    assert ops.last_op_time is not None
    assert values == [4, 3]
    assert indices == [3, 1]
    assert ops.long_dtype is int
    assert ops.bool_dtype is bool
    logger.info('test_topk_and_dtypes end')
