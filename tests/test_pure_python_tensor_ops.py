"""Tests for the pure-Python tensor operations."""

import pytest
import logging
import importlib.util
from speaktome.tensors import (
    PurePythonTensorOperations,
    PyTorchTensorOperations,
    JAXTensorOperations,
)
# --- END HEADER ---

logger = logging.getLogger(__name__)

BACKENDS = [PurePythonTensorOperations]
if importlib.util.find_spec("torch") is not None:
    BACKENDS.append(PyTorchTensorOperations)
if importlib.util.find_spec("jax") is not None:
    BACKENDS.append(JAXTensorOperations)


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
def test_stack_random_data(backend_cls) -> None:
    """Stack randomly generated tensors and verify dimensions."""
    rng = list(range(3))  # deterministic values avoid flaky tests
    ops = backend_cls(track_time=True)
    t = [[i + j for j in rng] for i in rng]
    stacked0 = ops.stack([t, t], dim=0)
    assert stacked0 == [t, t]
    stacked1 = ops.stack([t, t], dim=1)
    expected_dim1 = [[row, row] for row in t]
    assert stacked1 == expected_dim1


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
def test_pad_random_sizes(backend_cls) -> None:
    """Pad random tensor sizes and confirm output shape."""
    ops = backend_cls(track_time=True)
    base = [[i for i in range(3)] for _ in range(2)]
    padded = ops.pad(base, (0, 2, 1, 0), value=9)
    assert len(padded) == 3
    assert all(len(row) == 5 for row in padded)


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


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_topk_random(backend_cls) -> None:
    """Verify topk on random data matches sorted reference."""
    ops = backend_cls(track_time=True)
    data = [7, 1, 4, 9, 2, 6]
    values, indices = ops.topk(data, k=3, dim=-1)
    expect_vals = sorted(data, reverse=True)[:3]
    expect_inds = [data.index(v) for v in expect_vals]
    assert values == expect_vals
    assert indices == expect_inds
