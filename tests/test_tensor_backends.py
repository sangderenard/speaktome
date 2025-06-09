"""Comprehensive tensor backend checks."""

import importlib.util
import logging
import pytest

from speaktome.tensors import (
    PurePythonTensorOperations,
    NumPyTensorOperations,
    PyTorchTensorOperations,
    JAXTensorOperations,
)
from speaktome.tensors.faculty import detect_faculty

logger = logging.getLogger(__name__)


def available_backends():
    """Return list of backend classes based on installed packages."""
    backends = [PurePythonTensorOperations]
    if importlib.util.find_spec("numpy") is not None:
        backends.append(NumPyTensorOperations)
    if importlib.util.find_spec("torch") is not None:
        backends.append(PyTorchTensorOperations)
    if importlib.util.find_spec("jax") is not None:
        backends.append(JAXTensorOperations)
    return backends


def run_checks(ops):
    t = [[1, 2], [3, 4]]
    stacked0 = ops.stack([t, t], dim=0)
    assert stacked0 == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    stacked1 = ops.stack([t, t], dim=1)
    assert stacked1 == [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]

    rng = list(range(3))
    data = [[i + j for j in rng] for i in rng]
    stacked0 = ops.stack([data, data], dim=0)
    assert stacked0 == [data, data]
    stacked1 = ops.stack([data, data], dim=1)
    assert stacked1 == [[row, row] for row in data]

    padded = ops.pad(t, (1, 1, 1, 1), value=0)
    assert padded == [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]

    base = [[i for i in range(3)] for _ in range(2)]
    padded = ops.pad(base, (0, 2, 1, 0), value=9)
    assert len(padded) == 3
    assert all(len(row) == 5 for row in padded)

    values, indices = ops.topk([1, 3, 2, 4], k=2, dim=-1)
    assert values == [4, 3]
    assert indices == [3, 1]

    seq = [7, 1, 4, 9, 2, 6]
    vals, idxs = ops.topk(seq, k=3, dim=-1)
    expect_vals = sorted(seq, reverse=True)[:3]
    expect_inds = [seq.index(v) for v in expect_vals]
    assert vals == expect_vals
    assert idxs == expect_inds

    assert ops.long_dtype is not None
    assert ops.bool_dtype is not None


@pytest.mark.parametrize("backend_cls", available_backends())
def test_tensor_ops_across_backends(backend_cls, tensor_interactive) -> None:
    """Exercise multiple tensor operations for each backend."""
    logger.info("Detected faculty: %s", detect_faculty().name)
    ops = backend_cls(track_time=tensor_interactive)
    run_checks(ops)
    if tensor_interactive:
        assert ops.last_op_time is not None
