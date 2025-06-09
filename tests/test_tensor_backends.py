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
    stacked0 = ops.benchmark(lambda: ops.stack([t, t], dim=0))
    assert stacked0 == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    stacked1 = ops.benchmark(lambda: ops.stack([t, t], dim=1))
    assert stacked1 == [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]

    rng = list(range(3))
    data = [[i + j for j in rng] for i in rng]
    stacked0 = ops.benchmark(lambda: ops.stack([data, data], dim=0))
    assert stacked0 == [data, data]
    stacked1 = ops.benchmark(lambda: ops.stack([data, data], dim=1))
    assert stacked1 == [[row, row] for row in data]

    padded = ops.benchmark(lambda: ops.pad(t, (1, 1, 1, 1), value=0))
    assert padded == [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]

    base = [[i for i in range(3)] for _ in range(2)]
    padded = ops.benchmark(lambda: ops.pad(base, (0, 2, 1, 0), value=9))
    assert len(padded) == 3
    assert all(len(row) == 5 for row in padded)

    values, indices = ops.benchmark(lambda: ops.topk([1, 3, 2, 4], k=2, dim=-1))
    assert values == [4, 3]
    assert indices == [3, 1]

    seq = [7, 1, 4, 9, 2, 6]
    vals, idxs = ops.benchmark(lambda: ops.topk(seq, k=3, dim=-1))
    expect_vals = sorted(seq, reverse=True)[:3]
    expect_inds = [seq.index(v) for v in expect_vals]
    assert vals == expect_vals
    assert idxs == expect_inds

    assert ops.tolist(ops.sub_scalar([2, 4, 6], 1)) == [1, 3, 5]
    assert ops.tolist(ops.div_scalar([2.0, 4.0], 2.0)) == [1.0, 2.0]

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


def _norm(val):
    """Convert backend tensors to plain Python types for assertions."""
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


class DummyTensor:
    """Lightweight tensor wrapper exercising operator overloads."""

    def __init__(self, ops, data):
        self.ops = ops
        self.data = data

    def _apply(self, op, other):
        right = other.data if isinstance(other, DummyTensor) else other
        fn = getattr(self.ops, "_AbstractTensorOperations__apply_operator")
        return DummyTensor(self.ops, fn(op, self.data, right))

    def __add__(self, other):
        return self._apply("add", other)

    def __radd__(self, other):
        return self._apply("radd", other)

    def __sub__(self, other):
        return self._apply("sub", other)

    def __rsub__(self, other):
        return self._apply("rsub", other)

    def __mul__(self, other):
        return self._apply("mul", other)

    def __truediv__(self, other):
        return self._apply("truediv", other)

    def __floordiv__(self, other):
        return self._apply("floordiv", other)

    def __mod__(self, other):
        return self._apply("mod", other)

    def __pow__(self, other):
        return self._apply("pow", other)

    def tolist(self):
        return self.ops.tolist(self.data)


@pytest.mark.parametrize("backend_cls", available_backends())
def test_magic_operator_dispatch(backend_cls):
    """Verify arithmetic overloads funnel to backend ops."""
    ops = backend_cls()
    a = DummyTensor(ops, [1, 2])
    b = DummyTensor(ops, [3, 4])
    assert (a + b).tolist() == [4, 6]
    assert (b - a).tolist() == [2, 2]
    assert (a * b).tolist() == [3, 8]
    assert (b / a).tolist() == [3.0, 2.0]
    assert (b // a).tolist() == [3 // 1, 4 // 2]
    assert (b % a).tolist() == [0, 0]
    assert (a ** a).tolist() == [1, 4]


@pytest.mark.parametrize("backend_cls", available_backends())
def test_apply_operator_inaccessible(backend_cls):
    """Ensure private operator helper cannot be called directly."""
    ops = backend_cls()
    with pytest.raises(AttributeError):
        ops._apply_operator("add", [1], [2])

