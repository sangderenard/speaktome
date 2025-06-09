#!/usr/bin/env python3
"""Comprehensive tensor backend checks."""

from __future__ import annotations

try:
    import importlib.util
    import logging
    import pytest
    import itertools


    from speaktome.tensors import (
        PurePythonTensorOperations,
        NumPyTensorOperations,
        PyTorchTensorOperations,
        JAXTensorOperations,
    )
    from speaktome.tensors.faculty import detect_faculty

    from speaktome.tensors.pure_backend import PurePythonTensorOperations  # For isinstance check

except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
# --- END HEADER ---

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
    t = ops.tensor_from_list([[1, 2], [3, 4]], dtype=ops.float_dtype, device=None)
    stacked0 = ops.benchmark(lambda: ops.stack([t, t], dim=0))
    assert ops.tolist(stacked0) == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    stacked1 = ops.benchmark(lambda: ops.stack([t, t], dim=1))
    assert ops.tolist(stacked1) == [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]

    rng = list(range(3))
    data = ops.tensor_from_list(
        [[i + j for j in rng] for i in rng], dtype=ops.float_dtype, device=None
    )
    stacked0 = ops.benchmark(lambda: ops.stack([data, data], dim=0))
    assert ops.tolist(stacked0) == [ops.tolist(data), ops.tolist(data)]
    stacked1 = ops.benchmark(lambda: ops.stack([data, data], dim=1))
    data_list = ops.tolist(data)
    assert ops.tolist(stacked1) == [[row, row] for row in data_list]

    padded = ops.benchmark(lambda: ops.pad(t, (1, 1, 1, 1), value=0))
    assert ops.tolist(padded) == [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]

    base = ops.tensor_from_list(
        [[i for i in range(3)] for _ in range(2)], dtype=ops.float_dtype, device=None
    )
    padded = ops.benchmark(lambda: ops.pad(base, (0, 2, 1, 0), value=9))
    padded_list = ops.tolist(padded)
    assert len(padded_list) == 3
    assert all(len(row) == 5 for row in padded_list)

    topk_input = ops.tensor_from_list([1, 3, 2, 4], dtype=ops.float_dtype, device=None)
    values, indices = ops.benchmark(lambda: ops.topk(topk_input, k=2, dim=-1))
    assert ops.tolist(values) == [4, 3]
    assert ops.tolist(indices) == [3, 1]

    seq_list = [7, 1, 4, 9, 2, 6]
    seq = ops.tensor_from_list(seq_list, dtype=ops.float_dtype, device=None)
    vals, idxs = ops.benchmark(lambda: ops.topk(seq, k=3, dim=-1))
    expect_vals = sorted(seq_list, reverse=True)[:3]
    expect_inds = [seq_list.index(v) for v in expect_vals]
    assert ops.tolist(vals) == expect_vals
    assert ops.tolist(idxs) == expect_inds

    # Multi-dimensional topk
    matrix = ops.tensor_from_list(
        [[1, 5, 3], [4, 2, 6]], dtype=ops.float_dtype, device=None
    )
    tvals, tidxs = ops.benchmark(lambda: ops.topk(matrix, k=2, dim=1))
    assert ops.tolist(tvals) == [[5, 3], [6, 4]]
    assert ops.tolist(tidxs) == [[1, 2], [2, 0]]

    # Test operators on tensors created by the backend
    try:
        # Use float_dtype for these operations
        dtype = ops.float_dtype

        data_sub = ops.tensor_from_list([2.0, 4.0, 6.0], dtype=dtype, device=None)
        # Python float scalar, relying on backend's operator broadcasting/handling
        assert ops.tolist(data_sub - 1.0) == [1.0, 3.0, 5.0]

        data_div = ops.tensor_from_list([2.0, 4.0], dtype=dtype, device=None)  # type: ignore
        assert ops.tolist(data_div / 2.0) == [1.0, 2.0]
    except (TypeError, NotImplementedError, AttributeError) as exc:
        pytest.skip(f"Operator dispatch not supported: {exc}")

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


@pytest.mark.parametrize("backend_cls", available_backends())
def test_basic_operator_dispatch(backend_cls):
    """Verify arithmetic helpers via the private dispatcher."""
    if backend_cls is PurePythonTensorOperations:
        pytest.skip("Pure Python backend lacks operator overloading")

    ops = backend_cls()
    if isinstance(ops, PurePythonTensorOperations):
        pytest.skip("Pure Python backend lacks operator dispatch")

    a_list_float = [1.0, 2.0]
    b_list_float = [3.0, 4.0]
    a_list_int = [1, 2]
    b_list_int = [3, 4]

    if isinstance(ops, PurePythonTensorOperations):
        pytest.skip("Pure backend does not support arithmetic operators")

    try:
        # Use backend-specific dtypes
        float_dtype = ops.float_dtype
        long_dtype = ops.long_dtype

        # Create tensors using the backend's tensor_from_list
        a_float = ops.tensor_from_list(a_list_float, dtype=float_dtype, device=None)
        b_float = ops.tensor_from_list(b_list_float, dtype=float_dtype, device=None)

        a_int = ops.tensor_from_list(a_list_int, dtype=long_dtype, device=None)
        b_int = ops.tensor_from_list(b_list_int, dtype=long_dtype, device=None)

        # Test operators on these tensors
        assert _norm(ops.tolist(a_float + b_float)) == [4.0, 6.0]
        assert _norm(ops.tolist(b_float - a_float)) == [2.0, 2.0]
        assert _norm(ops.tolist(a_float * b_float)) == [3.0, 8.0]
        assert _norm(ops.tolist(b_float / a_float)) == [3.0, 2.0]

        # For floor division and modulo, integer-like inputs are typical
        assert _norm(ops.tolist(b_int // a_int)) == [3, 2]
        assert _norm(ops.tolist(b_int % a_int)) == [0, 0]

        # Power can use float or int base/exponent depending on desired outcome
        # Using a_float for potentially float results (e.g. non-integer exponents)
        # Here, a_float ** a_float (e.g., 1.0**1.0, 2.0**2.0)
        assert _norm(ops.tolist(a_float**a_float)) == [1.0, 4.0]

    except (TypeError, NotImplementedError, AttributeError):

        raise # Let the test fail if operators are not supported as expected


@pytest.mark.parametrize(
    "src_cls,tgt_cls",
    itertools.permutations(available_backends(), 2),
)
def test_to_backend_roundtrip(src_cls, tgt_cls):
    """Ensure ``to_backend`` converts tensors faithfully across backends."""
    src_ops = src_cls()
    tgt_ops = tgt_cls()
    data = [[1, 2], [3, 4]]
    tensor = src_ops.tensor_from_list(data, dtype=src_ops.float_dtype, device=None)
    converted = src_ops.to_backend(tensor, tgt_ops)
    assert tgt_ops.tolist(converted) == data
    roundtrip = tgt_ops.to_backend(converted, src_ops)
    assert src_ops.tolist(roundtrip) == data
        


def test_pure_python_matmul():
    """Verify matrix multiplication for the pure python backend."""
    ops = PurePythonTensorOperations()
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = ops._AbstractTensorOperations__apply_operator("matmul", a, b)
    assert result == [[19, 22], [43, 50]]
