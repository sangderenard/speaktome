#!/usr/bin/env python3
"""CTensorOperations.log_softmax behaviour."""

from __future__ import annotations

try:
    import math
    import pytest

    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from tensors.c_backend import CTensorOperations
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def test_log_softmax_1d():
    ops = CTensorOperations()
    tensor = ops.tensor_from_list([1.0, 2.0, 3.0], dtype=ops.float_dtype, device=None)
    result = ops.log_softmax(tensor, dim=0)
    values = ops.tolist(result)

    max_v = 3.0
    exps = [math.exp(1.0 - max_v), math.exp(2.0 - max_v), math.exp(3.0 - max_v)]
    total = sum(exps)
    expected = [math.log(v / total) for v in exps]

    assert pytest.approx(values) == expected


def test_log_softmax_nd():
    ops = CTensorOperations()
    tensor = ops.tensor_from_list(
        [[1.0, 2.0], [3.0, 4.0]], dtype=ops.float_dtype, device=None
    )
    result = ops.log_softmax(tensor, dim=1)
    values = ops.tolist(result)

    expected = []
    for row in [[1.0, 2.0], [3.0, 4.0]]:
        max_v = max(row)
        exps = [math.exp(v - max_v) for v in row]
        total = sum(exps)
        expected.append([math.log(v / total) for v in exps])

    for row_val, row_exp in zip(values, expected):
        assert row_val == pytest.approx(row_exp)
