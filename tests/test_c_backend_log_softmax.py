"""CTensorOperations.log_softmax behaviour."""

import math
import pytest

from speaktome.tensors.c_backend import CTensorOperations


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
