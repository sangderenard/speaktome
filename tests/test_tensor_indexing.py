import importlib.util
import pytest
from tensors import PurePythonTensorOperations, NumPyTensorOperations


def available_ops():
    ops = [PurePythonTensorOperations]
    if importlib.util.find_spec("numpy") is not None:
        ops.append(NumPyTensorOperations)
    return ops


@pytest.mark.parametrize("backend_cls", available_ops())
def test_get_set_item(backend_cls):
    ops = backend_cls()
    t = ops.tensor_from_list([[1, 2], [3, 4]], dtype=ops.float_dtype, device=None)
    # get
    first_row = t[0]
    assert isinstance(first_row, backend_cls)
    assert first_row[1] == 2
    # set
    t[1, 0] = 9
    assert ops.tolist(t)[1][0] == 9


@pytest.mark.parametrize("backend_cls", available_ops())
def test_tensor_indices(backend_cls):
    ops = backend_cls()
    t = ops.tensor_from_list([[10, 20], [30, 40]], dtype=ops.float_dtype, device=None)

    row_idx = ops.full((), 1, dtype=ops.long_dtype, device=None)
    col_idx = ops.full((), 0, dtype=ops.long_dtype, device=None)

    value = t[row_idx][col_idx]
    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    assert value == 30

    t[row_idx, col_idx] = ops.full((), 99, dtype=ops.float_dtype, device=None)
    assert ops.tolist(t)[1][0] == 99
