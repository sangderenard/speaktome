import pytest

from tensors import get_tensor_operations
from speaktome.core.abstract_linear_net import AbstractLinearLayer, SequentialLinearModel

# --- END HEADER ---


def test_sequential_linear_model_forward():
    ops = get_tensor_operations()
    w0 = ops.tensor_from_list([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=ops.float_dtype, device=None)
    b0 = ops.tensor_from_list([[0.5, -0.5]], dtype=ops.float_dtype, device=None)
    layer0 = AbstractLinearLayer(w0, b0, ops)

    w1 = ops.tensor_from_list([[1.0], [2.0]], dtype=ops.float_dtype, device=None)
    b1 = ops.tensor_from_list([[0.0]], dtype=ops.float_dtype, device=None)
    layer1 = AbstractLinearLayer(w1, b1, ops)

    model = SequentialLinearModel([layer0, layer1], ops)
    x = ops.tensor_from_list([[2.0, 3.0, 1.0]], dtype=ops.float_dtype, device=None)
    result = model.forward(x, None)["logits"]

    manual = ops._AbstractTensor__apply_operator("matmul", x, w0)
    manual = ops._AbstractTensor__apply_operator("add", manual, b0)
    manual = ops._AbstractTensor__apply_operator("matmul", manual, w1)
    manual = ops._AbstractTensor__apply_operator("add", manual, b1)

    assert ops.tolist(result) == ops.tolist(manual)


def test_sequential_linear_model_with_activation_and_from_weights():
    ops = get_tensor_operations()
    w0 = ops.tensor_from_list([[1.0, -1.0]], dtype=ops.float_dtype, device=None)
    b0 = ops.tensor_from_list([[0.0, 0.0]], dtype=ops.float_dtype, device=None)
    w1 = ops.tensor_from_list([[1.0], [1.0]], dtype=ops.float_dtype, device=None)
    b1 = ops.tensor_from_list([[0.0]], dtype=ops.float_dtype, device=None)

    model = SequentialLinearModel.from_weights([(w0, b0), (w1, b1)], ops, activation="relu")
    x = ops.tensor_from_list([[1.0]], dtype=ops.float_dtype, device=None)
    result = model.forward(x, None)["logits"]

    manual = ops._AbstractTensor__apply_operator("matmul", x, w0)
    manual = ops._AbstractTensor__apply_operator("add", manual, b0)
    manual = ops.clamp(manual, min_val=0.0)
    manual = ops._AbstractTensor__apply_operator("matmul", manual, w1)
    manual = ops._AbstractTensor__apply_operator("add", manual, b1)

    assert ops.tolist(result) == ops.tolist(manual)
