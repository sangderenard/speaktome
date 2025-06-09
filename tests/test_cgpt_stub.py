import importlib

from speaktome.tensors.models.c_gpt import cgpt


def test_cgpt_forward_placeholder():
    model = cgpt.CGPT()
    out = model.forward([1.0, 2.0, 3.0])
    assert out == [0.0, 0.0, 0.0]
