#!/usr/bin/env python3
"""Check next_token_logprob_score across tensor backends."""

from __future__ import annotations

try:
    import os
    import importlib.util
    import pytest

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    from tensors import (
        PurePythonTensorOperations,
        NumPyTensorOperations,
        PyTorchTensorOperations,
        JAXTensorOperations,
    )
    from speaktome.core.scorer import Scorer
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def available_backends():
    backends = [PurePythonTensorOperations]
    if importlib.util.find_spec("numpy") is not None:
        backends.append(NumPyTensorOperations)
    if importlib.util.find_spec("torch") is not None:
        backends.append(PyTorchTensorOperations)
    if importlib.util.find_spec("jax") is not None:
        backends.append(JAXTensorOperations)
    return backends


@pytest.mark.parametrize("backend_cls", available_backends())
def test_next_token_logprob_score_backend(backend_cls):
    ops = backend_cls.get_tensor()
    beams = backend_cls.tensor_from_list([[1, 2, 3], [4, 5, 6]], dtype=ops.long_dtype, device=None)
    scores = backend_cls.tensor_from_list(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=ops.float_dtype, device=None
    )
    lengths = backend_cls.tensor_from_list([2, 3], dtype=ops.long_dtype, device=None)
    tok = type("T", (), {"pad_token_id": 0})()

    result = Scorer.next_token_logprob_score(
        beams=beams, scores=scores, lengths=lengths, tokenizer=tok
    )
    expected = [0.2, 0.6]
    assert result.tolist() == expected
