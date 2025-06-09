"""LookaheadController backend integration tests."""

import logging
import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from speaktome.core.lookahead_controller import LookaheadController, LookaheadConfig
from speaktome.core.model_abstraction import AbstractModelWrapper
from speaktome.tensors import get_tensor_operations
from speaktome.tensors.faculty import available_faculties

# --- END HEADER ---

logger = logging.getLogger(__name__)


class DummyModel(AbstractModelWrapper):
    def forward(self, input_ids, attention_mask, **kwargs):
        if hasattr(input_ids, "shape"):
            batch, width = input_ids.shape
        else:
            batch = len(input_ids)
            width = len(input_ids[0]) if batch else 0
        if np is not None:
            logits = np.zeros((batch, width, 3), dtype=np.float32)
            logits[:, :, 1] = 1.0
        else:
            logits = [[[0.0, 1.0, 0.0] for _ in range(width)] for _ in range(batch)]
        return {"logits": logits}

    def get_device(self):
        return "cpu"


class DummyTokenizer:
    pad_token_id = 0


def aggregate_mean(ops):
    def inner(scores):
        try:
            return ops.mean(scores, dim=-1)
        except Exception:
            try:
                return ops.mean(scores, dim=1)
            except Exception:
                return ops.mean(scores, dim=None)
    return inner


@pytest.mark.parametrize("faculty", available_faculties())
def test_lookahead_across_backends(faculty, caplog):
    if faculty.name == "PURE_PYTHON":
        pytest.skip("Pure Python backend lacks advanced ops")
    ops = None
    try:
        ops = get_tensor_operations(faculty, track_time=True)
    except Exception as exc:  # pragma: no cover - backend missing
        pytest.skip(f"backend {faculty.name} unavailable: {exc}")

    cfg = LookaheadConfig(
        instruction=None,
        lookahead_top_k=2,
        lookahead_temp=1.0,
        aggregate_fn=aggregate_mean(ops),
    )
    controller = LookaheadController(
        lookahead_steps=1,
        max_len=3,
        device="cpu",
        tokenizer=DummyTokenizer(),
        config=cfg,
        tensor_ops=ops,
        model_wrapper=DummyModel(),
    )

    ptoks = ops.tensor_from_list([[1]], dtype=ops.long_dtype, device="cpu")
    pscores = ops.tensor_from_list([[0.0]], dtype=ops.float_dtype, device="cpu")
    plens = ops.tensor_from_list([1], dtype=ops.long_dtype, device="cpu")
    pidx = ops.tensor_from_list([0], dtype=ops.long_dtype, device="cpu")

    with caplog.at_level(logging.INFO):
        result = ops.benchmark(lambda: controller.run(ptoks, pscores, plens, pidx))
    tokens, _, lengths, *_ = result
    assert ops.shape(tokens)[0] > 0
    assert ops.shape(lengths)[0] > 0
    if ops.last_op_time is not None:
        logger.info("%s backend lookahead time: %.6f", faculty.name, ops.last_op_time)


def test_lookahead_default_backend():
    facs = available_faculties()
    if facs == [facs[0]] and facs[0].name == "PURE_PYTHON":
        pytest.skip("Pure Python backend only")
    cfg = LookaheadConfig(
        instruction=None,
        lookahead_top_k=1,
        lookahead_temp=1.0,
        aggregate_fn=lambda s: get_tensor_operations().mean(s, dim=-1),
    )
    controller = LookaheadController(
        lookahead_steps=1,
        max_len=2,
        device="cpu",
        tokenizer=DummyTokenizer(),
        config=cfg,
        tensor_ops=None,
        model_wrapper=DummyModel(),
    )

    ops = controller.tensor_ops
    ptoks = ops.tensor_from_list([[1]], dtype=ops.long_dtype, device="cpu")
    pscores = ops.tensor_from_list([[0.0]], dtype=ops.float_dtype, device="cpu")
    plens = ops.tensor_from_list([1], dtype=ops.long_dtype, device="cpu")
    pidx = ops.tensor_from_list([0], dtype=ops.long_dtype, device="cpu")

    result = ops.benchmark(lambda: controller.run(ptoks, pscores, plens, pidx))
    tokens, _, _, *_ = result
    assert ops.shape(tokens)[0] > 0
    assert ops.last_op_time is not None
