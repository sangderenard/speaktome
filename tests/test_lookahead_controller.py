"""LookaheadController backend integration tests."""

import logging
import pytest

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

logger = logging.getLogger(__name__)


class DummyModel(AbstractModelWrapper):
    """Minimal sequential linear network returning a fixed distribution."""

    def __init__(self, ops):
        self.ops = ops
        self.vocab_size = 3
        self.embed = self.ops.tensor_from_list(
            [[1.0 if i == j else 0.0 for j in range(self.vocab_size)] for i in range(self.vocab_size)],
            dtype=self.ops.float_dtype,
            device="cpu",
        )

        w0 = self.ops.zeros((self.vocab_size, 1), dtype=self.ops.float_dtype, device="cpu")
        b0 = self.ops.zeros((1, 1), dtype=self.ops.float_dtype, device="cpu")
        w1 = self.ops.zeros((1, self.vocab_size), dtype=self.ops.float_dtype, device="cpu")
        b1 = self.ops.tensor_from_list([[0.0, 1.0, 0.0]], dtype=self.ops.float_dtype, device="cpu")

        self.model = SequentialLinearModel(
            [AbstractLinearLayer(w0, b0, self.ops), AbstractLinearLayer(w1, b1, self.ops)],
            self.ops,
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        ops = self.ops
        if hasattr(input_ids, "shape"):
            batch, width = input_ids.shape
            flat_ids = input_ids.reshape(-1)
        else:
            batch = len(input_ids)
            width = len(input_ids[0]) if batch else 0
            flat_ids = [tok for row in input_ids for tok in row]

        one_hot = ops.index_select(self.embed, 0, flat_ids)
        logits_flat = self.model.forward(one_hot, None)["logits"]

        if hasattr(input_ids, "shape"):
            logits = logits_flat.reshape(batch, width, self.vocab_size)
        else:
            logits = []
            idx = 0
            for _ in range(batch):
                row = []
                for _ in range(width):
                    row.append(logits_flat[idx])
                    idx += 1
                logits.append(row)

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
        model_wrapper=DummyModel(ops),
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


@pytest.mark.parametrize("steps", [1, 2])
def test_lookahead_default_backend(steps):
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
        lookahead_steps=steps,
        max_len=2,
        device="cpu",
        tokenizer=DummyTokenizer(),
        config=cfg,
        tensor_ops=get_tensor_operations(track_time=True),
        model_wrapper=DummyModel(get_tensor_operations()),
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
