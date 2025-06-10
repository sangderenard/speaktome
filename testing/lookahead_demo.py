import types
import sys
try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class DummyModel(AbstractModelWrapper):
    def forward(self, input_ids, attention_mask, **kwargs):
        batch, width = input_ids.shape
        logits = np.zeros((batch, width, 3), dtype=np.float32)
        logits[:, :, 1] = 1.0
        return {"logits": logits}

    def get_device(self):
        return "cpu"

class DummyTokenizer:
    pad_token_id = 0


def _choose_ops():
    return get_tensor_operations(DEFAULT_FACULTY)


def aggregate(scores):
    ops = _choose_ops()
    return ops.mean(scores, dim=-1)


def main():
    ops = _choose_ops()
    model = DummyModel()
    tokenizer = DummyTokenizer()
    cfg = LookaheadConfig(
        instruction=None,
        lookahead_top_k=2,
        lookahead_temp=1.0,
        aggregate_fn=aggregate,
    )
    controller = LookaheadController(
        lookahead_steps=1,
        max_len=3,
        device="cpu",
        tokenizer=tokenizer,
        config=cfg,
        tensor_ops=ops,
        model_wrapper=model,
    )

    if isinstance(ops, NumPyTensorOperations) and np is not None:
        prefix_tokens = np.array([[1]], dtype=np.int64)
        prefix_scores = np.array([[0.0]], dtype=np.float32)
        prefix_lengths = np.array([1], dtype=np.int64)
        parent_idx = np.array([0], dtype=np.int64)
    else:
        prefix_tokens = [[1]]
        prefix_scores = [[0.0]]
        prefix_lengths = [1]
        parent_idx = [0]

    out = controller.run(prefix_tokens, prefix_scores, prefix_lengths, parent_idx)
    tokens, scores, lengths, parents, parent_lens, pruned = out
    print("Tokens:", tokens)
    print("Scores:", scores)
    print("Lengths:", lengths)
    print("Parents:", parents)
    print("Pruned:", pruned)

if __name__ == "__main__":
    main()
