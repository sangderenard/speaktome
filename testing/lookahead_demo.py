import types
import sys
import numpy as np

# Provide dummy sentence_transformers module so speaktome.config imports succeed
stub_st = types.ModuleType("sentence_transformers")
stub_st.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", stub_st)

stub_tx = types.ModuleType("transformers")
class _DummyTok: pass
stub_tx.PreTrainedTokenizer = _DummyTok
sys.modules.setdefault("transformers", stub_tx)

from speaktome.tensor_abstraction import NumPyTensorOperations
from speaktome.model_abstraction import AbstractModelWrapper
from speaktome.beam_search import LookaheadController, LookaheadConfig

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


def aggregate(scores):
    ops = NumPyTensorOperations()
    return ops.mean(scores, dim=-1)


def main():
    ops = NumPyTensorOperations()
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

    prefix_tokens = np.array([[1]], dtype=np.int64)
    prefix_scores = np.array([[0.0]], dtype=np.float32)
    prefix_lengths = np.array([1], dtype=np.int64)
    parent_idx = np.array([0], dtype=np.int64)

    out = controller.run(prefix_tokens, prefix_scores, prefix_lengths, parent_idx)
    tokens, scores, lengths, parents, parent_lens, pruned = out
    print("Tokens:", tokens)
    print("Scores:", scores)
    print("Lengths:", lengths)
    print("Parents:", parents)
    print("Pruned:", pruned)

if __name__ == "__main__":
    main()
