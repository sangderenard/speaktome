"""CPU-only demo exercising :class:`LookaheadController`.

This lightweight path demonstrates how the project can operate with
either NumPy or a pure Python fallback. A simple random model drives the
lookahead search using the generic tensor and model wrappers. The demo
prints the top ``k`` results after ``d`` lookahead steps.
"""

import argparse
from typing import Any, Dict

from . import Faculty

FACULTY_REQUIREMENT = Faculty.PURE_PYTHON
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

from .util.token_vocab import TokenVocabulary
from .core.tensor_abstraction import get_tensor_operations
from .core.model_abstraction import AbstractModelWrapper
from .core.lookahead_controller import LookaheadController, LookaheadConfig
# --- END HEADER ---

VOCAB = TokenVocabulary("abcdefghijklmnopqrstuvwxyz ")


class RandomModel(AbstractModelWrapper):
    """Dummy model producing random logits for each token position."""

    def forward(self, input_ids: Any, attention_mask: Any, **kwargs) -> Dict[str, Any]:
        if NUMPY_AVAILABLE:
            batch, width = input_ids.shape
            logits = np.random.rand(batch, width, len(VOCAB)).astype(np.float32)
        else:
            batch = len(input_ids)
            width = len(input_ids[0]) if batch > 0 else 0
            import random
            logits = [
                [
                    [random.random() for _ in range(len(VOCAB))]
                    for _ in range(width)
                ]
                for _ in range(batch)
            ]
        return {"logits": logits}

    def get_device(self) -> Any:
        return "cpu"


def aggregate_mean(scores: Any) -> Any:
    """Aggregate candidate scores using mean over sequence length."""
    ops = get_tensor_operations(
        Faculty.NUMPY if NUMPY_AVAILABLE else Faculty.PURE_PYTHON
    )
    return ops.mean(scores, dim=-1)


def run_lookahead(prefix: str, lookahead_steps: int = 5, beam_width: int = 3):
    """Execute LookaheadController with a random backend."""
    ops = get_tensor_operations(
        Faculty.NUMPY if NUMPY_AVAILABLE else Faculty.PURE_PYTHON
    )
    tokenizer = type("_T", (), {"pad_token_id": 0})()
    config = LookaheadConfig(
        instruction=None,
        lookahead_top_k=beam_width,
        lookahead_temp=1.0,
        aggregate_fn=aggregate_mean,
    )
    controller = LookaheadController(
        lookahead_steps=lookahead_steps,
        max_len=len(prefix) + lookahead_steps,
        device="cpu",
        tokenizer=tokenizer,
        config=config,
        tensor_ops=ops,
        model_wrapper=RandomModel(),
    )

    if NUMPY_AVAILABLE:
        prefix_ids = np.array([VOCAB.encode(prefix)], dtype=np.int64)
        prefix_scores = np.zeros_like(prefix_ids, dtype=np.float32)
        prefix_lens = np.array([prefix_ids.shape[1]], dtype=np.int64)
        parent = np.array([0], dtype=np.int64)
    else:
        prefix_ids = [VOCAB.encode(prefix)]
        prefix_scores = [[0.0] * len(prefix_ids[0])]
        prefix_lens = [len(prefix_ids[0])]
        parent = [0]

    tokens, scores, lengths, *_ = controller.run(
        prefix_ids, prefix_scores, prefix_lens, parent
    )

    agg = aggregate_mean(scores)
    if NUMPY_AVAILABLE:
        order = np.argsort(agg)[::-1][:beam_width]
        beams = [VOCAB.decode(tokens[i, : lengths[i]]) for i in order]
        beam_scores = [float(agg[i]) for i in order]
    else:
        order = sorted(range(len(agg)), key=lambda i: agg[i], reverse=True)[:beam_width]
        beams = [
            VOCAB.decode(tokens[i][: lengths[i]]) for i in order
        ]
        beam_scores = [float(agg[i]) for i in order]
    return beams, beam_scores


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="CPU lookahead demo")
    parser.add_argument('-s', '--seed', type=str, default='')
    parser.add_argument('-l', '--lookahead', '--depth', '-d', type=int, dest='lookahead', default=5)
    parser.add_argument('-k', '--beam_width', '--width', '-w', type=int, dest='beam_width', default=3)
    args, unknown = parser.parse_known_args(raw_args)

    current_faculty = Faculty.NUMPY if NUMPY_AVAILABLE else Faculty.PURE_PYTHON
    print(f"Faculty level: {current_faculty.name}")

    extras = []
    skip_next = False
    ignored = []
    for i, tok in enumerate(unknown):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith('-'):
            ignored.append(tok)
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('-'):
                skip_next = True
        else:
            extras.append(tok)
    if ignored:
        print(f"Ignoring unrecognized arguments in CPU demo: {ignored}")

    seed_parts = []
    if args.seed:
        seed_parts.append(args.seed)
    if extras:
        seed_parts.extend(extras)
    seed = ' '.join(seed_parts)

    beams, scores = run_lookahead(seed, args.lookahead, args.beam_width)
    print("Top sequences:")
    for b, s in zip(beams, scores):
        print(f"{b} (score={s:.2f})")

if __name__ == '__main__':
    main()
