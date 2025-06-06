"""Minimal CPU-only beam search demo using NumPy."""

import argparse
import numpy as np
from .token_vocab import TokenVocabulary
from .array_utils import topk

VOCAB = TokenVocabulary("abcdefghijklmnopqrstuvwxyz ")


def extend_once(prefix, lookahead_steps=5, beam_width=3):
    """Simplistic CPU beam search without neural networks."""
    beams = [prefix]
    scores = [0.0]
    for _ in range(lookahead_steps):
        new_beams = []
        new_scores = []
        for b, s in zip(beams, scores):
            probs = np.random.rand(len(VOCAB))
            _, idx = topk(probs, beam_width, dim=0)
            for i in idx:
                token = VOCAB.id_to_token[int(i)]
                new_beams.append(b + token)
                new_scores.append(s + float(probs[int(i)]))
        order = np.argsort(new_scores)[-beam_width:]
        beams = [new_beams[i] for i in order]
        scores = [new_scores[i] for i in order]
    return beams, scores


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="CPU lookahead demo")
    parser.add_argument('-s', '--seed', type=str, default='')
    parser.add_argument('-l', '--lookahead', type=int, default=5)
    parser.add_argument('-k', '--beam_width', type=int, default=3)
    args = parser.parse_args(raw_args)

    beams, scores = extend_once(args.seed, args.lookahead, args.beam_width)
    print("Top sequences:")
    for b, s in zip(beams, scores):
        print(f"{b} (score={s:.2f})")

if __name__ == '__main__':
    main()
