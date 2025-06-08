#!/usr/bin/env python3
"""Dispense a job using a Zipf distribution over priority ranks."""
from __future__ import annotations

import argparse
import random

from .prioritize_jobs import calculate_priorities

# --- END HEADER ---


def choose_job(s: float = 1.0, seed: int | None = None) -> str:
    """Return the name of a job chosen via Zipf distribution."""
    ranking = calculate_priorities()
    items = sorted(ranking.items(), key=lambda kv: kv[1])
    n = len(items)
    weights = [1 / (rank ** s) for _, rank in items]
    total = sum(weights)
    weights = [w / total for w in weights]
    rng = random.Random(seed)
    idx = rng.choices(range(n), weights=weights, k=1)[0]
    return items[idx][0]


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", type=float, default=1.0, help="Zipf exponent")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    job = choose_job(s=args.s, seed=args.seed)
    print(job)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
