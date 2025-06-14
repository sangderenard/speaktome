#!/usr/bin/env python3
"""Compute rank-based priorities for job descriptions.

Job descriptions live under ``AGENTS/job_descriptions``. This utility
counts words in each ``*_job.md`` file and ranks the files from most to
least verbose. The resulting mapping ``{filename: rank}`` is written to
``job_priority.json`` within the same directory.
"""
from __future__ import annotations

import json
from pathlib import Path
def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    required = {
        "speaktome",
        "laplace",
        "tensorprinting",
        "timesync",
        "AGENTS",
        "fontmapper",
        "tensors",
    }
    for parent in [current, *current.parents]:
        if all((parent / name).exists() for name in required):
            return parent
    return current

JOB_DIR = _find_repo_root(Path(__file__)) / "AGENTS" / "job_descriptions"
OUT_FILE = JOB_DIR / "job_priority.json"
# --- END HEADER ---


def calculate_priorities(job_dir: Path = JOB_DIR) -> dict[str, int]:
    """Return ``{filename: rank}`` sorted by word count descending."""
    word_counts = {
        p.name: len(p.read_text(encoding="utf-8").split())
        for p in job_dir.glob("*_job.md")
    }
    ranking = sorted(word_counts.items(), key=lambda kv: -kv[1])
    return {name: idx + 1 for idx, (name, _) in enumerate(ranking)}


def main() -> None:  # pragma: no cover - CLI utility
    ranking = calculate_priorities()
    OUT_FILE.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    for name, rank in ranking.items():
        print(f"{rank}\t{name}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
