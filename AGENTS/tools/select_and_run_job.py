#!/usr/bin/env python3
"""Interactive menu for job selection with optional command execution."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

from .prioritize_jobs import calculate_priorities, JOB_DIR

# --- END HEADER ---


def list_jobs() -> List[str]:
    ranking = calculate_priorities()
    return [name for name, _ in sorted(ranking.items(), key=lambda kv: kv[1])]


def display_menu(jobs: List[str]) -> str:
    for idx, job in enumerate(jobs, 1):
        print(f"{idx}. {job}")
    print("0. Quit")
    choice = input("Select a job: ")
    if choice == "0":
        raise SystemExit(0)
    idx = int(choice) - 1
    return jobs[idx]


def extract_commands(text: str) -> List[str]:
    cmds: List[str] = []
    for line in text.splitlines():
        if "run" in line.lower() and "`" in line:
            start = line.find("`") + 1
            end = line.find("`", start)
            if end > start:
                cmds.append(line[start:end])
    return cmds


def run_job(job_name: str) -> None:
    path = JOB_DIR / job_name
    text = path.read_text(encoding="utf-8")
    print("\n" + text)
    cmds = extract_commands(text)
    for cmd in cmds:
        print(f"\n[Running] {cmd}")
        subprocess.run(cmd, shell=True, check=False)


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    jobs = list_jobs()
    job = display_menu(jobs)
    run_job(job)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
