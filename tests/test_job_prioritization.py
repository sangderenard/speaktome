"""Tests for job prioritization and dispenser utilities."""

from pathlib import Path

import AGENTS.tools.prioritize_jobs as pj
from AGENTS.tools.dispense_job import choose_job

# --- END HEADER ---


def test_priority_ranking() -> None:
    ranking = pj.calculate_priorities()
    job_files = {p.name for p in Path('AGENTS/job_descriptions').glob('*_job.md')}
    assert set(ranking) == job_files
    assert set(ranking.values()) == set(range(1, len(job_files) + 1))


def test_choose_job_seeded() -> None:
    job = choose_job(seed=1)
    assert job in pj.calculate_priorities()
