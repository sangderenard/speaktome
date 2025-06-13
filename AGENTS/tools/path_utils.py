#!/usr/bin/env python3
"""Path utilities for locating the repository root."""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by searching for ``pyproject.toml``."""
    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current

__all__ = ["find_repo_root"]
