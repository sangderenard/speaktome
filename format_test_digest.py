#!/usr/bin/env python3
"""Format test logs into a compact Markdown digest.

This utility searches a pytest log for tagged messages such as
``[FACULTY_SKIP]`` or ``[AGENT_ACTIONABLE_ERROR]`` and summarises the
findings in Markdown.  If a directory is supplied rather than a file it
selects the most recent ``pytest_*.log`` entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- END HEADER ---

DEFAULT_LOG_DIR = Path("testing/logs")
TAG_RE = re.compile(r"\[(FACULTY_SKIP|AGENT_ACTIONABLE_ERROR|TEST_PASS)\]")


def _latest_log_file(log_dir: Path) -> Path | None:
    """Return the most recent ``pytest_*.log`` file in ``log_dir``."""
    logs = sorted(log_dir.glob("pytest_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def format_digest(log_path: Path) -> str:
    """Return a Markdown summary of tagged log lines."""
    if log_path.is_dir():
        latest = _latest_log_file(log_path)
        if not latest:
            return "No log file found."
        log_path = latest
    if not log_path.exists():
        return "No log file found."

    groups: dict[str, list[str]] = {
        "FACULTY_SKIP": [],
        "AGENT_ACTIONABLE_ERROR": [],
        "TEST_PASS": [],
    }

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TAG_RE.search(line)
        if match:
            groups.setdefault(match.group(1), []).append(line.strip())

    summary = [f"{tag}: {len(lines)}" for tag, lines in groups.items()]
    md = ["# Test Digest", "", f"Source log: `{log_path}`", ""]
    md.append("## Summary")
    for s in summary:
        md.append(f"- {s}")
    for tag, lines in groups.items():
        if not lines:
            continue
        md.append("")
        md.append(f"## {tag}")
        md.extend([f"- {line}" for line in lines])
    return "\n".join(md)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    print(format_digest(target))


if __name__ == "__main__":
    main()
