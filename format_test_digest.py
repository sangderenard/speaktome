#!/usr/bin/env python3
"""Format test logs into a compact Markdown digest."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ########## STUB: log digest formatter ##########
# PURPOSE: Parse pytest or script output and summarise important markers.
# EXPECTED BEHAVIOR: Recognise tags like ``[FACULTY_SKIP]`` and
# ``[AGENT_ACTIONABLE_ERROR]`` and generate a readable Markdown report.
# INPUTS: Path to a log file (default ``testing/logs/latest.log``).
# OUTPUTS: Markdown summary printed to stdout.
# KEY ASSUMPTIONS/DEPENDENCIES: Log lines contain square bracket tags.
# TODO:
#   - Produce per-module and per-class sections.
#   - Provide summary counts at the top of the digest.
# NOTES: This initial version only extracts tagged lines without structure.
# ###########################################################################

DEFAULT_LOG = Path("testing/logs/latest.log")
TAG_RE = re.compile(r"\[(FACULTY_SKIP|AGENT_ACTIONABLE_ERROR|TEST_PASS)\]")


def format_digest(log_path: Path) -> str:
    if not log_path.exists():
        return "No log file found."
    lines = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if TAG_RE.search(line):
            lines.append(line)
    md = ["# Test Digest", ""]
    md.extend(lines)
    return "\n".join(md)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    print(format_digest(path))


if __name__ == "__main__":
    main()
