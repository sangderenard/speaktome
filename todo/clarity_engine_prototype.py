# ########## STUB: clarity_engine ##########
# PURPOSE: Generate human-friendly summaries of failing test logs or
#          environment setup output, highlighting actionable insights.
# EXPECTED BEHAVIOR: Parse raw log lines, classify them by severity,
#          and emit Markdown sections explaining each issue.
# INPUTS: log text from CI runs or setup scripts.
# OUTPUTS: Markdown-formatted summary with links or commands to resolve
#          common problems.
# KEY ASSUMPTIONS/DEPENDENCIES: relies on regex patterns for known
#          errors; may integrate with PrettyLogger for structured output.
# TODO:
#   - Implement log parsing routines and severity detection.
#   - Add template rendering for Markdown summaries.
#   - Provide CLI wrapper for standalone use.
# NOTES: This stub realizes the "Clarity Engine" role proposed by
#        GPT-4o in the project messages.
# ###########################################################################

import re
from collections import defaultdict
from typing import Iterable


def _collect_lines(patterns: Iterable[str], lines: list[str]) -> list[str]:
    out: list[str] = []
    regex = re.compile('|'.join(patterns), re.IGNORECASE)
    for ln in lines:
        if regex.search(ln):
            out.append(ln.strip())
    return out


def summarize_logs(log_text: str) -> str:
    """Return a Markdown summary of the given log text.

    The function scans ``log_text`` for common terms such as ``ERROR`` or
    ``WARNING`` and returns a short markdown report grouping the matching
    lines. It is intentionally lightweight and does not attempt exhaustive
    parsing of every possible log format.
    """

    lines = log_text.splitlines()

    sections = defaultdict(list)
    sections['Errors'] = _collect_lines(['error', 'failed'], lines)
    sections['Warnings'] = _collect_lines(['warning'], lines)
    sections['Info'] = _collect_lines(['info'], lines)

    pieces = []
    for name, hits in sections.items():
        if not hits:
            continue
        pieces.append(f"## {name} ({len(hits)})")
        for ln in hits[:10]:
            pieces.append(f"- {ln}")
    if not pieces:
        return "No issues detected."
    return "\n".join(pieces)


if __name__ == "__main__":
    sample = "ERROR: missing dependency"  # Placeholder
    print(summarize_logs(sample))
