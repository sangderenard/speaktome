#!/usr/bin/env python
"""Generate a concise digest of recent reports and messages.

This script is intended for sharing context with an agent that cannot
access the repository filesystem. It collects the most recent experience
reports and messages, then wraps the content with a short header and
footer. The digest is truncated to fit within a single prompt.
"""

import argparse
import os
from typing import List
# --- END HEADER ---

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experience_reports')
INBOX_DIR = os.path.join(os.path.dirname(__file__), 'inbox')
OUTBOX_DIR = os.path.join(os.path.dirname(__file__), 'outbox')

HEADER = "# SPEAKTOME DIGEST"
LENGTH_LIMIT = 2000  # character limit for digest output


def _recent_files(directory: str, limit: int) -> List[str]:
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.endswith('.md') and not f.startswith('template')]
    files.sort()
    return files[-limit:]


def build_digest(reports: int = 3, messages: int = 3) -> str:
    parts: List[str] = []
    for path in _recent_files(REPORTS_DIR, reports):
        parts.append(f"## {os.path.basename(path)}")
        with open(path, 'r', encoding='utf-8') as f:
            parts.append(f.read().strip())
    for box in (INBOX_DIR, OUTBOX_DIR):
        for path in _recent_files(box, messages):
            parts.append(f"## {os.path.basename(path)}")
            with open(path, 'r', encoding='utf-8') as f:
                parts.append(f.read().strip())
    body = '\n\n'.join(parts)
    available = LENGTH_LIMIT - len(HEADER) * 2 - 4
    if len(body) > available:
        body = body[:available]
    return f"{HEADER}\n\n{body}\n\n{HEADER}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a short digest")
    parser.add_argument('-r', '--reports', type=int, default=3,
                        help='number of reports to include')
    parser.add_argument('-m', '--messages', type=int, default=3,
                        help='number of messages from each box')
    args = parser.parse_args()
    print(build_digest(args.reports, args.messages))


if __name__ == '__main__':
    main()
