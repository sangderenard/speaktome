#!/usr/bin/env python3
"""Utility to obtain the earliest and latest epoch timestamps among reports."""
from __future__ import annotations

try:
    import os
    import re
    from datetime import datetime
    import argparse
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experience_reports')
ARCHIVE_DIR = os.path.join(REPORTS_DIR, 'archive')
PATTERN = re.compile(r"^(\d{10})_")


def gather_epochs() -> list[int]:
    """Collect all epoch timestamps from report filenames."""
    epochs: list[int] = []
    for base in (REPORTS_DIR, ARCHIVE_DIR):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            m = PATTERN.match(name)
            if m:
                try:
                    epochs.append(int(m.group(1)))
                except ValueError:
                    pass
    return epochs


def get_date_range() -> tuple[int, int] | None:
    """Return the earliest and latest epoch timestamps or ``None`` if no reports."""
    epochs = gather_epochs()
    if not epochs:
        return None
    return min(epochs), max(epochs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", action="store_true", help="print dates in ISO format")
    args = parser.parse_args(argv)

    range_pair = get_date_range()
    if range_pair is None:
        print("no reports found")
        return 1
    start, end = range_pair
    if args.iso:
        start_str = datetime.fromtimestamp(start).isoformat()
        end_str = datetime.fromtimestamp(end).isoformat()
    else:
        start_str = str(start)
        end_str = str(end)
    print(f"{start_str} -> {end_str}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
