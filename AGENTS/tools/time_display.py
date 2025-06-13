#!/usr/bin/env python3
"""Display current adjusted time using :mod:`timesync`."""
from __future__ import annotations

import argparse

from timesync import (
    sync_offset, now,
    compose_ascii_digits, print_analog_clock, print_digital_clock, # Added print_digital_clock for consistency
    init_colorama_for_windows
)

# --- END HEADER ---


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-color-fix", action="store_true", help="skip fixing windows console for color"
    )
    parser.add_argument(
        "--update", action="store_true", help="sync offset with internet time"
    )
    parser.add_argument(
        "--ascii", action="store_true", help="print a digital ASCII clock"
    )
    parser.add_argument(
        "--analog", action="store_true", help="print an analog ASCII clock (stub)"
    )
    args = parser.parse_args(argv)

    if not args.no_color_fix:
        init_colorama_for_windows()

    if args.update:
        sync_offset()

    current = now()

    if args.analog:
        print_analog_clock(current)
    elif args.ascii:
        # Using print_digital_clock for consistency with demo,
        # it internally calls compose_ascii_digits.
        # If you specifically want the raw compose_ascii_digits output, revert this part.
        print_digital_clock(current)
    else:
        print(current.isoformat())


if __name__ == "__main__":
    main()
