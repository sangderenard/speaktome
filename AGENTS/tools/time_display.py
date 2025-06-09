#!/usr/bin/env python3
"""Display current adjusted time using :mod:`time_sync`."""
from __future__ import annotations

import argparse

from time_sync import sync_offset, now, compose_ascii_digits, print_analog_clock

# --- END HEADER ---


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    if args.update:
        sync_offset()

    current = now()

    if args.analog:
        print_analog_clock(current)
    elif args.ascii:
        print(compose_ascii_digits(current.strftime("%H:%M:%S")))
    else:
        print(current.isoformat())


if __name__ == "__main__":
    main()
