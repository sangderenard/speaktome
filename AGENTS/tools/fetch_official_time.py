#!/usr/bin/env python3
"""Fetch official UTC time from a trusted source and optionally set system time.

This script contacts ``worldtimeapi.org`` to retrieve the current UTC timestamp.
The timestamp can be written to a file or printed. With ``--apply`` the script
attempts to set the local system clock using ``sudo date`` on POSIX systems or
``Set-Date`` on Windows. Use with caution and appropriate privileges.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.request
# --- END HEADER ---


END_MARKER = "# --- END HEADER ---"


def fetch_utc_time() -> str:
    """Return the current UTC time string from worldtimeapi.org."""
    url = "https://worldtimeapi.org/api/timezone/Etc/UTC"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    return data["utc_datetime"]


def apply_time(utc_str: str) -> None:
    """Attempt to set the local system time to ``utc_str``."""
    if platform.system() == "Windows":
        cmd = ["powershell", "-Command", f"Set-Date -Date '{utc_str}'"]
    else:
        cmd = ["sudo", "date", "-u", "--set", utc_str]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="set the system clock")
    parser.add_argument("--outfile", type=str, help="write time to this file")
    args = parser.parse_args(argv)

    utc_time = fetch_utc_time()

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as fh:
            fh.write(utc_time + "\n")
    else:
        print(utc_time)

    if args.apply:
        apply_time(utc_time)


if __name__ == "__main__":
    main()
