"""Demonstrate time synchronization with analog and digital clocks."""

from __future__ import annotations

import datetime as _dt
import time

from colorama import Cursor, ansi, just_fix_windows_console

from time_sync import get_offset, sync_offset, print_analog_clock, print_digital_clock

# --- END HEADER ---


def _clear() -> None:
    """Clear the console and move the cursor to the top."""

    print(ansi.clear_screen() + Cursor.POS(0, 0), end="")


def main() -> None:
    """Run the stopwatch demo until interrupted."""

    just_fix_windows_console()
    sync_offset()
    start = time.perf_counter()
    try:
        while True:
            elapsed = time.perf_counter() - start
            offset = get_offset()
            system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
            internet = system + _dt.timedelta(seconds=offset)

            h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
            m, rem = divmod(rem, 60 * 1000)
            s, ms = divmod(rem, 1000)
            stopwatch = f"{h:02}:{m:02}:{s:02}.{ms:03}"

            _clear()
            print_analog_clock(internet)
            print()
            print_digital_clock(system)
            print()
            print_digital_clock(internet)
            print()
            print(f"Stopwatch: {stopwatch}")
            print(f"Offset: {_dt.timedelta(seconds=offset)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        _clear()
        print("Demo stopped.")
        print(f"Final system time: {system.strftime('%H:%M:%S')}")
        print(f"Final internet time: {internet.strftime('%H:%M:%S')}")
        print(f"Offset: {_dt.timedelta(seconds=offset)}")


if __name__ == "__main__":
    main()
