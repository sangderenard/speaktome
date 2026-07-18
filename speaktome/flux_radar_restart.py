#!/usr/bin/env python3
"""Force-kill whatever's listening on the flux-radar port and (by default) restart it.

The blunt instrument for "the server is stuck." A request truly wedged
inside a synchronous model call -- a stalled CUDA kernel, say -- is a
live thread this process can neither signal, cancel, nor interrupt;
Python simply has no mechanism for that. POST /api/reset (see
flux_radar_server.py) helps for the common non-hung case (clears cached
engines so *new* requests get fresh objects), but it cannot rescue an
already-wedged one. The only fully reliable fix at that point is killing
the OS process outright, which is what this does.

Safe to run at any time: settings and live-session progress are
autosaved continuously (every single tick, not just on a graceful
shutdown -- see flux_radar_server.py's _autosave), and a resumed live
session picks its graph back up from exactly where it left off. A hard
kill loses at most the one in-flight tick.

Usage:
    python -m speaktome.flux_radar_restart [--port 8877] [--no-restart] [-- --engine gpt2 --preload]
Anything after a literal "--" is passed through to
`python -m speaktome.flux_radar_server` when restarting.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from typing import List
# --- END HEADER ---


def find_pids_on_port(port: int) -> List[int]:
    """Best-effort PID lookup for whatever's listening on ``port``.

    Matches the port number exactly (the text after the *last* colon in
    the local-address column) -- a naive ``f":{port}" in line`` substring
    check is a real bug here: IPv6 loopback addresses like ``[::1]:1900``
    contain ":1" as a literal substring, so port 1 would falsely match
    nearly every IPv6 entry on the machine.
    """
    system = platform.system()
    if system == "Windows":
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, check=False
        ).stdout
        pids = set()
        target = str(port)
        for line in out.splitlines():
            parts = line.split()
            # "  TCP    127.0.0.1:8877    0.0.0.0:0    LISTENING    12345"
            if len(parts) < 5 or not parts[-1].isdigit():
                continue
            local_addr = parts[1]
            if local_addr.rsplit(":", 1)[-1] == target:
                pids.add(int(parts[-1]))
        return sorted(pids)
    # POSIX: lsof is the most broadly available option for this without
    # adding a dependency.
    out = subprocess.run(
        ["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False
    ).stdout
    return sorted({int(p) for p in out.split() if p.isdigit()})


def kill_pid(pid: int) -> None:
    system = platform.system()
    if system == "Windows":
        # PowerShell's Stop-Process, not the legacy taskkill.exe -- in at
        # least one real sandboxed environment taskkill reported "ERROR:
        # Not found" for a PID that Get-Process/Stop-Process could see
        # and kill just fine (a session-visibility quirk of taskkill
        # itself, not a real absence of the process).
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
            check=False,
        )
    else:
        subprocess.run(["kill", "-9", str(pid)], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument(
        "--no-restart", action="store_true",
        help="only kill whatever's on the port, don't start a new server",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    pids = find_pids_on_port(args.port)
    if not pids:
        print(f"[flux-radar-restart] nothing found listening on port {args.port}.")
    else:
        for pid in pids:
            print(f"[flux-radar-restart] killing PID {pid} (port {args.port})...")
            kill_pid(pid)
        time.sleep(1)

    if args.no_restart:
        return

    cmd = [sys.executable, "-m", "speaktome.flux_radar_server", "--port", str(args.port), *passthrough]
    print(f"[flux-radar-restart] starting: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    print("[flux-radar-restart] new server launching in the background.")


if __name__ == "__main__":
    main()
