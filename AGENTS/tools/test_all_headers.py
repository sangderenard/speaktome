#!/usr/bin/env python3
"""Run all class ``test()`` methods across faculty tiers."""

from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from tensors.faculty import Faculty, FORCE_ENV
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


# ########## STUB: recursive test runner ##########
# PURPOSE: Execute class level ``test()`` methods under each faculty tier.
# EXPECTED BEHAVIOR: Import each class in a subprocess with the environment
# variable ``SPEAKTOME_FACULTY`` set and call ``Class.test()``.
# INPUTS: JSON data produced by ``dump_headers.py`` or scanned directly.
# OUTPUTS: JSON summary of pass/fail results for optional markdown reporting.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires Python interpreter access and the
# ``speaktome`` package importable in subprocesses.
# TODO:
#   - Capture stdout/stderr from tests for logging. [DONE]
#   - Surface structured results for ``format_test_digest.py``. [DONE]
# NOTES: This is a minimal scaffold and does not implement all failure modes.
# ###########################################################################

SCRIPT_PATH = Path(__file__).with_name("dump_headers.py")


def load_headers() -> dict:
    data = subprocess.check_output([sys.executable, str(SCRIPT_PATH)])
    return json.loads(data)


def run_test(mod: str, cls: str, faculty: Faculty) -> dict[str, object]:
    """Return structured results for ``cls.test`` under ``faculty``."""
    env = os.environ.copy()
    env[FORCE_ENV] = faculty.name
    code = (
        "import importlib, sys; "
        f"mod = importlib.import_module('{mod}'); "
        f"cls = getattr(mod, '{cls}'); "
        "getattr(cls, 'test', lambda: None)();"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    headers = load_headers()
    results = {}
    for mod, items in headers.items():
        for item in items:
            cls = item["class"]
            key = f"{mod}.{cls}"
            results[key] = {}
            for fac in Faculty:
                results[key][fac.name] = run_test(mod, cls, fac)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
