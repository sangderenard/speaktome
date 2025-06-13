#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Pseudocode for detecting the running agent environment."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

# ########## STUB: detect_agent_environment ##########
# PURPOSE: Determine if the current Python process is executed within a
#          configured SPEAKTOME agent environment.
# EXPECTED BEHAVIOR: Inspect environment variables or presence of a custom
#          flag file to ascertain whether necessary packages are already
#          installed. When uncertain, return False so calling code may
#          trigger setup scripts.
# INPUTS: Optional path to a flag file or dictionary of environment variables.
# OUTPUTS: Boolean indicating agent environment detection result.
# KEY ASSUMPTIONS/DEPENDENCIES: `setup_env_dev` may export unique variables
#          such as SPEAKTOME_ENV, or create a sentinel file.
# TODO:
#   - Agree on standardized environment variable or flag naming.
#   - Implement detection logic using `os.environ` and filesystem checks.
#   - Add unit tests covering positive and negative scenarios.
# NOTES: Placeholder pseudocode only. Actual implementation depends on how
#        development environments mark successful setup.
# ###########################################################################

import json
import os
from pathlib import Path


def detect_agent_environment(flag_path: str | None = None) -> bool:
    """Return True if running inside a configured SPEAKTOME agent environment."""
    if flag_path and Path(flag_path).is_file():
        return True

    env_dir = Path(os.environ.get("VIRTUAL_ENV", "./.venv")).resolve()
    marker = env_dir / "pytest_enabled"
    active = Path(os.environ.get("SPEAKTOME_ACTIVE_FILE", "/tmp/speaktome_active.json"))

    if not (marker.exists() and active.exists()):
        return False

    try:
        data = json.loads(active.read_text())
    except Exception:
        return False

    return bool(data.get("codebases"))


if __name__ == "__main__":
    print("Agent environment detected?", detect_agent_environment())
