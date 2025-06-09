#!/usr/bin/env python3
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import your_modules
    import sys
except Exception:
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
