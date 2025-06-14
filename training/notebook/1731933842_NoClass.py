#!/usr/bin/env python3
"""Legacy packaging example using a pyproject-based workflow."""
from __future__ import annotations

try:
    from AGENTS.tools.headers.header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print("This training file requires repository utilities.")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

# ########## STUB: pyproject_packaging_example ##########
# PURPOSE: Demonstrates how this training notebook would be packaged using
# a modern ``pyproject.toml`` configuration instead of setuptools.
# EXPECTED BEHAVIOR: When implemented, this file would generate a minimal
# ``pyproject.toml`` and build a distributable
# package.
# INPUTS: None for now.
# OUTPUTS: No runtime behavior; serves as documentation for developers.
# KEY ASSUMPTIONS/DEPENDENCIES: Packaging is managed through ``pyproject.toml``.
# TODO:
#   - Provide a concrete ``pyproject.toml`` example for the Alpha project.
#   - Integrate with automated training utilities if needed.
# NOTES: This file intentionally raises ``NotImplementedError`` to signal
# that packaging steps are yet to be defined.
# ###########################################################################

raise NotImplementedError(
    "Packaging for the Alpha training example now relies on a pyproject.toml "
    "instead of setuptools."
)
