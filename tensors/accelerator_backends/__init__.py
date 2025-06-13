#!/usr/bin/env python3
"""Accelerator-specific tensor backends."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from .c_backend import CTensorOperations
    from .opengl_backend import OpenGLTensorOperations
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = ["CTensorOperations", "OpenGLTensorOperations"]
