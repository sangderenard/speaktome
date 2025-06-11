#!/usr/bin/env python3
"""CGPT stub test."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import importlib

    from tensors.models.c_gpt import cgpt
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def test_cgpt_forward_placeholder():
    model = cgpt.CGPT()
    out = model.forward([1.0, 2.0, 3.0])
    assert out == [0.0, 0.0, 0.0]
