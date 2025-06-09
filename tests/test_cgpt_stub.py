#!/usr/bin/env python3
"""CGPT stub test."""
from __future__ import annotations

try:
    import importlib

    from speaktome.tensors.models.c_gpt import cgpt
except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
# --- END HEADER ---


def test_cgpt_forward_placeholder():
    model = cgpt.CGPT()
    out = model.forward([1.0, 2.0, 3.0])
    assert out == [0.0, 0.0, 0.0]
