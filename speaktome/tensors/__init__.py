#!/usr/bin/env python3
"""Tensor backends and abstraction layer."""
from __future__ import annotations

try:
    from .abstraction import (
        AbstractTensorOperations,
        get_tensor_operations,
    )
    from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV, detect_faculty
    from .torch_backend import PyTorchTensorOperations
    from .numpy_backend import NumPyTensorOperations
    from .pure_backend import PurePythonTensorOperations
    from .jax_backend import JAXTensorOperations
    from .c_backend import CTensorOperations
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

JAXTensorOperations = locals().get("JAXTensorOperations", None)  # pragma: no cover
CTensorOperations = locals().get("CTensorOperations", None)  # pragma: no cover

__all__ = [
    "AbstractTensorOperations",
    "get_tensor_operations",
    "PyTorchTensorOperations",
    "NumPyTensorOperations",
    "PurePythonTensorOperations",
    "Faculty",
    "DEFAULT_FACULTY",
    "FORCE_ENV",
    "detect_faculty",
]

if JAXTensorOperations is not None:
    __all__.append("JAXTensorOperations")

if CTensorOperations is not None:
    __all__.append("CTensorOperations")
