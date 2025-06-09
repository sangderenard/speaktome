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

    try:
        from .jax_backend import JAXTensorOperations
    except Exception:  # pragma: no cover - optional backend
        JAXTensorOperations = None  # type: ignore
    try:
        from .c_backend import CTensorOperations
    except Exception:  # pragma: no cover - optional backend
        CTensorOperations = None  # type: ignore

except Exception:
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---


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
