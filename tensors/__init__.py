"""Tensor backends and abstraction layer."""
from __future__ import annotations

DEBUG = False
# Only import abstraction and utilities that do not import backends.
try:
    from .abstraction import (
        AbstractTensor,
        get_tensor_operations,
    )
    from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV, detect_faculty
except Exception as e:
    import sys
    import traceback
    print("Failed to import core tensor abstraction or faculty utilities")
    print(f"Exception: {e}")
    traceback.print_exc()
    sys.exit(1)

# Concrete backends are optional and imported gracefully -- speaktome's
# consumer code (speaktome/core/*.py, tests) expects these names at the
# package root, so we surface them here even though this module otherwise
# favors lazy/registry-based backend loading. A backend that fails to
# import (missing torch/numpy/jax, or an unbuilt native extension)
# degrades to ``None`` rather than aborting the process.
PyTorchTensorOperations = None
try:  # optional torch backend
    from .torch_backend import PyTorchTensorOperations  # type: ignore
except Exception:  # pragma: no cover - torch missing
    PyTorchTensorOperations = None  # type: ignore

NumPyTensorOperations = None
try:  # optional numpy backend
    from .numpy_backend import NumPyTensorOperations  # type: ignore
except Exception:  # pragma: no cover - numpy missing
    NumPyTensorOperations = None  # type: ignore

JAXTensorOperations = None
try:  # optional jax backend
    from .jax_backend import JAXTensorOperations  # type: ignore
except Exception:  # pragma: no cover - jax missing
    JAXTensorOperations = None  # type: ignore

from .pure_backend import PurePythonTensorOperations

CTensorOperations = None
try:  # optional C backend
    from .accelerator_backends.c_backend import CTensorOperations  # type: ignore
except Exception:  # pragma: no cover - c backend missing
    CTensorOperations = None  # type: ignore

OpenGLTensorOperations = None
try:  # optional OpenGL backend
    from .accelerator_backends.opengl_backend import OpenGLTensorOperations  # type: ignore
except Exception:  # pragma: no cover - opengl missing
    OpenGLTensorOperations = None  # type: ignore

RustTensorOperations = None
try:  # optional Rust backend
    from .accelerator_backends.rust_backend import RustTensorOperations  # type: ignore
except Exception:  # pragma: no cover - rust missing
    RustTensorOperations = None  # type: ignore

AcceleratorCoordinator = None
try:
    from .accelerator_backends.coordinator import AcceleratorCoordinator  # type: ignore
except Exception:  # pragma: no cover - missing dependencies
    AcceleratorCoordinator = None  # type: ignore

__all__ = [
    "AbstractTensor",
    "get_tensor_operations",
    "Faculty",
    "DEFAULT_FACULTY",
    "FORCE_ENV",
    "detect_faculty",
    "PyTorchTensorOperations",
    "NumPyTensorOperations",
    "PurePythonTensorOperations",
    "JAXTensorOperations",
    "CTensorOperations",
    "OpenGLTensorOperations",
    "RustTensorOperations",
    "AcceleratorCoordinator",
]
