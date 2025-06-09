"""Tensor backends and abstraction layer."""

from .abstraction import (
    AbstractTensorOperations,
    get_tensor_operations,
)
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

__all__ = [
    "AbstractTensorOperations",
    "get_tensor_operations",
    "PyTorchTensorOperations",
    "NumPyTensorOperations",
    "PurePythonTensorOperations",
]

if JAXTensorOperations is not None:
    __all__.append("JAXTensorOperations")

if CTensorOperations is not None:
    __all__.append("CTensorOperations")
