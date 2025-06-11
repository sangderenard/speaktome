#!/usr/bin/env python3
"""Abstraction layer for tensor operations."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from abc import ABC, abstractmethod
    from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque
    import math
    import time
    from collections import deque

    from .faculty import Faculty, DEFAULT_FACULTY
    import speaktome.config as config
    import torch
    import numpy as np
except ModuleNotFoundError:
    torch = None  # type: ignore
    np = None  # type: ignore
except Exception:
    print("Failed to import required modules for tensor operations.")
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

CONVERSION_REGISTRY: Dict[Tuple[type, type], Callable[["AbstractTensor", Any, "AbstractTensor"], Any]] = {}

OPS_CACHE: Dict[type, "AbstractTensor"] = {}

def register_conversion(src_cls: type, tgt_cls: type, func: Callable[["AbstractTensor", Any, "AbstractTensor"], Any]) -> None:
    """Register a direct tensor conversion function."""
    CONVERSION_REGISTRY[(src_cls, tgt_cls)] = func

def _get_ops_for_class(cls: type) -> "AbstractTensor":
    if cls in OPS_CACHE:
        return OPS_CACHE[cls]
    if cls.__name__.startswith("PyTorch"):
        ops = get_tensor_operations(Faculty.TORCH)
    elif cls.__name__.startswith("NumPy"):
        ops = get_tensor_operations(Faculty.NUMPY)
    elif cls.__name__.startswith("PurePython"):
        ops = get_tensor_operations(Faculty.PURE_PYTHON)
    elif cls.__name__.startswith("JAX"):
        ops = get_tensor_operations(Faculty.NUMPY)  # approximate
    else:
        ops = get_tensor_operations()
    OPS_CACHE[cls] = ops
    return ops

def _find_conversion_path(src_cls: type, tgt_cls: type) -> List[Tuple[type, type]]:
    if src_cls == tgt_cls:
        return []
    q: Deque[Tuple[type, List[Tuple[type, type]]]] = deque([(src_cls, [])])
    seen = {src_cls}
    while q:
        cur, path = q.popleft()
        if cur == tgt_cls:
            return path
        for (a, b), _ in CONVERSION_REGISTRY.items():
            if a == cur and b not in seen:
                q.append((b, path + [(a, b)]))
                seen.add(b)
    return []

class AbstractTensor(ABC):
    def __init__(self, track_time: bool = False) -> None:
        """Optional benchmark support for tensor operations."""
        self.track_time = track_time
        self.last_op_time: float | None = None
        self.data = None  # Holds the tensor's data

    def benchmark(self, call: "Callable[[], Any]") -> Any:
        """Run ``call`` and store elapsed time if benchmarking is enabled."""
        if self.track_time:
            start = time.process_time()
            result = call()
            self.last_op_time = time.process_time() - start
            return result
        return call()

    # --- Tensor creation and manipulation methods ---
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any = None, device: Any = None):
        result = type(self)(track_time=self.track_time)
        result.data = self.full_(size, fill_value, dtype, device)
        return result

    def zeros(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        result = type(self)(track_time=self.track_time)
        result.data = self.zeros_(size, dtype, device)
        return result

    def clone(self, tensor: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.clone_(tensor)
        return result

    def to_device(self, tensor: Any = None, device: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.to_device_(tensor, device)
        return result

    def get_device(self, tensor: Any = None) -> Any:
        tensor = self.data_or(tensor)
        return self.get_device_(tensor)

    def get_dtype(self, tensor: Any = None) -> Any:
        tensor = self.data_or(tensor)
        return self.get_dtype_(tensor)

    def item(self, tensor: Any = None) -> Union[int, float, bool]:
        tensor = self.data_or(tensor)
        return self.item_(tensor)

    def max(self, tensor: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.max_(tensor)
        return result

    def long_cast(self, tensor: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.long_cast_(tensor)
        return result

    def not_equal(self, tensor1: Any = None, tensor2: Any = None) -> "AbstractTensor":
        t1 = self.ensure_tensor(self.data_or(tensor1))
        t2 = self.ensure_tensor(self.data_or(tensor2))
        result = type(self)(track_time=self.track_time)
        result.data = self.not_equal_(t1, t2)
        return result

    def arange(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.arange_(start, end, step, device, dtype)
        return result

    def select_by_indices(self, tensor: Any = None, indices_dim0: Any = None, indices_dim1: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.select_by_indices_(tensor, indices_dim0, indices_dim1)
        return result

    def log_softmax(self, tensor: Any = None, dim: int = -1) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.log_softmax_(tensor, dim)
        return result

    def pad(self, tensor: Any = None, pad: Tuple[int, ...] = (0, 0), value: float = 0) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.pad_(tensor, pad, value)
        return result

    def cat(self, tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        tensors = [self.ensure_tensor(t) for t in tensors]
        result = type(self)(track_time=self.track_time)
        result.data = self.cat_(tensors, dim)
        return result

    def topk(self, tensor: Any = None, k: int = 1, dim: int = -1) -> Tuple["AbstractTensor", Any]:
        tensor = self.ensure_tensor(self.data_or(tensor))
        values, idxs = self.topk_(tensor, k, dim)
        result = type(self)(track_time=self.track_time)
        result.data = values
        return result, idxs

    def stack(self, tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        tensors = [self.ensure_tensor(t) for t in tensors]
        result = type(self)(track_time=self.track_time)
        result.data = self.stack_(tensors, dim)
        return result

    def repeat_interleave(self, tensor: Any = None, repeats: int = 1, dim: Optional[int] = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.repeat_interleave_(tensor, repeats, dim)
        return result

    def view_flat(self, tensor: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.view_flat_(tensor)
        return result

    def assign_at_indices(self, tensor_to_modify: Any = None, indices_dim0: Any = None, indices_dim1: Any = None, values_to_assign: Any = None) -> "AbstractTensor":
        tensor_to_modify = self.ensure_tensor(self.data_or(tensor_to_modify))
        result = type(self)(track_time=self.track_time)
        result.data = self.assign_at_indices_(tensor_to_modify, indices_dim0, indices_dim1, values_to_assign)
        return result

    def increment_at_indices(self, tensor_to_modify: Any = None, mask: Any = None) -> "AbstractTensor":
        tensor_to_modify = self.ensure_tensor(self.data_or(tensor_to_modify))
        result = type(self)(track_time=self.track_time)
        result.data = self.increment_at_indices_(tensor_to_modify, mask)
        return result

    def clamp(self, tensor: Any = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.clamp_(tensor, min_val, max_val)
        return result

    def shape(self, tensor: Any = None) -> Tuple[int, ...]:
        tensor = self.data_or(tensor)
        return self.shape_(tensor)

    def numel(self, tensor: Any = None) -> int:
        tensor = self.data_or(tensor)
        return self.numel_(tensor)

    def mean(self, tensor: Any = None, dim: Optional[int] = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.mean_(tensor, dim)
        return result

    def pow(self, tensor: Any = None, exponent: float = 1.0) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.pow_(tensor, exponent)
        return result

    def sqrt(self, tensor: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.sqrt_(tensor)
        return result

    def tensor_from_list(self, data: List[Any], dtype: Any = None, device: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.tensor_from_list_(data, dtype, device)
        return result

    def boolean_mask_select(self, tensor: Any = None, mask: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.boolean_mask_select_(tensor, mask)
        return result

    def tolist(self, tensor: Any = None) -> List[Any]:
        tensor = self.ensure_tensor(self.data_or(tensor))
        return self.tolist_(tensor)

    def less(self, tensor: Any = None, value: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.less_(tensor, value)
        return result

    def index_select(self, tensor: Any = None, dim: int = 0, indices: Any = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.index_select_(tensor, dim, indices)
        return result

    def argmin(self, tensor: Any = None, dim: Optional[int] = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.argmin_(tensor, dim)
        return result

    def interpolate(self, tensor: Any = None, size: Tuple[int, ...] = None) -> "AbstractTensor":
        tensor = self.ensure_tensor(self.data_or(tensor))
        result = type(self)(track_time=self.track_time)
        result.data = self.interpolate_(tensor, size)
        return result

    def save(self, tensor: Any = None, filepath: str = None) -> None:
        tensor = self.ensure_tensor(self.data_or(tensor))
        self.save_(tensor, filepath)

    def load(self, filepath: str, dtype: Any = None, device: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.load_(filepath, dtype, device)
        return result

    def to_dtype(self, dtype: str = "float") -> "AbstractTensor":
        """Convert self.data to the specified dtype using the backend's to_dtype_ method."""
        result = type(self)(track_time=self.track_time)
        result.data = self.to_dtype_(self.data, dtype)
        return result

    # --- Dtype helpers ---
    @property
    def long_dtype(self) -> Any:
        return self.long_dtype_

    @property
    def bool_dtype(self) -> Any:
        return self.bool_dtype_

    @property
    def float_dtype(self) -> Any:
        return self.float_dtype_

    @property
    def tensor_type(self) -> type:
        return self.tensor_type_

    # Lightweight helper to coerce arbitrary input to this backend's tensor type
    def to_backend(self, target_ops: "AbstractTensor") -> "AbstractTensor":
        """Convert this tensor to the target backend, returning a new AbstractTensor instance with .data set."""
        if type(self) is type(target_ops):
            # Same backend: clone or return self
            new_tensor = type(self)()
            new_tensor.data = self.clone(self.data)
            return new_tensor
        # Find conversion function
        conv_func = CONVERSION_REGISTRY.get((type(self), type(target_ops)), None)
        if conv_func is None:

            # Fallback to default conversion
            converted_data = default_to_backend(self, self.data, target_ops)
        else:
            converted_data = conv_func(self, self.data, target_ops)
        new_tensor = type(target_ops)()
        new_tensor.data = converted_data
        return new_tensor

    def ensure_tensor(self, tensor: Any) -> Any:
        if tensor is None:
            raise ValueError("ensure_tensor called with tensor=None")
        if isinstance(tensor, self.tensor_type):
            return tensor
        # If tensor is an AbstractTensor instance, convert using to_backend
        if isinstance(tensor, AbstractTensor):
            return tensor.to_backend(self).data
        if torch is not None and isinstance(tensor, torch.Tensor):
            torch_ops = get_tensor_operations(Faculty.TORCH)
            return torch_ops.__class__().to_backend(self.__class__()).data if not isinstance(self, torch_ops.__class__) else tensor
        if np is not None and isinstance(tensor, np.ndarray):
            numpy_ops = get_tensor_operations(Faculty.NUMPY)
            numpy_tensor = numpy_ops.__class__()  # create a new NumPyTensorOperations instance
            numpy_tensor.data = tensor
            return numpy_tensor.to_backend(self).data
        if isinstance(tensor, list):
            return self.tensor_from_list(tensor, dtype=None, device=None).data
        if hasattr(tensor, "tolist"):
            return self.tensor_from_list(tensor.tolist(), dtype=None, device=None).data
        return self.tensor_from_list([tensor], dtype=None, device=None).data

    # --- Operator routing ---
    def __apply_operator(self, op: str, left: Any, right: Any):
        """Apply ``op`` to ``left`` and ``right`` returning a new tensor."""
        l = self.ensure_tensor(self.data_or(left))
        r = self.ensure_tensor(self.data_or(right))
        result = type(self)(track_time=self.track_time)
        result.data = self.__apply_operator_(op, l, r)
        return result

    # legacy entry point should be inaccessible
    def _apply_operator(self, *args, **kwargs):  # pragma: no cover - convenience
        raise AttributeError("_apply_operator is internal")

    def __add__(self, other):
        return self.__apply_operator('add', self, other)

    def __sub__(self, other):
        return self.__apply_operator('sub', self, other)

    def __mul__(self, other):
        return self.__apply_operator('mul', self, other)

    def __truediv__(self, other):
        return self.__apply_operator('truediv', self, other)

    def __floordiv__(self, other):
        return self.__apply_operator('floordiv', self, other)

    def __mod__(self, other):
        return self.__apply_operator('mod', self, other)

    def __pow__(self, other):
        return self.__apply_operator('pow', self, other)

    def __matmul__(self, other):
        return self.__apply_operator('matmul', self, other)

    # Reverse operators
    def __radd__(self, other):
        return self.__apply_operator('radd', other, self)

    def __rsub__(self, other):
        return self.__apply_operator('rsub', other, self)

    def __rmul__(self, other):
        return self.__apply_operator('rmul', other, self)

    def __rtruediv__(self, other):
        return self.__apply_operator('rtruediv', other, self)

    def __rfloordiv__(self, other):
        return self.__apply_operator('rfloordiv', other, self)

    def __rmod__(self, other):
        return self.__apply_operator('rmod', other, self)

    def __rpow__(self, other):
        return self.__apply_operator('rpow', other, self)

    def __rmatmul__(self, other):
        return self.__apply_operator('rmatmul', other, self)

    # In-place operators
    def __iadd__(self, other):
        return self.__apply_operator('iadd', self, other)

    def __isub__(self, other):
        return self.__apply_operator('isub', self, other)

    def __imul__(self, other):
        return self.__apply_operator('imul', self, other)

    def __itruediv__(self, other):
        return self.__apply_operator('itruediv', self, other)

    def __ifloordiv__(self, other):
        return self.__apply_operator('ifloordiv', self, other)

    def __imod__(self, other):
        return self.__apply_operator('imod', self, other)

    def __ipow__(self, other):
        return self.__apply_operator('ipow', self, other)

    def __imatmul__(self, other):
        return self.__apply_operator('imatmul', self, other)

    def data_or(self, obj: Any = None) -> Any:
        """Return self.data if no argument is passed, otherwise return the argument unchanged."""
        if obj is None:
            return self.data
        return obj

# Remove stray demo/test code (stacked = ..., values, idxs = ...)

def _get_shape(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])


def _flatten(data):
    if not isinstance(data, list):
        return [data]
    return [item for sublist in data for item in _flatten(sublist)]


def default_to_backend(
    source_ops: "AbstractTensor", tensor: Any, target_ops: "AbstractTensor"
) -> Any:
    """Fallback conversion using :meth:`tolist` and :meth:`tensor_from_list`."""
    if type(source_ops) is type(target_ops):
        return source_ops.clone(tensor)
    data = source_ops.tolist(tensor)
    dtype = None
    device = None
    try:
        dtype = source_ops.get_dtype(tensor)
    except Exception:
        pass
    try:
        device = source_ops.get_device(tensor)
    except Exception:
        pass
    return target_ops.tensor_from_list(data, dtype=dtype, device=device)


def get_tensor_operations(faculty: Faculty | None = None, *, track_time: bool = False) -> AbstractTensor:
    """Return a tensor operations backend based on the faculty tier."""

    faculty = faculty or DEFAULT_FACULTY
    if faculty in (Faculty.TORCH, Faculty.PYGEO):
        from .torch_backend import PyTorchTensorOperations
        return PyTorchTensorOperations(default_device=config.DEVICE, track_time=track_time)
    if faculty is Faculty.NUMPY and np is not None:
        from .numpy_backend import NumPyTensorOperations
        return NumPyTensorOperations(track_time=track_time)
    if faculty is Faculty.CTENSOR:
        from .c_backend import CTensorOperations
        return CTensorOperations(track_time=track_time)
    from .pure_backend import PurePythonTensorOperations
    return PurePythonTensorOperations(track_time=track_time)


# --- Conversion registration ---
try:
    from .torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - optional backend
    PyTorchTensorOperations = None  # type: ignore
try:
    from .numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional backend
    NumPyTensorOperations = None  # type: ignore
from .pure_backend import PurePythonTensorOperations

if np is not None and NumPyTensorOperations is not None:
    register_conversion(NumPyTensorOperations, PurePythonTensorOperations,
                        PurePythonTensorOperations.from_numpy)
    register_conversion(PurePythonTensorOperations, NumPyTensorOperations,
                        NumPyTensorOperations.from_pure)

if torch is not None and PyTorchTensorOperations is not None and NumPyTensorOperations is not None:
    register_conversion(PyTorchTensorOperations, NumPyTensorOperations,
                        NumPyTensorOperations.from_torch)
    register_conversion(NumPyTensorOperations, PyTorchTensorOperations,
                        PyTorchTensorOperations.from_numpy)
    register_conversion(PyTorchTensorOperations, PurePythonTensorOperations,
                        PurePythonTensorOperations.from_torch)
    register_conversion(PurePythonTensorOperations, PyTorchTensorOperations,
                        PyTorchTensorOperations.from_pure)

try:
    from .jax_backend import JAXTensorOperations
except Exception:  # pragma: no cover - optional backend
    JAXTensorOperations = None  # type: ignore

if (
    np is not None
    and NumPyTensorOperations is not None
    and JAXTensorOperations is not None
):
    register_conversion(JAXTensorOperations, NumPyTensorOperations,
                        NumPyTensorOperations.from_jax)
    register_conversion(NumPyTensorOperations, JAXTensorOperations,
                        JAXTensorOperations.from_numpy)

if (
    torch is not None
    and PyTorchTensorOperations is not None
    and JAXTensorOperations is not None
):
    register_conversion(PyTorchTensorOperations, JAXTensorOperations,
                        JAXTensorOperations.from_torch)
    register_conversion(JAXTensorOperations, PyTorchTensorOperations,
                        PyTorchTensorOperations.from_jax)

if JAXTensorOperations is not None:
    register_conversion(JAXTensorOperations, PurePythonTensorOperations,
                        PurePythonTensorOperations.from_jax)
    register_conversion(PurePythonTensorOperations, JAXTensorOperations,
                        JAXTensorOperations.from_pure)


