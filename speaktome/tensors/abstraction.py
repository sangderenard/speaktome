from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque
import math
import time
from collections import deque

from .faculty import Faculty, DEFAULT_FACULTY
from .. import config
# --- END HEADER ---
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

CONVERSION_REGISTRY: Dict[Tuple[type, type], Callable[["AbstractTensorOperations", Any, "AbstractTensorOperations"], Any]] = {}

OPS_CACHE: Dict[type, "AbstractTensorOperations"] = {}

def register_conversion(src_cls: type, tgt_cls: type, func: Callable[["AbstractTensorOperations", Any, "AbstractTensorOperations"], Any]) -> None:
    """Register a direct tensor conversion function."""
    CONVERSION_REGISTRY[(src_cls, tgt_cls)] = func

def _get_ops_for_class(cls: type) -> "AbstractTensorOperations":
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

class AbstractTensorOperations(ABC):
    def __init__(self, track_time: bool = False) -> None:
        """Optional benchmark support for tensor operations."""
        self.track_time = track_time
        self.last_op_time: float | None = None

    def benchmark(self, call: "Callable[[], Any]") -> Any:
        """Run ``call`` and store elapsed time if benchmarking is enabled."""
        if self.track_time:
            start = time.process_time()
            result = call()
            self.last_op_time = time.process_time() - start
            return result
        return call()

    @abstractmethod
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        pass

    @abstractmethod
    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        pass

    @abstractmethod
    def clone(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def to_device(self, tensor: Any, device: Any) -> Any:
        pass

    @abstractmethod
    def get_device(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def get_dtype(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def item(self, tensor: Any) -> Union[int, float, bool]:
        pass

    @abstractmethod
    def max(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def long_cast(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def not_equal(self, tensor1: Any, tensor2: Any) -> Any:
        pass

    @abstractmethod
    def arange(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def select_by_indices(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        pass

    @abstractmethod
    def log_softmax(self, tensor: Any, dim: int) -> Any:
        pass

    @abstractmethod
    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        """Pad tensor according to `pad` specification."""
        pass

    @abstractmethod
    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        """Concatenate tensors along `dim`."""
        pass

    @abstractmethod
    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        pass

    @abstractmethod
    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def view_flat(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        pass

    @abstractmethod
    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        pass

    @abstractmethod
    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        pass

    @abstractmethod
    def shape(self, tensor: Any) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def numel(self, tensor: Any) -> int:
        pass

    @abstractmethod
    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def pow(self, tensor: Any, exponent: float) -> Any:
        pass

    @abstractmethod
    def sqrt(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> Any:
        pass

    @abstractmethod
    def boolean_mask_select(self, tensor: Any, mask: Any) -> Any:
        pass

    @abstractmethod
    def tolist(self, tensor: Any) -> List[Any]:
        pass

    @abstractmethod
    def less(self, tensor: Any, value: Any) -> Any:
        pass

    @abstractmethod
    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        pass

    def to_backend(self, tensor: Any, other: "AbstractTensorOperations") -> Any:
        """Convert ``tensor`` to ``other`` backend using registered paths."""
        if type(self) is type(other):
            return self.clone(tensor)
        path = _find_conversion_path(type(self), type(other))
        if not path:
            return default_to_backend(self, tensor, other)
        current_tensor = tensor
        current_ops: AbstractTensorOperations = self
        for src_cls, tgt_cls in path:
            func = CONVERSION_REGISTRY.get((src_cls, tgt_cls))
            if func is None:
                return default_to_backend(self, tensor, other)
            next_ops = other if tgt_cls is type(other) else _get_ops_for_class(tgt_cls)
            current_tensor = func(current_ops, current_tensor, next_ops)
            current_ops = next_ops
        if current_ops is not other:
            current_tensor = default_to_backend(current_ops, current_tensor, other)
        return current_tensor


    # --- Persistence helpers ---
    @abstractmethod
    def save(self, tensor: Any, filepath: str) -> None:
        """Persist ``tensor`` to ``filepath``."""
        pass

    @abstractmethod
    def load(self, filepath: str, dtype: Any, device: Any) -> Any:
        """Load tensor data from ``filepath``."""
        pass

    # --- Dtype helpers ---
    @property
    @abstractmethod
    def long_dtype(self) -> Any:
        """Return the integer dtype used for indices."""
        pass

    @property
    @abstractmethod
    def bool_dtype(self) -> Any:
        """Return the boolean dtype."""
        pass

    @property
    @abstractmethod
    def float_dtype(self) -> Any:
        """Return the default floating point dtype."""
        pass

    # --- Operator routing ---
    @abstractmethod
    def __apply_operator(self, op: str, left: Any, right: Any):
        """Apply an arithmetic operator to ``left`` and ``right``."""
        pass

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
    source_ops: "AbstractTensorOperations", tensor: Any, target_ops: "AbstractTensorOperations"
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


def get_tensor_operations(faculty: Faculty | None = None, *, track_time: bool = False) -> AbstractTensorOperations:
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
                        lambda s, t, o: o.tensor_from_list(t.tolist(), dtype=None, device=None))
    register_conversion(PurePythonTensorOperations, NumPyTensorOperations,
                        lambda s, t, o: o.tensor_from_list(t, dtype=None, device=None))

if torch is not None and PyTorchTensorOperations is not None and NumPyTensorOperations is not None:
    register_conversion(PyTorchTensorOperations, NumPyTensorOperations,
                        lambda s, t, o: t.detach().cpu().numpy())
    register_conversion(NumPyTensorOperations, PyTorchTensorOperations,
                        lambda s, t, o: o.to_device(o.tensor_from_list(t.tolist(), dtype=None, device=None), o.default_device))
    register_conversion(PyTorchTensorOperations, PurePythonTensorOperations,
                        lambda s, t, o: o.tensor_from_list(t.detach().cpu().tolist(), dtype=None, device=None))
    register_conversion(PurePythonTensorOperations, PyTorchTensorOperations,
                        lambda s, t, o: o.tensor_from_list(t, dtype=None, device=o.default_device))


