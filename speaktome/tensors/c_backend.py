"""Dynamic C backend for tensor operations."""

import os
import ctypes
import ctypes.util
import json
from cffi import FFI
from typing import Any, Tuple, Optional, List

from .tensor_abstraction import AbstractTensorOperations, _get_shape, _flatten

# --- END HEADER ---

class CTensorOperations(AbstractTensorOperations):
    """Wrap a shared C library via ``ctypes`` for tensor operations."""

    def __init__(self, lib_path: Optional[str] = None, track_time: bool = False) -> None:
        """Attempt to load ``libm`` for basic math operations."""
        super().__init__(track_time=track_time)

        self.lib_path = lib_path or os.environ.get("SPEAKTOME_CLIB")
        if self.lib_path is None:
            self.lib_path = ctypes.util.find_library("m")

        self.lib = None
        self.ffi: Optional[FFI] = None
        if self.lib_path:
            try:
                self.ffi = FFI()
                self.ffi.cdef("double sqrt(double x);")
                self.lib = self.ffi.dlopen(self.lib_path)
            except OSError:
                self.lib = None
        if self.lib is None:
            raise NotImplementedError("C math library could not be loaded")

    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        """Create a nested Python list filled with ``fill_value``."""
        if not size:
            return fill_value
        return [self.full(size[1:], fill_value, dtype, device) for _ in range(size[0])]

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        """Return a tensor of zeros."""
        return self.full(size, 0, dtype, device)

    def clone(self, tensor: Any) -> Any:
        """Deep copy the given tensor."""
        if isinstance(tensor, list):
            return [self.clone(t) for t in tensor]
        if hasattr(tensor, "__len__"):
            return [tensor[i] for i in range(len(tensor))]
        return tensor

    def to_device(self, tensor: Any, device: Any) -> Any:
        """Device management is a no-op for this backend."""
        return tensor

    def get_device(self, tensor: Any) -> Any:
        return "cpu_cffi"

    def get_dtype(self, tensor: Any) -> Any:
        return self.float_dtype

    def item(self, tensor: Any) -> Any:
        return self.tolist(tensor)[0] if hasattr(tensor, "__len__") else tensor

    def max(self, tensor: Any) -> Any:
        return max(self.tolist(tensor))

    def long_cast(self, tensor: Any) -> Any:
        data = [int(x) for x in self.tolist(tensor)]
        return self.tensor_from_list(data, self.long_dtype, None)

    def not_equal(self, tensor1: Any, tensor2: Any) -> Any:
        t1 = self.tolist(tensor1)
        t2 = self.tolist(tensor2)
        return [a != b for a, b in zip(t1, t2)]

    def arange(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        if end is None:
            data = list(range(start))
        else:
            data = list(range(start, end, step))
        return self.tensor_from_list(data, dtype or self.long_dtype, device)

    def select_by_indices(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        """Gather elements from a 2D list based on two index lists."""
        return [tensor[i][j] for i, j in zip(indices_dim0, indices_dim1)]

    def log_softmax(self, tensor: Any, dim: int) -> Any:
        import math
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("log_softmax only supports last dimension")
        def _compute(row: List[float]) -> List[float]:
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps)
            return [math.log(v / s) for v in exps]
        if isinstance(tensor[0], list):
            return [_compute(r) for r in tensor]
        return _compute(self.tolist(tensor))

    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) != 2:
            raise NotImplementedError("pad only supports 1D tensors")
        left, right = pad
        data = self.tolist(tensor)
        return [value] * left + data + [value] * right

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        if dim == 0:
            result = []
            for t in tensors:
                result.extend(self.tolist(t))
            return result
        if dim == 1:
            if not all(len(t) == len(tensors[0]) for t in tensors):
                raise ValueError("Tensors must have same number of rows for dim 1")
            out = []
            for i in range(len(tensors[0])):
                row = []
                for t in tensors:
                    row.extend(self.tolist(t[i]))
                out.append(row)
            return out
        raise NotImplementedError("cat only implemented for dim 0 and 1")

    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("topk only supports last dimension")
        data = self.tolist(tensor)
        idx = sorted(range(len(data)), key=lambda i: data[i], reverse=True)[:k]
        values = [data[i] for i in idx]
        return values, idx

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        if dim == 0:
            return [self.clone(t) for t in tensors]
        if dim == 1:
            ref_shape = _get_shape(self.tolist(tensors[0]))
            for t in tensors:
                if _get_shape(self.tolist(t)) != ref_shape:
                    raise ValueError("All tensors must have the same shape")
            return [[t[i] for t in tensors] for i in range(len(tensors[0]))]
        raise NotImplementedError("stack only implemented for dim 0 and 1")

    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        data = self.tolist(tensor)
        if dim is None or dim == 0:
            result = []
            for v in data:
                result.extend([v] * repeats)
            return result
        raise NotImplementedError("repeat_interleave only supports dim 0 or None")

    def view_flat(self, tensor: Any) -> Any:
        return _flatten(self.tolist(tensor))

    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        for i, j, v in zip(indices_dim0, indices_dim1, values_to_assign):
            tensor_to_modify[i][j] = v

    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        for i, flag in enumerate(mask):
            if flag:
                tensor_to_modify[i] += 1

    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        if isinstance(tensor, list):
            return [self.clamp(t, min_val, max_val) for t in tensor]
        value = tensor
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        return _get_shape(self.tolist(tensor))

    def numel(self, tensor: Any) -> int:
        return len(_flatten(self.tolist(tensor)))

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        data = self.tolist(tensor)
        if dim is None:
            flat = _flatten(data)
            return sum(flat) / len(flat) if flat else 0.0
        if dim == 0:
            return sum(data) / len(data) if data else 0.0
        if dim == 1:
            return [sum(row) / len(row) if row else 0.0 for row in data]
        raise NotImplementedError("mean only implemented for dim 0, 1, or None")

    def pow(self, tensor: Any, exponent: float) -> Any:
        if isinstance(tensor, list):
            return [self.pow(t, exponent) for t in tensor]
        return tensor ** exponent

    def sqrt(self, tensor: Any) -> Any:
        if self.lib is None:
            raise NotImplementedError("C math library not available")

        if isinstance(tensor, (list, tuple)):
            return [self.lib.sqrt(float(val)) for val in tensor]

        return self.lib.sqrt(float(tensor))

    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> Any:
        array_type = ctypes.c_double * len(data)
        arr = array_type()
        for i, v in enumerate(data):
            arr[i] = float(v)
        return arr

    def boolean_mask_select(self, tensor: Any, mask: Any) -> Any:
        data = self.tolist(tensor)
        return [v for v, m in zip(data, mask) if m]

    def tolist(self, tensor: Any) -> List[Any]:
        return [tensor[i] for i in range(len(tensor))]

    def less(self, tensor: Any, value: Any) -> Any:
        data = self.tolist(tensor)
        return [v < value for v in data]

    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        if dim == 0:
            data = self.tolist(tensor)
            return [data[i] for i in indices]
        if dim == 1:
            return [[row[i] for i in indices] for row in tensor]
        raise NotImplementedError("index_select only implemented for dim 0 or 1")

    def save(self, tensor: Any, filepath: str) -> None:
        """Persist ``tensor`` as JSON at ``filepath``."""
        with open(filepath, "w") as f:
            json.dump(self.tolist(tensor), f)

    def load(self, filepath: str, dtype: Any, device: Any) -> Any:
        """Load tensor data from ``filepath`` using :func:`tensor_from_list`."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return self.tensor_from_list(data, dtype or self.float_dtype, device)

    @property
    def long_dtype(self) -> Any:
        return ctypes.c_long

    @property
    def bool_dtype(self) -> Any:
        return ctypes.c_bool

    @property
    def float_dtype(self) -> Any:
        return ctypes.c_double

    @staticmethod
    def test() -> None:
        """Simple self-check calling ``sqrt`` from ``libm``."""
        ops = CTensorOperations()
        result = ops.sqrt([4.0, 9.0])
        assert [round(x, 1) for x in result] == [2.0, 3.0]
