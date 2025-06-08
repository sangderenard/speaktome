"""Dynamic C backend for tensor operations."""

import os
import ctypes
import ctypes.util
from cffi import FFI
from typing import Any, Tuple, Optional, List

from .tensor_abstraction import AbstractTensorOperations

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
        raise NotImplementedError("full not implemented for CTensorOperations")

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        raise NotImplementedError("zeros not implemented for CTensorOperations")

    def clone(self, tensor: Any) -> Any:
        raise NotImplementedError("clone not implemented for CTensorOperations")

    def to_device(self, tensor: Any, device: Any) -> Any:
        raise NotImplementedError("to_device not implemented for CTensorOperations")

    def get_device(self, tensor: Any) -> Any:
        raise NotImplementedError("get_device not implemented for CTensorOperations")

    def get_dtype(self, tensor: Any) -> Any:
        raise NotImplementedError("get_dtype not implemented for CTensorOperations")

    def item(self, tensor: Any) -> Any:
        raise NotImplementedError("item not implemented for CTensorOperations")

    def max(self, tensor: Any) -> Any:
        raise NotImplementedError("max not implemented for CTensorOperations")

    def long_cast(self, tensor: Any) -> Any:
        raise NotImplementedError("long_cast not implemented for CTensorOperations")

    def not_equal(self, tensor1: Any, tensor2: Any) -> Any:
        raise NotImplementedError("not_equal not implemented for CTensorOperations")

    def arange(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        raise NotImplementedError("arange not implemented for CTensorOperations")

    def select_by_indices(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        raise NotImplementedError("select_by_indices not implemented for CTensorOperations")

    def log_softmax(self, tensor: Any, dim: int) -> Any:
        raise NotImplementedError("log_softmax not implemented for CTensorOperations")

    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        raise NotImplementedError("pad not implemented for CTensorOperations")

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        raise NotImplementedError("cat not implemented for CTensorOperations")

    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        raise NotImplementedError("topk not implemented for CTensorOperations")

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        raise NotImplementedError("stack not implemented for CTensorOperations")

    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        raise NotImplementedError("repeat_interleave not implemented for CTensorOperations")

    def view_flat(self, tensor: Any) -> Any:
        raise NotImplementedError("view_flat not implemented for CTensorOperations")

    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        raise NotImplementedError("assign_at_indices not implemented for CTensorOperations")

    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        raise NotImplementedError("increment_at_indices not implemented for CTensorOperations")

    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        raise NotImplementedError("clamp not implemented for CTensorOperations")

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        raise NotImplementedError("shape not implemented for CTensorOperations")

    def numel(self, tensor: Any) -> int:
        raise NotImplementedError("numel not implemented for CTensorOperations")

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        raise NotImplementedError("mean not implemented for CTensorOperations")

    def pow(self, tensor: Any, exponent: float) -> Any:
        raise NotImplementedError("pow not implemented for CTensorOperations")

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
        raise NotImplementedError("boolean_mask_select not implemented for CTensorOperations")

    def tolist(self, tensor: Any) -> List[Any]:
        return [tensor[i] for i in range(len(tensor))]

    def less(self, tensor: Any, value: Any) -> Any:
        raise NotImplementedError("less not implemented for CTensorOperations")

    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        raise NotImplementedError("index_select not implemented for CTensorOperations")

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
