"""Dynamic C backend for tensor operations."""

import os
import ctypes
import ctypes.util
import json
from typing import Any, Tuple, Optional, List
from cffi import FFI

# The tensor abstraction module was renamed to ``abstraction``. Update imports
# accordingly so the C backend stays in sync with the other backends.
from .abstraction import AbstractTensorOperations, _get_shape, _flatten

# --- END HEADER ---

ffi = FFI()
ffi.cdef("""
    void add_double(const double* a, const double* b, double* out, int n);
    void sub_double(const double* a, const double* b, double* out, int n);
    void mul_double(const double* a, const double* b, double* out, int n);
    void div_double(const double* a, const double* b, double* out, int n);
    void pow_double(const double* a, const double* b, double* out, int n);
    void mod_double(const double* a, const double* b, double* out, int n);
    void floordiv_double(const double* a, const double* b, double* out, int n);
    // Scalar ops
    void add_scalar(const double* a, double b, double* out, int n);
    void sub_scalar(const double* a, double b, double* out, int n);
    void rsub_scalar(const double* a, double b, double* out, int n);
    void mul_scalar(const double* a, double b, double* out, int n);
    void div_scalar(const double* a, double b, double* out, int n);
    void rdiv_scalar(const double* a, double b, double* out, int n);
    void pow_scalar(const double* a, double b, double* out, int n);
    void rpow_scalar(const double* a, double b, double* out, int n);
    void mod_scalar(const double* a, double b, double* out, int n);
    void rmod_scalar(const double* a, double b, double* out, int n);
    void floordiv_scalar(const double* a, double b, double* out, int n);
    void rfloordiv_scalar(const double* a, double b, double* out, int n);
""")

C_SOURCE = """
    #include <math.h>
    void add_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    void sub_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }
    void mul_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
    }
    void div_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] / b[i];
    }
    void pow_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(a[i], b[i]);
    }
    void mod_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(a[i], b[i]);
    }
    void floordiv_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(a[i] / b[i]);
    }
    // Scalar ops
    void add_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] + b;
    }
    void sub_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b;
    }
    void rsub_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = b - a[i];
    }
    void mul_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] * b;
    }
    void div_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] / b;
    }
    void rdiv_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = b / a[i];
    }
    void pow_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(a[i], b);
    }
    void rpow_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(b, a[i]);
    }
    void mod_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(a[i], b);
    }
    void rmod_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(b, a[i]);
    }
    void floordiv_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(a[i] / b);
    }
    void rfloordiv_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(b / a[i]);
    }
"""

C = ffi.verify(C_SOURCE)

class CTensor:
    """C-backed tensor using cffi buffer."""
    def __init__(self, shape: Tuple[int, ...], buffer=None):
        self.shape = shape
        self.size = -1
        self.buffer = buffer if buffer is not None else ffi.new("double[]", self.size)

    def as_c_ptr(self):
        return self.buffer

    def tolist(self):
        return [self.buffer[i] for i in range(self.size)]

    @classmethod
    def from_list(cls, data: list, shape: Tuple[int, ...]):
        flat = []
        def flatten(x):
            if isinstance(x, list):
                for item in x:
                    flatten(item)
            else:
                flat.append(float(x))
        flatten(data)
        buf = ffi.new("double[]", flat)
        return cls(shape, buf)

class CTensorOperations(AbstractTensorOperations):
    """C backend using cffi for all arithmetic ops."""

    def _apply_operator(self, op: str, other):
        # Only support CTensor <op> CTensor or CTensor <op> scalar
        if isinstance(other, CTensor):
            if self.shape != other.shape:
                raise ValueError("Shape mismatch")
            out = CTensor(self.shape)
            n = self.size
            if op in ('add', 'iadd'):
                C.add_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.sub_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.div_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floordiv_double(self.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), n)
            else:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            return out
        elif isinstance(other, (int, float)):
            out = CTensor(self.shape)
            n = self.size
            val = float(other)
            if op in ('add', 'iadd'):
                C.add_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'radd':
                C.add_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.sub_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rsub':
                C.rsub_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmul':
                C.mul_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.div_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rtruediv':
                C.rdiv_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rpow':
                C.rpow_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmod':
                C.rmod_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floordiv_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rfloordiv':
                C.rfloordiv_scalar(self.as_c_ptr(), val, out.as_c_ptr(), n)
            else:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            return out
        else:
            raise TypeError("CTensorOperations only supports CTensor or scalar operands.")

    # Creation ops
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        t = CTensor(size)
        for i in range(t.size):
            t.buffer[i] = float(fill_value)
        return t

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return self.full(size, 0.0, dtype, device)

    def clone(self, tensor: CTensor) -> CTensor:
        t = CTensor(tensor.shape)
        for i in range(t.size):
            t.buffer[i] = tensor.buffer[i]
        return t

    def to_device(self, tensor: CTensor, device: Any) -> CTensor:
        return tensor  # No-op for now

    def stack(self, tensors: list, dim: int = 0) -> Any:
        # Not implemented for C backend in this example
        raise NotImplementedError("stack not implemented for C backend")

    def cat(self, tensors: list, dim: int = 0) -> Any:
        # Not implemented for C backend in this example
        raise NotImplementedError("cat not implemented for C backend")

    def get_device(self, tensor: CTensor) -> str:
        return "cpu_cffi"

    def get_dtype(self, tensor: CTensor) -> Any:
        return float

    def item(self, tensor: CTensor) -> Any:
        if tensor.size == 1:
            return tensor.buffer[0]
        raise ValueError("Tensor has more than one element")

    def max(self, tensor: CTensor) -> float:
        return max(tensor.tolist())

    def long_cast(self, tensor: CTensor) -> CTensor:
        t = CTensor(tensor.shape)
        for i in range(t.size):
            t.buffer[i] = int(tensor.buffer[i])
        return t

    def not_equal(self, tensor1: CTensor, tensor2: CTensor) -> list:
        return [tensor1.buffer[i] != tensor2.buffer[i] for i in range(tensor1.size)]

    def save(self, tensor: CTensor, filepath: str) -> None:
        with open(filepath, "wb") as f:
            f.write(ffi.buffer(tensor.buffer, tensor.size * 8))

    def load(self, filepath: str, dtype: Any, device: Any) -> CTensor:
        with open(filepath, "rb") as f:
            data = f.read()
        n = len(data) // 8
        buf = ffi.new("double[]", n)
        ffi.memmove(buf, data, len(data))
        # You must provide shape info externally!
        return CTensor((n,), buf)

    @property
    def long_dtype(self) -> Any:
        return int

    @property
    def bool_dtype(self) -> Any:
        return bool

    @property
    def float_dtype(self) -> Any:
        return float

    @staticmethod
    def test() -> None:
        """Simple self-check calling ``sqrt`` from ``libm``."""
        ops = CTensorOperations()
        result = ops.sqrt([4.0, 9.0])