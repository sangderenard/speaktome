"""Dynamic C backend for tensor operations."""

# TENSOR BACKEND IMPLEMENTATION GUIDELINES:
# ----------------------------------------
# 1. OPERATOR IMPLEMENTATION:
#    - DO NOT implement magic methods (__add__, __mul__, etc.)
#    - These are handled by AbstractTensorOperations
#    - Only implement the single designated operator method from the abstract class
#
# 2. TEST COMPLIANCE:
#    - DO NOT create dummy/mock classes to pass tests
#    - DO NOT implement functions just to satisfy test requirements
#    - Either implement full functionality or leave as documented stub
#    - Failed tests are preferable to false implementations
#
# 3. BACKEND RESPONSIBILITIES:
#    - Implement only the core tensor operations defined in AbstractTensorOperations
#    - All operator routing happens through the abstract class
#    - Let test failures expose missing functionality naturally
#
# 4. DEPENDENCIES:
#    - Import only the strictly required packages
#    - Handle import failures gracefully for optional backends
#    - Do not add dummy fallbacks for missing dependencies
#
# Remember: Magic methods and operator overloading are EXCLUSIVELY handled by
# AbstractTensorOperations. Backend implementations provide only the raw
# tensor operations.

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
    void subtract_const(const double* a, double b, double* out, int n);
    void rsubtract_const(const double* a, double b, double* out, int n);
    void mul_scalar(const double* a, double b, double* out, int n);
    void divide_const(const double* a, double b, double* out, int n);
    void rdivide_const(const double* a, double b, double* out, int n);
    void pow_scalar(const double* a, double b, double* out, int n);
    void rpow_scalar(const double* a, double b, double* out, int n);
    void mod_scalar(const double* a, double b, double* out, int n);
    void rmod_scalar(const double* a, double b, double* out, int n);
    void floor_div_const(const double* a, double b, double* out, int n);
    void rfloor_div_const(const double* a, double b, double* out, int n);
    void sqrt_double(const double* a, double* out, int n);
    void log_softmax_1d(const double* a, double* out, int n);
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
    void subtract_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b;
    }
    void rsubtract_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = b - a[i];
    }
    void mul_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] * b;
    }
    void divide_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] / b;
    }
    void rdivide_const(const double* a, double b, double* out, int n) {
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
    void floor_div_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(a[i] / b);
    }
    void rfloor_div_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(b / a[i]);
    }
    void sqrt_double(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = sqrt(a[i]);
    }
    void log_softmax_1d(const double* a, double* out, int n) {
        double max_val = a[0];
        for (int i = 1; i < n; ++i) {
            if (a[i] > max_val) max_val = a[i];
        }
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            out[i] = exp(a[i] - max_val);
            sum += out[i];
        }
        for (int i = 0; i < n; ++i) {
            out[i] = log(out[i] / sum);
        }
    }
"""

C = ffi.verify(C_SOURCE)

class CTensor:
    """C-backed tensor using cffi buffer."""
    def __init__(self, shape: Tuple[int, ...], buffer=None):
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        self.buffer = buffer if buffer is not None else ffi.new("double[]", self.size)

    def as_c_ptr(self):
        return self.buffer

    def tolist(self):
        def build(offset: int, shp: Tuple[int, ...]):
            if not shp:
                return float(self.buffer[offset])
            step = 1
            for s in shp[1:]:
                step *= s
            return [build(offset + i * step, shp[1:]) for i in range(shp[0])]

        return build(0, self.shape)

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
        buf = ffi.new("double[]", [float(x) for x in flat])
        return cls(shape, buf)

class CTensorOperations(AbstractTensorOperations):
    """C backend using cffi for all arithmetic ops."""

    def _AbstractTensorOperations__apply_operator(self, op: str, left: CTensor, right: Any):
        """Operate on ``CTensor`` objects or scalars."""
        if isinstance(right, CTensor) and isinstance(left, CTensor):
            if left.shape != right.shape:
                raise ValueError("Shape mismatch")
            out = CTensor(left.shape)
            n = left.size
            if op in ('add', 'iadd'):
                C.add_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.sub_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.div_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floordiv_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            else:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            return out
        elif isinstance(left, CTensor) and isinstance(right, (int, float)):
            out = CTensor(left.shape)
            n = left.size
            val = float(right)
            if op in ('add', 'iadd'):
                C.add_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'radd':
                C.add_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.subtract_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rsub':
                C.rsubtract_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmul':
                C.mul_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.divide_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rtruediv':
                C.rdivide_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rpow':
                C.rpow_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmod':
                C.rmod_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floor_div_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rfloordiv':
                C.rfloor_div_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
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

    def arange(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> CTensor:
        data = list(range(start)) if end is None else list(range(start, end, step))
        return CTensor.from_list(data, (len(data),))

    def pow(self, tensor: Any, exponent: float) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        out = CTensor(tensor.shape)
        C.pow_scalar(tensor.as_c_ptr(), float(exponent), out.as_c_ptr(), tensor.size)
        return out

    def sqrt(self, tensor: Any) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        out = CTensor(tensor.shape)
        C.sqrt_double(tensor.as_c_ptr(), out.as_c_ptr(), tensor.size)
        return out

    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> CTensor:
        shape = _get_shape(data)
        return CTensor.from_list(data, shape)

    def shape(self, tensor: CTensor) -> Tuple[int, ...]:
        return tensor.shape

    def numel(self, tensor: CTensor) -> int:
        return tensor.size

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        values = _flatten(tensor.tolist())
        if dim is None:
            return sum(values) / len(values) if values else 0.0
        # ########## STUB: CTensorOperations.mean_dim ##########
        # PURPOSE: Placeholder for dimension-wise mean on CTensors.
        # EXPECTED BEHAVIOR: Compute mean along the specified dimension.
        # INPUTS: ``tensor`` CTensor, ``dim`` dimension index
        # OUTPUTS: CTensor or float representing the mean.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires proper shape traversal.
        # TODO:
        #   - Implement mean over arbitrary dimensions.
        # NOTES: Current implementation only handles ``dim=None``.
        # ############################################################
        raise NotImplementedError("mean(dim) not implemented for C backend")

    def less(self, tensor: Any, value: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return [tensor.buffer[i] < value for i in range(tensor.size)]

    def view_flat(self, tensor: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return _flatten(tensor.tolist())

    def tolist(self, tensor: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return tensor.tolist()

    def clamp(
        self,
        tensor: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        out = CTensor(tensor.shape)
        for i in range(tensor.size):
            val = tensor.buffer[i]
            if min_val is not None and val < min_val:
                val = min_val
            if max_val is not None and val > max_val:
                val = max_val
            out.buffer[i] = val
        return out

    def select_by_indices(self, tensor: CTensor, indices_dim0: Any, indices_dim1: Any) -> Any:
        # ########## STUB: CTensorOperations.select_by_indices ##########
        # PURPOSE: Gather elements from ``tensor`` using two index arrays.
        # EXPECTED BEHAVIOR: Return a 1D CTensor of selected values.
        # INPUTS: ``tensor`` CTensor, ``indices_dim0`` list, ``indices_dim1`` list.
        # OUTPUTS: CTensor with values from ``tensor[indices_dim0[i], indices_dim1[i]]``.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires stride calculations.
        # TODO:
        #   - Implement efficient index selection.
        # NOTES: Complex indexing left for future work.
        # ############################################################
        raise NotImplementedError("select_by_indices not implemented for C backend")

    def log_softmax(self, tensor: CTensor, dim: int) -> Any:
        """Compute log softmax along ``dim`` using C routines."""
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        if dim < 0:
            dim += len(tensor.shape)
        if dim != 0 or len(tensor.shape) != 1:
            raise NotImplementedError(
                "log_softmax only implemented for 1D tensors on the C backend"
            )
        out = CTensor(tensor.shape)
        C.log_softmax_1d(tensor.as_c_ptr(), out.as_c_ptr(), tensor.size)
        return out

    def pad(self, tensor: CTensor, pad: Tuple[int, ...], value: float = 0) -> Any:
        # ########## STUB: CTensorOperations.pad ##########
        # PURPOSE: Pad ``tensor`` with ``value`` according to ``pad`` spec.
        # EXPECTED BEHAVIOR: Return new CTensor with additional elements.
        # INPUTS: ``tensor`` CTensor, padding tuple.
        # OUTPUTS: Padded CTensor.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires dynamic shape manipulation.
        # TODO:
        #   - Implement generic padding logic for all dimensions.
        # NOTES: Currently unsupported.
        # ############################################################
        raise NotImplementedError("pad not implemented for C backend")

    def topk(self, tensor: CTensor, k: int, dim: int) -> Tuple[Any, Any]:
        # ########## STUB: CTensorOperations.topk ##########
        # PURPOSE: Return the top ``k`` values and their indices along ``dim``.
        # EXPECTED BEHAVIOR: Similar to numpy.take_along_axis with sorting.
        # INPUTS: ``tensor`` CTensor, ``k`` int, ``dim`` dimension index.
        # OUTPUTS: Tuple of (values CTensor, indices list).
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires sorting capabilities.
        # TODO:
        #   - Implement efficient top-k selection.
        # NOTES: Complex due to absence of vectorized sort in this backend.
        # ############################################################
        if dim < 0:
            dim += len(tensor.shape)
        if dim != len(tensor.shape) - 1:
            raise NotImplementedError(
                "topk only implemented for the last dimension"
            )

        if len(tensor.shape) != 1:
            raise NotImplementedError("topk only implemented for 1D tensors")

        data = tensor.tolist()
        indexed = sorted(
            [(i, v) for i, v in enumerate(data)], key=lambda x: x[1], reverse=True
        )[:k]
        values = [v for _, v in indexed]
        indices = [i for i, _ in indexed]
        return CTensor.from_list(values, (len(values),)), CTensor.from_list(
            indices, (len(indices),)
        )

    def repeat_interleave(self, tensor: CTensor, repeats: int, dim: Optional[int] = None) -> Any:
        # ########## STUB: CTensorOperations.repeat_interleave ##########
        # PURPOSE: Repeat each element in ``tensor`` ``repeats`` times along ``dim``.
        # EXPECTED BEHAVIOR: New CTensor with expanded dimension.
        # INPUTS: CTensor, repeats int, optional dimension.
        # OUTPUTS: CTensor with repeated values.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced slicing helpers.
        # TODO:
        #   - Implement axis-aware repetition.
        # NOTES: Only full flattening supported currently.
        # ############################################################
        raise NotImplementedError("repeat_interleave not implemented for C backend")

    def assign_at_indices(
        self,
        tensor_to_modify: CTensor,
        indices_dim0: Any,
        indices_dim1: Any,
        values_to_assign: Any,
    ) -> None:
        # ########## STUB: CTensorOperations.assign_at_indices ##########
        # PURPOSE: In-place assignment into ``tensor_to_modify`` at specified indices.
        # EXPECTED BEHAVIOR: Modifies tensor values according to index lists.
        # INPUTS: target CTensor, two index lists, values list.
        # OUTPUTS: None (in-place modification).
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires index math support.
        # TODO:
        #   - Implement multi-dimensional indexing.
        # NOTES: Stubbed pending full CTensor infrastructure.
        # ############################################################
        raise NotImplementedError("assign_at_indices not implemented for C backend")

    def increment_at_indices(self, tensor_to_modify: CTensor, mask: Any) -> None:
        # ########## STUB: CTensorOperations.increment_at_indices ##########
        # PURPOSE: Increment elements of ``tensor_to_modify`` where ``mask`` is True.
        # EXPECTED BEHAVIOR: Element-wise increment using a boolean mask.
        # INPUTS: target CTensor, boolean mask list.
        # OUTPUTS: None (in-place update).
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires broadcasting rules.
        # TODO:
        #   - Implement efficient masked increment.
        # NOTES: Stub to signal incomplete feature.
        # ############################################################
        raise NotImplementedError("increment_at_indices not implemented for C backend")

    def boolean_mask_select(self, tensor: CTensor, mask: Any) -> Any:
        # ########## STUB: CTensorOperations.boolean_mask_select ##########
        # PURPOSE: Select values from ``tensor`` where ``mask`` is True.
        # EXPECTED BEHAVIOR: Return CTensor or list of filtered values.
        # INPUTS: CTensor, mask list.
        # OUTPUTS: CTensor containing selected elements.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires shape-aware masking.
        # TODO:
        #   - Implement boolean masking for CTensors.
        # NOTES: Not yet supported in C backend.
        # ############################################################
        raise NotImplementedError("boolean_mask_select not implemented for C backend")

    def index_select(self, tensor: CTensor, dim: int, indices: Any) -> Any:
        # ########## STUB: CTensorOperations.index_select ##########
        # PURPOSE: Select entries along ``dim`` using ``indices``.
        # EXPECTED BEHAVIOR: Mirror numpy.take along the specified dimension.
        # INPUTS: CTensor, dimension index, index list.
        # OUTPUTS: CTensor with gathered values.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced indexing support.
        # TODO:
        #   - Implement general index selection.
        # NOTES: Current design does not provide necessary helpers.
        # ############################################################
        raise NotImplementedError("index_select not implemented for C backend")

    def stack(self, tensors: list, dim: int = 0) -> Any:
        # ########## STUB: CTensorOperations.stack ##########
        # PURPOSE: Concatenate a sequence of CTensors along a new dimension.
        # EXPECTED BEHAVIOR: Should return a CTensor representing ``tensors``
        #     stacked along ``dim``.
        # INPUTS: list of CTensors, dimension index.
        # OUTPUTS: New CTensor with increased rank.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced shape handling.
        # TODO:
        #   - Implement shape validation and memory allocation logic.
        # NOTES: Current C backend lacks generic tensor helpers.
        # ############################################################
        raise NotImplementedError("stack not implemented for C backend")

    def cat(self, tensors: list, dim: int = 0) -> Any:
        # ########## STUB: CTensorOperations.cat ##########
        # PURPOSE: Concatenate CTensors along an existing dimension.
        # EXPECTED BEHAVIOR: Should join ``tensors`` on ``dim`` similar to
        #     numpy.concatenate.
        # INPUTS: list of CTensors, dimension index.
        # OUTPUTS: New CTensor with combined size.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires complex stride math.
        # TODO:
        #   - Implement concatenation across arbitrary dimensions.
        # NOTES: Placeholder until a more complete C tensor API exists.
        # ############################################################
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
        assert [round(x, 1) for x in result.tolist()] == [2.0, 3.0]

