"""JAX implementation of :class:`AbstractTensorOperations`."""

from __future__ import annotations

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

from typing import Any, Tuple, List, Optional

from .abstraction import AbstractTensorOperations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import jax
    import jax.numpy as jnp
    from jax import lax
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore
    lax = None  # type: ignore
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class JAXTensorOperations(AbstractTensorOperations):
    """Tensor operations powered by `jax.numpy`."""

    def __init__(self, default_device: Optional[Any] = None, track_time: bool = False) -> None:
        super().__init__(track_time=track_time)
        self.default_device = default_device
        self._validate_jax_setup()

    def _validate_jax_setup(self) -> None:
        """Validates JAX installation and available devices."""
        try:
            devices = jax.devices()
            self.available_devices = {str(d): d for d in devices}
            self.has_gpu = any('gpu' in str(d).lower() for d in devices)
            self.has_tpu = any('tpu' in str(d).lower() for d in devices)
        except Exception as e:
            raise RuntimeError(f"JAX initialization failed: {str(e)}")

    def _to_jnp(self, tensor: Any) -> jnp.ndarray:
        """Safely convert input to JAX array."""
        if isinstance(tensor, jnp.ndarray):
            return tensor
        return jnp.array(tensor)

    def to_device(self, tensor: Any, device: Any) -> Any:
        """Move tensor to specified device with validation."""
        target_device = device or self.default_device
        if target_device is not None:
            device_str = str(target_device).lower()
            if ('gpu' in device_str and not self.has_gpu) or \
               ('tpu' in device_str and not self.has_tpu):
                print(f"Warning: Requested device {device_str} not available. Using CPU.")
                target_device = jax.devices('cpu')[0]
        
        return jax.device_put(self._to_jnp(tensor), target_device)

    def _AbstractTensorOperations__apply_operator(self, op: str, left: Any, right: Any):
        """Apply arithmetic ops using JAX arrays."""
        a = self._to_jnp(left)
        b = self._to_jnp(right)
        if op in ("add", "iadd"):
            return a + b
        if op == "radd":
            return b + a
        if op in ("sub", "isub"):
            return a - b
        if op == "rsub":
            return b - a
        if op in ("mul", "imul"):
            return a * b
        if op == "rmul":
            return b * a
        if op in ("truediv", "itruediv"):
            return a / b
        if op == "rtruediv":
            return b / a
        if op in ("floordiv", "ifloordiv"):
            return jnp.floor(a / b)
        if op == "rfloordiv":
            return jnp.floor(b / a)
        if op in ("mod", "imod"):
            return jnp.mod(a, b)
        if op == "rmod":
            return jnp.mod(b, a)
        if op in ("pow", "ipow"):
            return jnp.power(a, b)
        if op == "rpow":
            return jnp.power(b, a)
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for JAX backend.")

    # ------------------------------------------------------------------
    # Creation ops
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        return jax.device_put(jnp.full(size, fill_value, dtype=dtype), device or self.default_device)

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return jax.device_put(jnp.zeros(size, dtype=dtype), device or self.default_device)

    def clone(self, tensor: Any) -> Any:
        return jnp.array(tensor, copy=True)

    # ------------------------------------------------------------------
    # Basic info
    def get_device(self, tensor: Any) -> Any:
        return tensor.device

    def get_dtype(self, tensor: Any) -> Any:
        return tensor.dtype

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def max(self, tensor: Any) -> Any:
        return jnp.max(tensor)

    def long_cast(self, tensor: Any) -> Any:
        return tensor.astype(jnp.int64)

    def not_equal(self, tensor1: Any, tensor2: Any) -> Any:
        return jnp.not_equal(tensor1, tensor2)

    def arange(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        arr = jnp.arange(start, end, step, dtype=dtype) if end is not None else jnp.arange(start, dtype=dtype)
        return jax.device_put(arr, device or self.default_device)

    def select_by_indices(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        return self._to_jnp(tensor)[indices_dim0, indices_dim1]

    def log_softmax(self, tensor: Any, dim: int) -> Any:
        from jax.nn import log_softmax
        return log_softmax(self._to_jnp(tensor), axis=dim)

    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")
        num_dims_to_pad = len(pad) // 2
        pad_width: List[Tuple[int, int]] = []
        tensor = self._to_jnp(tensor)
        for _ in range(tensor.ndim - num_dims_to_pad):
            pad_width.append((0, 0))
        for i in range(num_dims_to_pad):
            left = pad[-2 * (i + 1)]
            right = pad[-2 * (i + 1) + 1]
            pad_width.append((left, right))
        return jnp.pad(tensor, pad_width=tuple(pad_width), constant_values=value).tolist()

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [self._to_jnp(self.ensure_tensor(t)) for t in tensors]
        return jnp.concatenate(tensors, axis=dim).tolist()

    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        """Return the top ``k`` values and indices along ``dim``."""
        tensor = self._to_jnp(tensor)
        if dim < 0:
            dim = tensor.ndim + dim
        if dim < 0 or dim >= tensor.ndim:
            raise ValueError("dim out of range")

        if dim == tensor.ndim - 1:
            values, idxs = lax.top_k(tensor, k)
        else:
            moved = jnp.moveaxis(tensor, dim, -1)
            values, idxs = lax.top_k(moved, k)
            values = jnp.moveaxis(values, -1, dim)
            idxs = jnp.moveaxis(idxs, -1, dim)
        return values.tolist(), idxs.tolist()

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [self._to_jnp(self.ensure_tensor(t)) for t in tensors]
        return jnp.stack(tensors, axis=dim).tolist()

    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        return jnp.repeat(self._to_jnp(tensor), repeats, axis=dim).tolist()

    def view_flat(self, tensor: Any) -> Any:
        return jnp.ravel(self._to_jnp(tensor)).tolist()

    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        tensor_to_modify = self._to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[indices_dim0, indices_dim1].set(self._to_jnp(values_to_assign))
        return updated

    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        tensor_to_modify = self._to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[mask].add(1)
        return updated

    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        return jnp.clip(self._to_jnp(tensor), a_min=min_val, a_max=max_val)

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        return tuple(self._to_jnp(tensor).shape)

    def numel(self, tensor: Any) -> int:
        return self._to_jnp(tensor).size

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return jnp.mean(self._to_jnp(tensor), axis=dim)

    def pow(self, tensor: Any, exponent: float) -> Any:
        return jnp.power(self._to_jnp(tensor), exponent)

    def sqrt(self, tensor: Any) -> Any:
        return jnp.sqrt(self._to_jnp(tensor))

    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> Any:
        arr = jnp.array(data, dtype=dtype)
        return jax.device_put(arr, device or self.default_device)

    def boolean_mask_select(self, tensor: Any, mask: Any) -> Any:
        return self._to_jnp(tensor)[mask]

    def tolist(self, tensor: Any) -> List[Any]:
        return list(self._to_jnp(tensor).tolist())

    def less(self, tensor: Any, value: Any) -> Any:
        return jnp.less(self._to_jnp(tensor), value)

    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        return jnp.take(self._to_jnp(tensor), indices, axis=dim)

    def argmin(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return jnp.argmin(self._to_jnp(tensor)) if dim is None else jnp.argmin(self._to_jnp(tensor), axis=dim)

    def interpolate(self, tensor: Any, size: Tuple[int, ...]) -> Any:
        arr = self._to_jnp(tensor)
        if len(size) != arr.ndim:
            raise ValueError("size must match tensor dimensions")

        def interp_axis(a, new_len, axis):
            old_len = a.shape[axis]
            if old_len == new_len:
                return a
            pos = jnp.linspace(0, old_len - 1, new_len)
            left = jnp.floor(pos).astype(jnp.int32)
            right = jnp.clip(left + 1, 0, old_len - 1)
            weight = pos - left
            left_vals = jnp.take(a, left, axis=axis)
            right_vals = jnp.take(a, right, axis=axis)
            shape = [1] * a.ndim
            shape[axis] = new_len
            weight = weight.reshape(shape)
            return left_vals * (1 - weight) + right_vals * weight

        result = arr
        for d in range(arr.ndim):
            result = interp_axis(result, size[d], d)
        return result


    # --- Persistence helpers ---
    def save(self, tensor: Any, filepath: str) -> None:
        import numpy as np
        np.save(filepath, np.array(tensor))

    def load(self, filepath: str, dtype: Any, device: Any) -> Any:
        import numpy as np
        arr = np.load(f"{filepath}.npy") if not filepath.endswith('.npy') else np.load(filepath)
        arr = jnp.array(arr, dtype=dtype) if dtype is not None else jnp.array(arr)
        return jax.device_put(arr, device or self.default_device)

    # --- Dtype helpers ---
    @property
    def long_dtype(self) -> Any:
        return int

    @property
    def bool_dtype(self) -> Any:
        return bool

    @property
    def float_dtype(self) -> Any:
        return float

    @property
    def tensor_type(self) -> type:
        return jnp.ndarray

    @staticmethod
    def test() -> None:
        """Quick smoke test for the JAX backend."""
        if jax is None:
            print("JAX not available")
            return
        ops = JAXTensorOperations()
        stacked = ops.stack([jnp.array([1, 2]), jnp.array([3, 4])], dim=0)
        assert ops.tolist(stacked) == [[1, 2], [3, 4]]
        values, idxs = ops.topk(jnp.array([1, 3, 2, 4]), k=2, dim=-1)
        assert ops.tolist(values) == [4, 3] and ops.tolist(idxs) == [3, 1]
