"""JAX implementation of :class:`AbstractTensorOperations`."""

from typing import Any, Tuple, List, Optional

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore
    lax = None  # type: ignore

from .abstraction import AbstractTensorOperations

# --- END HEADER ---

def _to_jnp(x: Any) -> Any:
    """Convert nested lists to a ``jax.numpy`` array if needed."""
    if jnp is None:
        raise RuntimeError("JAX is not available")
    if isinstance(x, (list, tuple)):
        return jnp.array(x)
    return x

class JAXTensorOperations(AbstractTensorOperations):
    """Tensor operations powered by `jax.numpy`."""

    def __init__(self, default_device: str = "cpu", track_time: bool = False) -> None:
        super().__init__(track_time=track_time)
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for this backend")
        if isinstance(default_device, str):
            devices = jax.devices(default_device)
            self.default_device = devices[0] if devices else jax.devices()[0]
        else:
            self.default_device = default_device

    # ------------------------------------------------------------------
    # Creation ops
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        return jax.device_put(jnp.full(size, fill_value, dtype=dtype), device or self.default_device)

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return jax.device_put(jnp.zeros(size, dtype=dtype), device or self.default_device)

    def clone(self, tensor: Any) -> Any:
        return jnp.array(tensor, copy=True)

    def to_device(self, tensor: Any, device: Any) -> Any:
        return jax.device_put(_to_jnp(tensor), device or self.default_device)

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
        return _to_jnp(tensor)[indices_dim0, indices_dim1]

    def log_softmax(self, tensor: Any, dim: int) -> Any:
        from jax.nn import log_softmax
        return log_softmax(_to_jnp(tensor), axis=dim)

    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")
        num_dims_to_pad = len(pad) // 2
        pad_width: List[Tuple[int, int]] = []
        tensor = _to_jnp(tensor)
        for _ in range(tensor.ndim - num_dims_to_pad):
            pad_width.append((0, 0))
        for i in range(num_dims_to_pad):
            left = pad[-2 * (i + 1)]
            right = pad[-2 * (i + 1) + 1]
            pad_width.append((left, right))
        return jnp.pad(tensor, pad_width=tuple(pad_width), constant_values=value).tolist()

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [_to_jnp(t) for t in tensors]
        return jnp.concatenate(tensors, axis=dim).tolist()

    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        if dim != -1 and dim != tensor.ndim - 1:
            raise NotImplementedError("topk only implemented for last dimension")
        tensor = _to_jnp(tensor)
        values, idxs = lax.top_k(tensor, k)
        return values.tolist(), idxs.tolist()

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [_to_jnp(t) for t in tensors]
        return jnp.stack(tensors, axis=dim).tolist()

    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        return jnp.repeat(_to_jnp(tensor), repeats, axis=dim).tolist()

    def view_flat(self, tensor: Any) -> Any:
        return jnp.ravel(_to_jnp(tensor)).tolist()

    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        tensor_to_modify = _to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[indices_dim0, indices_dim1].set(_to_jnp(values_to_assign))
        return updated

    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        tensor_to_modify = _to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[mask].add(1)
        return updated

    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        return jnp.clip(_to_jnp(tensor), a_min=min_val, a_max=max_val)

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        return tuple(_to_jnp(tensor).shape)

    def numel(self, tensor: Any) -> int:
        return _to_jnp(tensor).size

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return jnp.mean(_to_jnp(tensor), axis=dim)

    def pow(self, tensor: Any, exponent: float) -> Any:
        return jnp.power(_to_jnp(tensor), exponent)

    def sqrt(self, tensor: Any) -> Any:
        return jnp.sqrt(_to_jnp(tensor))

    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> Any:
        arr = jnp.array(data, dtype=dtype)
        return jax.device_put(arr, device or self.default_device)

    def boolean_mask_select(self, tensor: Any, mask: Any) -> Any:
        return _to_jnp(tensor)[mask]

    def tolist(self, tensor: Any) -> List[Any]:
        return list(_to_jnp(tensor).tolist())

    def less(self, tensor: Any, value: Any) -> Any:
        return jnp.less(_to_jnp(tensor), value)

    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        return jnp.take(_to_jnp(tensor), indices, axis=dim)

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

