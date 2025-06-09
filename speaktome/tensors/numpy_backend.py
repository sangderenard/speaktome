"""NumPy implementation of :class:`AbstractTensorOperations`."""

from typing import Any, Tuple, List, Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from .abstraction import AbstractTensorOperations

# --- END HEADER ---

class NumPyTensorOperations(AbstractTensorOperations):
    def __init__(self, track_time: bool = False):
        super().__init__(track_time=track_time)

    def _apply_operator(self, op: str, left: Any, right: Any):
        """Apply arithmetic operators on NumPy arrays."""
        a = np.array(left) if not isinstance(left, np.ndarray) else left
        b = np.array(right) if not isinstance(right, np.ndarray) else right
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
            return np.floor_divide(a, b)
        if op == "rfloordiv":
            return np.floor_divide(b, a)
        if op in ("mod", "imod"):
            return np.mod(a, b)
        if op == "rmod":
            return np.mod(b, a)
        if op in ("pow", "ipow"):
            return np.power(a, b)
        if op == "rpow":
            return np.power(b, a)
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for NumPy backend.")

    def _torch_dtype_to_numpy(self, dtype):
        if torch is None:
            return dtype
        if dtype == torch.float32:
            return np.float32
        if dtype == torch.int64:
            return np.int64
        if dtype == torch.float64:
            return np.float64
        if dtype == torch.int32:
            return np.int32
        if dtype == torch.bool:
            return np.bool_
        return None

    def _numpy_dtype_to_torch(self, dtype):
        if torch is None:
            return dtype
        if dtype == np.float32:
            return torch.float32
        if dtype == np.float64:
            return torch.float64
        if dtype == np.int64:
            return torch.int64
        if dtype == np.int32:
            return torch.int32
        if dtype == np.bool_:
            return torch.bool
        return None

    def full(self, size, fill_value, dtype, device):
        return np.full(size, fill_value, dtype=self._torch_dtype_to_numpy(dtype))

    def zeros(self, size, dtype, device):
        return np.zeros(size, dtype=self._torch_dtype_to_numpy(dtype))

    def clone(self, tensor):
        return np.array(tensor, copy=True)

    def to_device(self, tensor, device):
        return tensor

    def get_device(self, tensor):
        return 'cpu'

    def get_dtype(self, tensor):
        if isinstance(tensor, np.ndarray):
            return self._numpy_dtype_to_torch(tensor.dtype)
        return tensor.dtype

    def item(self, tensor):
        return tensor.item()

    def max(self, tensor):
        return np.max(tensor)

    def long_cast(self, tensor):
        return tensor.astype(np.int64)

    def not_equal(self, tensor1, tensor2):
        return tensor1 != tensor2

    def arange(self, start, end=None, step=1, device=None, dtype=None):
        np_dtype = self._torch_dtype_to_numpy(dtype) if dtype is not None else None
        if end is None:
            return np.arange(start, dtype=np_dtype)
        return np.arange(start, end, step, dtype=np_dtype)

    def select_by_indices(self, tensor, indices_dim0, indices_dim1):
        return tensor[indices_dim0, indices_dim1]

    def log_softmax(self, tensor, dim):
        e_x = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=dim, keepdims=True)
        return np.log(softmax)

    def topk(self, tensor, k, dim):
        if dim < 0:
            dim = tensor.ndim + dim
        sorted_indices = np.argsort(tensor, axis=dim)
        idx_slice = [slice(None)] * tensor.ndim
        idx_slice[dim] = slice(tensor.shape[dim] - k, tensor.shape[dim])
        top_k_indices_ascending = sorted_indices[tuple(idx_slice)]
        top_k_indices = np.flip(top_k_indices_ascending, axis=dim)
        values = np.take_along_axis(tensor, top_k_indices, axis=dim)
        return values, top_k_indices

    def stack(self, tensors, dim=0):
        return np.stack(tensors, axis=dim)

    def pad(self, tensor, pad, value=0.0):
        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")
        num_dims_to_pad = len(pad) // 2
        if num_dims_to_pad > tensor.ndim:
            raise ValueError("Padding tuple length implies padding more dimensions than tensor has.")
        np_pad_width = []
        for _ in range(tensor.ndim - num_dims_to_pad):
            np_pad_width.append((0, 0))
        for i in range(num_dims_to_pad):
            left = pad[-2 * (i + 1)]
            right = pad[-2 * (i + 1) + 1]
            np_pad_width.append((left, right))
        return np.pad(tensor, pad_width=np_pad_width, constant_values=value)

    def cat(self, tensors, dim=0):
        return np.concatenate(tensors, axis=dim)

    def repeat_interleave(self, tensor, repeats, dim=None):
        return np.repeat(tensor, repeats, axis=dim)

    def view_flat(self, tensor):
        return tensor.reshape(-1)

    def assign_at_indices(self, tensor_to_modify, indices_dim0, indices_dim1, values_to_assign):
        tensor_to_modify[indices_dim0, indices_dim1] = values_to_assign

    def increment_at_indices(self, tensor_to_modify, mask):
        tensor_to_modify[mask] += 1

    def clamp(self, tensor, min_val=None, max_val=None):
        return np.clip(tensor, a_min=min_val, a_max=max_val)

    def shape(self, tensor):
        return tuple(tensor.shape)

    def numel(self, tensor):
        return tensor.size

    def mean(self, tensor, dim=None):
        return np.mean(tensor, axis=dim)

    def pow(self, tensor, exponent: float):
        return np.power(tensor, exponent)

    def sqrt(self, tensor):
        return np.sqrt(tensor)

    def tensor_from_list(self, data, dtype, device):
        return np.array(data, dtype=self._torch_dtype_to_numpy(dtype))

    def boolean_mask_select(self, tensor, mask):
        return tensor[mask]

    def tolist(self, tensor):
        return tensor.tolist()

    def less(self, tensor, value):
        return tensor < value

    def index_select(self, tensor, dim, indices):
        return np.take(tensor, indices, axis=dim)

    def sub_scalar(self, tensor, value):
        return tensor - value

    def div_scalar(self, tensor, value):
        return tensor / value

    def save(self, tensor, filepath: str) -> None:
        np.save(filepath, tensor)

    def load(self, filepath: str, dtype, device):
        arr = np.load(f"{filepath}.npy") if not filepath.endswith('.npy') else np.load(filepath)
        if dtype is not None:
            arr = arr.astype(self._torch_dtype_to_numpy(dtype))
        return arr

    @property
    def long_dtype(self):
        return np.int64

    @property
    def bool_dtype(self):
        return np.bool_

    @property
    def float_dtype(self):
        return np.float32
