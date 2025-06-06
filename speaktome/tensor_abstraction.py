from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union
import torch
import torch.nn.functional as F
import numpy as np

class AbstractTensorOperations(ABC):
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


class PyTorchTensorOperations(AbstractTensorOperations):
    def __init__(self, default_device: Union[str, torch.device] = "cpu"):
        self.default_device = torch.device(default_device)

    def full(self, size, fill_value, dtype, device):
        return torch.full(size, fill_value, dtype=dtype, device=device or self.default_device)

    def zeros(self, size, dtype, device):
        return torch.zeros(size, dtype=dtype, device=device or self.default_device)

    def clone(self, tensor):
        return tensor.clone()

    def to_device(self, tensor, device):
        return tensor.to(device or self.default_device)

    def get_device(self, tensor):
        return tensor.device

    def get_dtype(self, tensor):
        return tensor.dtype

    def item(self, tensor):
        return tensor.item()

    def max(self, tensor):
        return torch.max(tensor)

    def long_cast(self, tensor):
        return tensor.long()

    def not_equal(self, tensor1, tensor2):
        return tensor1 != tensor2

    def arange(self, start, end=None, step=1, device=None, dtype=None):
        if end is None:
            return torch.arange(start, device=device or self.default_device, dtype=dtype)
        return torch.arange(start, end, step, device=device or self.default_device, dtype=dtype)

    def select_by_indices(self, tensor, indices_dim0, indices_dim1):
        return tensor[indices_dim0, indices_dim1]

    def log_softmax(self, tensor, dim):
        return F.log_softmax(tensor, dim=dim)

    def topk(self, tensor, k, dim):
        return torch.topk(tensor, k=k, dim=dim)

    def pad(self, tensor, pad, value=0.0):
        return F.pad(tensor, pad, value=value)

    def cat(self, tensors, dim=0):
        return torch.cat(tensors, dim=dim)

    def repeat_interleave(self, tensor, repeats, dim=None):
        return tensor.repeat_interleave(repeats, dim=dim)

    def view_flat(self, tensor):
        return tensor.view(-1)

    def assign_at_indices(self, tensor_to_modify, indices_dim0, indices_dim1, values_to_assign):
        tensor_to_modify[indices_dim0, indices_dim1] = values_to_assign

    def increment_at_indices(self, tensor_to_modify, mask):
        tensor_to_modify[mask] += 1

    def clamp(self, tensor, min_val=None, max_val=None):
        return torch.clamp(tensor, min=min_val, max=max_val)

    def shape(self, tensor):
        return tuple(tensor.shape)

    def numel(self, tensor):
        return tensor.numel()

    def mean(self, tensor, dim=None):
        return torch.mean(tensor, dim=dim)

    def pow(self, tensor, exponent: float):
        return torch.pow(tensor, exponent)

    def sqrt(self, tensor):
        return torch.sqrt(tensor)

    def tensor_from_list(self, data, dtype, device):
        return torch.tensor(data, dtype=dtype, device=device or self.default_device)

    def boolean_mask_select(self, tensor, mask):
        return tensor[mask]


class NumPyTensorOperations(AbstractTensorOperations):
    def __init__(self):
        pass

    def _torch_dtype_to_numpy(self, dtype):
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
        if dim != -1:
            raise NotImplementedError("topk only implemented for last dim in numpy backend")
        indices = np.argsort(tensor, axis=dim)[:, -k:][:, ::-1]
        values = np.take_along_axis(tensor, indices, axis=dim)
        return values, indices

    def pad(self, tensor, pad, value=0.0):
        if len(pad) % 2 != 0:
            raise ValueError("pad must have even length")
        pad_width = [(pad[i], pad[i+1]) for i in range(0, len(pad), 2)]
        return np.pad(tensor, pad_width, constant_values=value)

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
        return tensor.shape

    def numel(self, tensor):
        return tensor.size

    def mean(self, tensor, dim=None):
        return np.mean(tensor, axis=dim)

    def pow(self, tensor, exponent):
        return np.power(tensor, exponent)

    def sqrt(self, tensor):
        return np.sqrt(tensor)

    def tensor_from_list(self, data, dtype, device):
        return np.array(data, dtype=self._torch_dtype_to_numpy(dtype))

    def boolean_mask_select(self, tensor, mask):
        return tensor[mask]

