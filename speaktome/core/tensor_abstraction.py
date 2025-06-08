from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union
import math

from ..faculty import Faculty, DEFAULT_FACULTY
from .. import config
# --- END HEADER ---
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

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


class PyTorchTensorOperations(AbstractTensorOperations):
    def __init__(self, default_device: Union[str, "torch.device"] = "cpu"):
        if torch is None:
            raise RuntimeError("PyTorch is required for this backend")
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

    def stack(self, tensors, dim=0):
        return torch.stack(tensors, dim=dim)

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

    def tolist(self, tensor):
        return tensor.tolist()

    def less(self, tensor, value):
        return tensor < value
    def index_select(self, tensor, dim, indices):
        return torch.index_select(tensor, dim, indices)

    # Dtype helpers
    @property
    def long_dtype(self):
        return torch.long

    @property
    def bool_dtype(self):
        return torch.bool

    @property
    def float_dtype(self):
        return torch.float32


class NumPyTensorOperations(AbstractTensorOperations):
    def __init__(self):
        pass

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
        # Ensure dim is positive for consistent slicing behavior with argsort
        if dim < 0:
            dim = tensor.ndim + dim

        # Get the indices that would sort 'tensor' along 'dim' in ascending order
        sorted_indices = np.argsort(tensor, axis=dim)

        # Create slices to select the last k elements along 'dim'
        # These are the indices of the k largest values, but sorted by value ascendingly
        idx_slice = [slice(None)] * tensor.ndim
        idx_slice[dim] = slice(tensor.shape[dim] - k, tensor.shape[dim])

        # Get the indices of the k largest values
        top_k_indices_ascending = sorted_indices[tuple(idx_slice)]

        # Flip them along 'dim' to get them in descending order of value (largest first)
        top_k_indices = np.flip(top_k_indices_ascending, axis=dim)

        # Use these indices to gather the top k values
        values = np.take_along_axis(tensor, top_k_indices, axis=dim)

        return values, top_k_indices

    def stack(self, tensors, dim=0):
        return np.stack(tensors, axis=dim)

    def pad(self, tensor, pad, value=0.0):
        # PyTorch F.pad format: (pad_left, pad_right, pad_top, pad_bottom, ...)
        # padding is specified for dimensions from last to first.
        # NumPy np.pad format: ((before_axis_0, after_axis_0), (before_axis_1, after_axis_1), ...)
        # We need to convert from PyTorch style to NumPy style.
        if len(pad) % 2 != 0: # Should be an even number of elements
            raise ValueError("Padding length must be even.")
        
        num_dims_to_pad = len(pad) // 2
        if num_dims_to_pad > tensor.ndim:
            raise ValueError("Padding tuple length implies padding more dimensions than tensor has.")

        # Create (before, after) pairs from the PyTorch-style pad tuple
        # PyTorch pad: (last_dim_begin, last_dim_end, second_last_dim_begin, second_last_dim_end, ...)
        # NumPy pad_width: ((first_dim_begin, first_dim_end), (second_dim_begin, second_dim_end), ...)
        
        np_pad_width = []
        # Fill padding for dimensions not specified in `pad` tuple (typically leading dimensions)
        for _ in range(tensor.ndim - num_dims_to_pad):
            np_pad_width.append((0, 0))
            
        # Add padding for the dimensions specified in `pad`, in reverse order of pairs
        for i in range(num_dims_to_pad):
            # pad elements are (left, right) for each dim, starting from the last dim
            # So pad[2*i] is left pad for (num_dims_to_pad - 1 - i)-th dim from the end
            # And pad[2*i+1] is right pad for that dim
            left_pad = pad[2 * (num_dims_to_pad - 1 - i)]
            right_pad = pad[2 * (num_dims_to_pad - 1 - i) + 1]
            np_pad_width.append((left_pad, right_pad))
        
        return np.pad(tensor, np_pad_width, constant_values=value)

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

    def tolist(self, tensor):
        return tensor.tolist()

    def less(self, tensor, value):
        return tensor < value
    def index_select(self, tensor, dim, indices):
        return np.take(tensor, indices, axis=dim)


    # Dtype helpers
    @property
    def long_dtype(self):
        return np.int64

    @property
    def bool_dtype(self):
        return np.bool_

    @property
    def float_dtype(self):
        return np.float32


class JAXTensorOperations(AbstractTensorOperations):
    """Stub backend for future JAX-based tensor operations."""

    # ########## STUB: JAXTensorOperations ##########
    # PURPOSE: Provide an optional JAX backend mirroring the
    #          :class:`PyTorchTensorOperations` API.
    # EXPECTED BEHAVIOR: Implement all tensor operations using
    #          ``jax.numpy`` with device management for CPU/GPU/TPU.
    # INPUTS: JAX arrays and standard Python lists.
    # OUTPUTS: JAX arrays or converted Python data structures.
    # KEY ASSUMPTIONS/DEPENDENCIES: Requires the ``jax`` package and
    #          compatible accelerator drivers.
    # TODO:
    #   - Implement each method using ``jax.numpy`` equivalents.
    #   - Handle device placement and data transfer semantics.
    #   - Integrate this class with :func:`get_tensor_operations`.
    # NOTES: This class currently raises ``NotImplementedError`` to
    #        indicate the backend is not yet available.
    # ###############################################################
    def __init__(self, default_device: str = "cpu") -> None:
        raise NotImplementedError("JAX backend not yet implemented")

    @staticmethod
    def test() -> None:
        """Demonstrate the current stub behaviour."""
        try:
            JAXTensorOperations()
        except NotImplementedError:
            return
        raise AssertionError("JAXTensorOperations should be a stub")


class PurePythonTensorOperations(AbstractTensorOperations):
    """Educational tensor ops using nested Python lists."""

    def __init__(self):
        # ########## STUB: PurePythonTensorOperations.__init__ ##########
        # PURPOSE: Placeholder for any future initialization logic needed for
        #          the pure Python backend.
        # EXPECTED BEHAVIOR: Should set up attributes or configuration values
        #          when this backend requires them.
        # INPUTS: None at present.
        # OUTPUTS: None.
        # KEY ASSUMPTIONS/DEPENDENCIES: Currently no dependencies beyond the
        #          abstract interface.
        # TODO:
        #   - Add configurable parameters if performance tuning becomes
        #     relevant.
        # NOTES: Implementation intentionally empty.
        # ###############################################################
        pass

    # Creation ops
    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        if not size:
            return fill_value
        return [self.full(size[1:], fill_value, dtype, device) for _ in range(size[0])]

    def zeros(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return self.full(size, 0, dtype, device)

    def clone(self, tensor: Any) -> Any:
        if not isinstance(tensor, list):
            return tensor
        return [self.clone(item) for item in tensor]

    def to_device(self, tensor: Any, device: Any) -> Any:
        return tensor

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        if not tensors:
            return []
        if dim == 0:
            return [self.clone(t) for t in tensors]
        # verify shapes match
        ref_shape = _get_shape(tensors[0])
        for t in tensors:
            if _get_shape(t) != ref_shape:
                raise ValueError("All tensors must have the same shape")
        return [self.stack([t[i] for t in tensors], dim=dim - 1) for i in range(len(tensors[0]))]

    def get_device(self, tensor: Any) -> Any:
        return "cpu_pure_python"

    def get_dtype(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            if not tensor:
                return None
            return self.get_dtype(tensor[0])
        return type(tensor)

    def item(self, tensor: Any) -> Union[int, float, bool]:
        if isinstance(tensor, list) and len(tensor) == 1:
            return tensor[0]
        return tensor

    def max(self, tensor: Any) -> Any:
        flat = _flatten(tensor)
        return max(flat) if flat else None

    def long_cast(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            return [self.long_cast(item) for item in tensor]
        return int(tensor)

    def not_equal(self, tensor1: Any, tensor2: Any) -> Any:
        if isinstance(tensor1, list) and isinstance(tensor2, list):
            return [self.not_equal(t1, t2) for t1, t2 in zip(tensor1, tensor2)]
        return tensor1 != tensor2

    def arange(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> Any:
        if end is None:
            return list(range(start))
        return list(range(start, end, step))

    def select_by_indices(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            raise NotImplementedError("select_by_indices only supports 2D lists for now")

        selected_rows = [tensor[i] for i in indices_dim0]
        if isinstance(indices_dim1, list):
            if len(indices_dim0) != len(indices_dim1):
                raise ValueError("Index lists must have same length for element-wise selection")
            return [selected_rows[i][indices_dim1[i]] for i in range(len(selected_rows))]
        elif isinstance(indices_dim1, slice):
            return [row[indices_dim1] for row in selected_rows]
        else:
            return [row[indices_dim1] for row in selected_rows]

    def log_softmax(self, tensor: Any, dim: int) -> Any:
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("log_softmax only implemented for last dimension")

        if not isinstance(tensor, list):
            return math.log(1.0)
        if not isinstance(tensor[0], list):
            max_val = max(tensor)
            exp_tensor = [math.exp(x - max_val) for x in tensor]
            sum_exp = sum(exp_tensor)
            return [math.log(x / sum_exp) for x in exp_tensor]
        return [self.log_softmax(sublist, dim=-1) for sublist in tensor]

    def topk(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("topk only implemented for last dimension")
        if not isinstance(tensor, list):
            return [tensor], [0]
        if not isinstance(tensor[0], list):
            indexed = sorted([(v, i) for i, v in enumerate(tensor)], reverse=True)[:k]
            values = [v for v, _ in indexed]
            indices = [i for _, i in indexed]
            return values, indices
        topk_values = []
        topk_indices = []
        for sublist in tensor:
            values, idxs = self.topk(sublist, k, dim=-1)
            topk_values.append(values)
            topk_indices.append(idxs)
        return topk_values, topk_indices

    def pad(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) != 4:
            raise NotImplementedError("pad only implemented for 2D tensors")
        pad_left, pad_right, pad_top, pad_bottom = pad
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            raise ValueError("pad expects a 2D list")
        rows = len(tensor)
        cols = len(tensor[0]) if rows > 0 else 0
        padded_rows = []
        for _ in range(pad_top):
            padded_rows.append([value] * (cols + pad_left + pad_right))
        for row in tensor:
            padded_rows.append([value] * pad_left + row + [value] * pad_right)
        for _ in range(pad_bottom):
            padded_rows.append([value] * (cols + pad_left + pad_right))
        return padded_rows

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        if not tensors:
            return []
        if dim == 0:
            result = []
            for t in tensors:
                result.extend(t)
            return result
        if dim == 1:
            if not all(len(t) == len(tensors[0]) for t in tensors):
                raise ValueError("Tensors must have same number of rows for dim 1 concatenation")
            result = []
            for i in range(len(tensors[0])):
                combined = []
                for t in tensors:
                    combined.extend(t[i])
                result.append(combined)
            return result
        raise NotImplementedError("cat only implemented for dim 0 and 1")

    def repeat_interleave(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        if dim is None or dim == 0:
            if not isinstance(tensor, list):
                return [tensor] * repeats
            result = []
            for item in tensor:
                result.extend([item] * repeats)
            return result
        raise NotImplementedError("repeat_interleave only implemented for dim 0 or None")

    def view_flat(self, tensor: Any) -> Any:
        return _flatten(tensor)

    def assign_at_indices(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        if not isinstance(tensor_to_modify, list) or not isinstance(tensor_to_modify[0], list):
            raise NotImplementedError("assign_at_indices only supports 2D lists for now")
        if not isinstance(indices_dim0, list) or not isinstance(indices_dim1, list):
            raise ValueError("indices_dim0 and indices_dim1 must be lists")
        if len(indices_dim0) != len(indices_dim1) or len(indices_dim0) != len(values_to_assign):
            raise ValueError("Index lists and values list must have same length")
        for i in range(len(indices_dim0)):
            row_idx = indices_dim0[i]
            col_idx = indices_dim1[i]
            value = values_to_assign[i]
            tensor_to_modify[row_idx][col_idx] = value

    def increment_at_indices(self, tensor_to_modify: Any, mask: Any):
        if not isinstance(tensor_to_modify, list) or not isinstance(mask, list) or len(tensor_to_modify) != len(mask):
            raise NotImplementedError("increment_at_indices only supports flat lists with boolean mask")
        for i in range(len(tensor_to_modify)):
            if mask[i]:
                tensor_to_modify[i] += 1

    def clamp(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        if isinstance(tensor, list):
            return [self.clamp(item, min_val, max_val) for item in tensor]
        value = tensor
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        return _get_shape(tensor)

    def numel(self, tensor: Any) -> int:
        return len(_flatten(tensor))

    def mean(self, tensor: Any, dim: Optional[int] = None) -> Any:
        if not isinstance(tensor, list):
            return tensor
        if dim is None or dim == 0:
            flat = _flatten(tensor)
            return sum(flat) / len(flat) if flat else 0.0
        if dim == 1:
            if not isinstance(tensor[0], list):
                raise ValueError("Tensor must be 2D for mean along dim 1")
            return [sum(row) / len(row) if row else 0.0 for row in tensor]
        raise NotImplementedError("mean only implemented for dim 0, 1, or None")

    def pow(self, tensor: Any, exponent: float) -> Any:
        if isinstance(tensor, list):
            return [self.pow(item, exponent) for item in tensor]
        return tensor ** exponent

    def sqrt(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            return [self.sqrt(item) for item in tensor]
        return math.sqrt(tensor)

    def tensor_from_list(self, data: List[Any], dtype: Any, device: Any) -> Any:
        return data

    def boolean_mask_select(self, tensor: Any, mask: Any) -> Any:
        if not isinstance(tensor, list) or not isinstance(mask, list) or len(tensor) != len(mask):
            raise NotImplementedError("boolean_mask_select only supports flat lists with boolean mask")
        return [tensor[i] for i in range(len(tensor)) if mask[i]]

    def tolist(self, tensor: Any) -> List[Any]:
        return self.clone(tensor)

    def less(self, tensor: Any, value: Any) -> Any:
        if isinstance(tensor, list):
            return [self.less(item, value) for item in tensor]
        return tensor < value
    def index_select(self, tensor: Any, dim: int, indices: Any) -> Any:
        if dim == 0:
            return [tensor[i] for i in indices]
        if dim == 1:
            return [[row[i] for i in indices] for row in tensor]
        raise NotImplementedError("index_select only implemented for dim 0 or 1")

    # Dtype helpers
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
        """Quick smoke test for basic operations."""
        ops = PurePythonTensorOperations()
        stacked = ops.stack([[1, 2], [3, 4]], dim=0)
        assert stacked == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
        values, idxs = ops.topk([1, 3, 2, 4], k=2, dim=-1)
        assert values == [4, 3] and idxs == [3, 1]


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


def get_tensor_operations(faculty: Faculty | None = None) -> AbstractTensorOperations:
    """Return a tensor operations backend based on the faculty tier."""

    faculty = faculty or DEFAULT_FACULTY
    if faculty in (Faculty.TORCH, Faculty.PYGEO):
        return PyTorchTensorOperations(default_device=config.DEVICE)
    if faculty is Faculty.NUMPY and np is not None:
        return NumPyTensorOperations()
    if faculty is Faculty.CTENSOR:
        from .c_tensor_ops import CTensorOperations
        return CTensorOperations()
    return PurePythonTensorOperations()


