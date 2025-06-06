"""Pure Python implementation of AbstractTensorOperations using lists."""
from typing import Any, Tuple, Optional, List, Union
from .tensor_abstraction import AbstractTensorOperations
import math

# Helper function to get shape of nested lists

def _get_shape(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])

# Helper function to flatten nested lists

def _flatten(data):
    if not isinstance(data, list):
        return [data]
    return [item for sublist in data for item in _flatten(sublist)]

# Recursive indexing helpers

def _recursive_access(data, indices):
    if not indices:
        return data
    idx = indices[0]
    if isinstance(idx, list):
        return [_recursive_access(data[i], indices[1:]) for i in idx]
    return _recursive_access(data[idx], indices[1:])


def _recursive_assign(data, indices, values):
    if not indices:
        return
    idx = indices[0]
    if isinstance(idx, list):
        if len(idx) != len(values):
            raise ValueError(
                f"Index list length ({len(idx)}) must match values length ({len(values)})"
            )
        for i, sub_idx in enumerate(idx):
            _recursive_assign(data[sub_idx], indices[1:], values[i])
    elif isinstance(idx, slice):
        start, stop, step = idx.indices(len(data))
        slice_indices = list(range(start, stop, step))
        if len(slice_indices) != len(values):
            raise ValueError(
                f"Slice length ({len(slice_indices)}) must match values length ({len(values)})"
            )
        for i, data_idx in enumerate(slice_indices):
            _recursive_assign(data[data_idx], indices[1:], values[i])
    else:
        if not indices[1:]:
            data[idx] = values
        else:
            _recursive_assign(data[idx], indices[1:], values)


class PurePythonTensorOperations(AbstractTensorOperations):
    """Educational tensor ops using nested Python lists."""

    def __init__(self):
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

