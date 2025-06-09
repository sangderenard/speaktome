"""Pure Python implementation of :class:`AbstractTensorOperations`."""

from __future__ import annotations

from typing import Any, Tuple, Optional, List
import math
import json

from .abstraction import AbstractTensorOperations, _get_shape, _flatten

# --- END HEADER ---

class PurePythonTensorOperations(AbstractTensorOperations):
    """Educational tensor ops using nested Python lists."""

    def __init__(self, track_time: bool = False):
        super().__init__(track_time=track_time)
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

    def _apply_operator(self, op: str, other):
        # Only support operations with other Python-list tensors or scalars
        if isinstance(other, list):
            return self._elementwise_op(op, self, other)
        else:
            return self._elementwise_op_scalar(op, self, other)

    def _elementwise_op(self, op: str, a, b):
        # Recursively apply op to nested lists
        if not isinstance(a, list) and not isinstance(b, list):
            return self._apply_scalar_op(op, a, b)
        return [self._elementwise_op(op, ai, bi) for ai, bi in zip(a, b)]

    def _elementwise_op_scalar(self, op: str, a, scalar):
        if not isinstance(a, list):
            return self._apply_scalar_op(op, a, scalar)
        return [self._elementwise_op_scalar(op, ai, scalar) for ai in a]

    def _apply_scalar_op(self, op: str, x, y):
        if op in ('add', 'radd', 'iadd'):
            return x + y
        if op in ('sub', 'rsub', 'isub'):
            return x - y if op != 'rsub' else y - x
        if op in ('mul', 'rmul', 'imul'):
            return x * y
        if op in ('truediv', 'rtruediv', 'itruediv'):
            return x / y if op != 'rtruediv' else y / x
        if op in ('floordiv', 'rfloordiv', 'ifloordiv'):
            return x // y if op != 'rfloordiv' else y // x
        if op in ('mod', 'rmod', 'imod'):
            return x % y if op != 'rmod' else y % x
        if op in ('pow', 'rpow', 'ipow'):
            return x ** y if op != 'rpow' else y ** x
        raise NotImplementedError(f"Operator {op} not implemented for pure Python backend.")

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

    def item(self, tensor: Any) -> Any:
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

    def sub_scalar(self, tensor: Any, value: Any) -> Any:
        if isinstance(tensor, list):
            return [self.sub_scalar(item, value) for item in tensor]
        return tensor - value

    def div_scalar(self, tensor: Any, value: Any) -> Any:
        if isinstance(tensor, list):
            return [self.div_scalar(item, value) for item in tensor]
        return tensor / value

    def save(self, tensor: Any, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(tensor, f)

    def load(self, filepath: str, dtype: Any, device: Any) -> Any:
        with open(filepath, "r") as f:
            return json.load(f)

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
