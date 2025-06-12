"""PyTorch implementation of :class:`AbstractTensor`."""

# TENSOR BACKEND IMPLEMENTATION GUIDELINES:
# ----------------------------------------
# 1. OPERATOR IMPLEMENTATION:
#    - DO NOT implement magic methods (__add__, __mul__, etc.)
#    - These are handled by AbstractTensor
#    - Only implement the single designated operator method from the abstract class
#
# 2. TEST COMPLIANCE:
#    - DO NOT create dummy/mock classes to pass tests
#    - DO NOT implement functions just to satisfy test requirements
#    - Either implement full functionality or leave as documented stub
#    - Failed tests are preferable to false implementations
#
# 3. BACKEND RESPONSIBILITIES:
#    - Implement only the core tensor operations defined in AbstractTensor
#    - All operator routing happens through the abstract class
#    - Let test failures expose missing functionality naturally
#
# 4. DEPENDENCIES:
#    - Import only the strictly required packages
#    - Handle import failures gracefully for optional backends
#    - Do not add dummy fallbacks for missing dependencies
#
# Remember: Magic methods and operator overloading are EXCLUSIVELY handled by
# AbstractTensor. Backend implementations provide only the raw
# tensor operations.

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from typing import Any, Tuple, List, Optional, Union

    from .abstraction import AbstractTensor

    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore
    F = None  # type: ignore
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class PyTorchTensorOperations(AbstractTensor):
    def __init__(self, default_device: Union[str, "torch.device"] = "cpu", track_time: bool = False):
        super().__init__(track_time=track_time)
        if torch is None:
            raise RuntimeError("PyTorch is required for this backend")
        self.default_device = torch.device(default_device)

    def _AbstractTensor__apply_operator_(self, op: str, left: Any, right: Any):
        """Delegate arithmetic ops to PyTorch tensors. Always unwrap to raw tensors."""
        a = self._AbstractTensor__unwrap(left)
        b = self._AbstractTensor__unwrap(right)
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
            return torch.floor_divide(a, b)
        if op == "rfloordiv":
            return torch.floor_divide(b, a)
        if op in ("mod", "imod"):
            return a % b
        if op == "rmod":
            return b % a
        if op in ("pow", "ipow"):
            return a ** b
        if op == "rpow":
            return b ** a
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for PyTorch backend.")

    def full_(self, size, fill_value, dtype, device):
        return torch.full(size, fill_value, dtype=dtype, device=device or self.default_device)

    def zeros_(self, size, dtype, device):
        return torch.zeros(size, dtype=dtype, device=device or self.default_device)

    def clone_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).clone()

    def to_device_(self, tensor, device):
        return self._AbstractTensor__unwrap(tensor).to(device or self.default_device)

    def get_device_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).device

    def get_dtype_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).dtype

    def item_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).item()

    def max_(self, tensor):
        return torch.max(self._AbstractTensor__unwrap(tensor))

    def long_cast_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).long()

    def float_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).float()

    def double_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).double()

    def int_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).int()

    def long_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).long()

    def bool_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).bool()

    def not_equal_(self, tensor1, tensor2):
        return self._AbstractTensor__unwrap(tensor1) != self._AbstractTensor__unwrap(tensor2)

    def arange_(self, start, end=None, step=1, device=None, dtype=None):
        if end is None:
            return torch.arange(start, device=device or self.default_device, dtype=dtype)
        return torch.arange(start, end, step, device=device or self.default_device, dtype=dtype)

    def select_by_indices_(self, tensor, indices_dim0, indices_dim1):
        t = self._AbstractTensor__unwrap(tensor)
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        return t[i0, i1]

    def log_softmax_(self, tensor, dim):
        return F.log_softmax(self._AbstractTensor__unwrap(tensor), dim=dim)

    def topk_(self, tensor, k, dim):
        return torch.topk(self._AbstractTensor__unwrap(tensor), k=k, dim=dim)

    def stack_(self, tensors, dim=0):
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        return torch.stack(tensors, dim=dim)

    def pad_(self, tensor, pad, value=0.0):
        return F.pad(tensor, pad, value=value)

    def cat_(self, tensors, dim=0):
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        return torch.cat(tensors, dim=dim)

    def repeat_interleave_(self, tensor, repeats, dim=None):
        return self._AbstractTensor__unwrap(tensor).repeat_interleave(repeats, dim=dim)

    def view_flat_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).view(-1)

    def assign_at_indices_(self, tensor_to_modify, indices_dim0, indices_dim1, values_to_assign):
        t = self._AbstractTensor__unwrap(tensor_to_modify)
        v = self._AbstractTensor__unwrap(values_to_assign)
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        t[i0, i1] = v
        return t

    def increment_at_indices_(self, tensor_to_modify, mask):
        t = self._AbstractTensor__unwrap(tensor_to_modify)
        m = self._AbstractTensor__unwrap(mask)
        t[m] += 1
        return t

    def clamp_(self, tensor, min_val=None, max_val=None):
        return torch.clamp(self._AbstractTensor__unwrap(tensor), min=min_val, max=max_val)

    def shape_(self, tensor):
        return tuple(self._AbstractTensor__unwrap(tensor).shape)

    def numel_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).numel()

    def mean_(self, tensor, dim=None):
        return torch.mean(self._AbstractTensor__unwrap(tensor), dim=dim)

    def pow_(self, tensor, exponent: float):
        return torch.pow(self._AbstractTensor__unwrap(tensor), exponent)

    def sqrt_(self, tensor):
        return torch.sqrt(self._AbstractTensor__unwrap(tensor))

    def tensor_from_list_(self, data, dtype, device):
        return torch.tensor(data, dtype=dtype, device=device or self.default_device)

    def boolean_mask_select_(self, tensor, mask):
        t = self._AbstractTensor__unwrap(tensor)
        m = self._AbstractTensor__unwrap(mask)
        return t[m]

    def tolist_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).tolist()

    def less_(self, tensor, value):
        return self._AbstractTensor__unwrap(tensor) < value

    def index_select_(self, tensor, dim, indices):
        t = self._AbstractTensor__unwrap(tensor)
        idx = self._AbstractTensor__unwrap(indices)
        return torch.index_select(t, dim, idx)

    def argmin_(self, tensor, dim=None):
        t = self._AbstractTensor__unwrap(tensor)
        return torch.argmin(t) if dim is None else torch.argmin(t, dim=dim)

    def get_shape(self, tensor=None):
        t = self._AbstractTensor__unwrap(tensor) if tensor is not None else self.data
        return tuple(t.shape)

    def get_ndims(self, tensor=None):
        t = self._AbstractTensor__unwrap(tensor) if tensor is not None else self.data
        return t.dim()

    def interpolate_(self, tensor, size):
        t = self._AbstractTensor__unwrap(tensor)
        if isinstance(size, int):
            size = (size,)
        if len(size) != self.get_ndims(t):
            raise ValueError("size must match tensor dimensions")
        def interp_dim(t, new_len, axis):
            old_len = t.shape[axis]
            if old_len == new_len:
                return t
            pos = torch.linspace(0, old_len - 1, steps=new_len, device=t.device, dtype=torch.float32)
            left = pos.floor().long()
            right = torch.clamp(left + 1, max=old_len - 1)
            weight = (pos - left.float()).view([-1 if i == axis else 1 for i in range(self.get_ndims(t))])
            left_vals = torch.gather(t, axis, left.view([-1 if i == axis else 1 for i in range(self.get_ndims(t))]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            right_vals = torch.gather(t, axis, right.view([-1 if i == axis else 1 for i in range(self.get_ndims(t))]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            return left_vals * (1 - weight) + right_vals * weight
        result = t
        for d in range(self.get_ndims(t)):
            result = interp_dim(result, size[d], d)
        return result

    def save_(self, tensor, filepath: str) -> None:
        torch.save(tensor, filepath)

    def load_(self, filepath: str, dtype, device):
        t = torch.load(filepath, map_location=device or self.default_device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    @property
    def long_dtype_(self):
        return torch.long

    @property
    def bool_dtype_(self):
        return torch.bool

    @property
    def float_dtype_(self):
        return torch.float32

    @property
    def tensor_type_(self) -> type:
        return torch.Tensor

    @staticmethod
    def from_numpy(source_ops, tensor, target_ops):
        if tensor is None:
            raise ValueError("from_numpy called with tensor=None")
        import torch
        arr = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.from_numpy(arr).to(target_ops.default_device)
        return result

    @staticmethod
    def from_torch(source_ops, tensor, target_ops):
        # Already a torch tensor, just move to correct device if needed
        t = tensor.data if hasattr(tensor, "data") else tensor
        if isinstance(source_ops, PyTorchTensorOperations):
            t = source_ops.data
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = t.to(target_ops.default_device)
        return result

    @staticmethod
    def from_pure(source_ops, tensor, target_ops):
        import torch
        data = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.tensor(data, device=target_ops.default_device)
        return result

    @staticmethod
    def from_jax(source_ops, tensor, target_ops):
        import torch
        import numpy as np
        arr = tensor.data if hasattr(tensor, "data") else tensor
        np_array = np.array(arr)
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.from_numpy(np_array).to(target_ops.default_device)
        return result

    def to_dtype_(self, tensor, dtype: str = "float"):
        import torch
        if dtype in ("float", "float32", "f32"):
            return tensor.float()
        elif dtype in ("float64", "double", "f64"):
            return tensor.double()
        elif dtype in ("int", "int32", "i32"):
            return tensor.int()
        elif dtype in ("int64", "long", "i64"):
            return tensor.long()
        elif dtype in ("uint8", "byte"):
            return tensor.byte()
        elif dtype in ("bool",):
            return tensor.bool()
        else:
            # Default to float32
            return tensor.float()
