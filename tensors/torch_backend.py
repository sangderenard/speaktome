"""PyTorch implementation of :class:`AbstractTensorOperations`."""

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

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from typing import Any, Tuple, List, Optional, Union

    from .abstraction import AbstractTensorOperations

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

class PyTorchTensorOperations(AbstractTensorOperations):
    def __init__(self, default_device: Union[str, "torch.device"] = "cpu", track_time: bool = False):
        super().__init__(track_time=track_time)
        if torch is None:
            raise RuntimeError("PyTorch is required for this backend")
        self.default_device = torch.device(default_device)

    def _AbstractTensorOperations__apply_operator(self, op: str, left: Any, right: Any):
        """Delegate arithmetic ops to PyTorch tensors."""
        a = left
        b = right
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

    def argmin(self, tensor, dim=None):
        return torch.argmin(tensor) if dim is None else torch.argmin(tensor, dim=dim)

    def interpolate(self, tensor, size):
        if isinstance(size, int):
            size = (size,)
        if len(size) != tensor.dim():
            raise ValueError("size must match tensor dimensions")

        def interp_dim(t, new_len, axis):
            old_len = t.shape[axis]
            if old_len == new_len:
                return t
            pos = torch.linspace(0, old_len - 1, steps=new_len, device=t.device, dtype=torch.float32)
            left = pos.floor().long()
            right = torch.clamp(left + 1, max=old_len - 1)
            weight = (pos - left.float()).view([-1 if i == axis else 1 for i in range(t.dim())])
            left_vals = torch.gather(t, axis, left.view([-1 if i == axis else 1 for i in range(t.dim())]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            right_vals = torch.gather(t, axis, right.view([-1 if i == axis else 1 for i in range(t.dim())]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            return left_vals * (1 - weight) + right_vals * weight

        result = tensor
        for d in range(tensor.dim()):
            result = interp_dim(result, size[d], d)
        return result


    def save(self, tensor, filepath: str) -> None:
        torch.save(tensor, filepath)

    def load(self, filepath: str, dtype, device):
        t = torch.load(filepath, map_location=device or self.default_device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    @property
    def long_dtype(self):
        return torch.long

    @property
    def bool_dtype(self):
        return torch.bool

    @property
    def float_dtype(self):
        return torch.float32
