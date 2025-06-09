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
    void log_softmax_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        double* out);

    void pad_double_nd(const double* input, double* output, const int* shape, const int* new_shape, const int* left_pad, int dims, double value);
    void mean_dim(const double* a, double* out, const int* shape, int ndim, int dim);
    void gather_pairs_2d(const double* a, const int* rows, const int* cols,
                         double* out, int n_pairs, int stride);
    double sum_double(const double* a, int n);
    void create_arange(double start, double step, int n, double* out);
    void topk_double(const double* a, int n, int k, int* indices, double* out);
    void topk_double_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        int k,
        double* indices,
        double* out);
    void for_each_cell_along_dim(
        const double* data,
        const int* shape,
        int ndim,
        int batch_dim,
        void (*callback)(const double*, int, int, void*),
        void* user_data);
""")

C_SOURCE = """
    #include <math.h>
    #include <stdlib.h>
    #include <stddef.h>

    typedef struct {
        double* out;
        int dim_size;
        int cell_len;
        int out_index;
        double* accum;
    } mean_dim_ctx;

    static void mean_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        mean_dim_ctx* ctx = (mean_dim_ctx*)user_data;
        if (idx == 0) {
            for (int i = 0; i < ctx->cell_len; ++i) ctx->accum[i] = 0.0;
        }
        for (int i = 0; i < ctx->cell_len; ++i) ctx->accum[i] += data[i];
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) {
                ctx->out[ctx->out_index * ctx->cell_len + i] = ctx->accum[i] / ctx->dim_size;
            }
            ctx->out_index++;
        }
    }

    typedef struct {
        double* values;
        double* indices;
        int k;
        int dim_size;
        int cell_len;
        int out_index;
        double* buffer;
    } topk_dim_ctx;

    static void topk_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        topk_dim_ctx* ctx = (topk_dim_ctx*)user_data;
        for (int i = 0; i < ctx->cell_len; ++i) {
            ctx->buffer[idx * ctx->cell_len + i] = data[i];
        }
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) {
                const double* slice = ctx->buffer + i;
                int* used = (int*)malloc(ctx->dim_size * sizeof(int));
                for (int u = 0; u < ctx->dim_size; ++u) used[u] = 0;
                for (int t = 0; t < ctx->k; ++t) {
                    int best = -1;
                    double best_val = -1e300;
                    for (int j = 0; j < ctx->dim_size; ++j) {
                        double val = slice[j * ctx->cell_len];
                        if (!used[j] && val > best_val) {
                            best_val = val;
                            best = j;
                        }
                    }
                    int out_pos = (ctx->out_index * ctx->cell_len + i) * ctx->k + t;
                    if (best != -1) {
                        used[best] = 1;
                        ctx->indices[out_pos] = best;
                        ctx->values[out_pos] = best_val;
                    } else {
                        ctx->indices[out_pos] = -1;
                        ctx->values[out_pos] = 0.0;
                    }
                }
                free(used);
            }
            ctx->out_index++;
        }
    }

    typedef struct {
        double* out;
        int dim_size;
        int cell_len;
        int out_index;
        double* buffer;
        double* max_vals;
        double* sum_vals;
    } log_softmax_dim_ctx;

    static void log_softmax_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        log_softmax_dim_ctx* ctx = (log_softmax_dim_ctx*)user_data;
        double* slice = ctx->buffer + idx * ctx->cell_len;
        for (int i = 0; i < ctx->cell_len; ++i) {
            slice[i] = data[i];
            if (idx == 0 || data[i] > ctx->max_vals[i]) ctx->max_vals[i] = data[i];
        }
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) ctx->sum_vals[i] = 0.0;
            for (int j = 0; j < ctx->dim_size; ++j) {
                double* row = ctx->buffer + j * ctx->cell_len;
                for (int i = 0; i < ctx->cell_len; ++i) {
                    row[i] = exp(row[i] - ctx->max_vals[i]);
                    ctx->sum_vals[i] += row[i];
                }
            }
            for (int j = 0; j < ctx->dim_size; ++j) {
                double* row = ctx->buffer + j * ctx->cell_len;
                for (int i = 0; i < ctx->cell_len; ++i) {
                    int pos = (ctx->out_index * ctx->dim_size + j) * ctx->cell_len + i;
                    ctx->out[pos] = log(row[i] / ctx->sum_vals[i]);
                }
            }
            ctx->out_index++;
        }
    }

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

    typedef struct {
        double* out;
        int axis;
        int* out_index;
    } log_softmax_ctx;

    static void log_softmax_callback(const double* slice, int stride, int idx, void* user_data) {
        log_softmax_ctx* ctx = (log_softmax_ctx*)user_data;
        double max_val = slice[0];
        for (int i = 1; i < ctx->axis; ++i) {
            double v = slice[i * stride];
            if (v > max_val) max_val = v;
        }
        double sum = 0.0;
        for (int i = 0; i < ctx->axis; ++i) {
            double e = exp(slice[i * stride] - max_val);
            ctx->out[*ctx->out_index + i] = e;
            sum += e;
        }
        for (int i = 0; i < ctx->axis; ++i) {
            double e = ctx->out[*ctx->out_index + i];
            ctx->out[*ctx->out_index + i] = log(e / sum);
        }
        *ctx->out_index += ctx->axis;
    }

    void log_softmax_dim(
        const double* a,
        double* out,
        const int* shape,
        int ndim,
        int dim) {
        int out_idx = 0;
        log_softmax_ctx ctx = {out, shape[dim], &out_idx};
        for_each_cell_along_dim(a, shape, ndim, dim, log_softmax_callback, &ctx);
    }

    void pad_double_nd(const double* input, double* output, const int* shape,
                       const int* new_shape, const int* left_pad, int dims,
                       double value) {
        int i;
        int total_out = 1;
        for (i = 0; i < dims; ++i) total_out *= new_shape[i];
        for (i = 0; i < total_out; ++i) output[i] = value;

        int input_size = 1;
        for (i = 0; i < dims; ++i) input_size *= shape[i];

        int* in_stride = (int*)malloc(sizeof(int) * dims);
        int* out_stride = (int*)malloc(sizeof(int) * dims);
        int* idx = (int*)malloc(sizeof(int) * dims);

        in_stride[dims - 1] = 1;
        out_stride[dims - 1] = 1;
        for (i = dims - 2; i >= 0; --i) {
            in_stride[i] = in_stride[i + 1] * shape[i + 1];
            out_stride[i] = out_stride[i + 1] * new_shape[i + 1];
        }

        for (i = 0; i < dims; ++i) idx[i] = 0;
        for (i = 0; i < input_size; ++i) {
            int out_index = 0;
            for (int d = 0; d < dims; ++d) {
                out_index += (idx[d] + left_pad[d]) * out_stride[d];
            }
            output[out_index] = input[i];
            idx[dims - 1]++;
            for (int d = dims - 1; d > 0; --d) {
                if (idx[d] >= shape[d]) {
                    idx[d] = 0;
                    idx[d - 1]++;
                }
            }
        }

        free(in_stride);
        free(out_stride);
        free(idx);

    }

    void mean_dim(const double* a, double* out, const int* shape, int ndim, int dim) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        double* accum = (double*)malloc(after * sizeof(double));
        mean_dim_ctx ctx = {out, shape[dim], after, 0, accum};
        for_each_cell_along_dim(a, shape, ndim, dim, mean_dim_callback, &ctx);
        free(accum);
    }

    void gather_pairs_2d(const double* a, const int* rows, const int* cols,
                         double* out, int n_pairs, int stride) {
        for (int i = 0; i < n_pairs; ++i) {
            out[i] = a[rows[i] * stride + cols[i]];
        }
    }
    double sum_double(const double* a, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += a[i];
        }
        return s;
    }

    void create_arange(double start, double step, int n, double* out) {
        for (int i = 0; i < n; i++) {
            out[i] = start + i * step;
        }
    }

    void topk_double(const double* a, int n, int k, int* indices, double* out) {
        int i, j, best;
        double best_val;
        int *used = (int*)malloc(n * sizeof(int));
        for (i = 0; i < n; i++) used[i] = 0;
        for (i = 0; i < k; i++) {
            best = -1;
            best_val = -1e300; // a very small number as initial comparator
            for (j = 0; j < n; j++) {
                if (!used[j] && a[j] > best_val) {
                    best_val = a[j];
                    best = j;
                }
            }
            if (best != -1) {
                used[best] = 1;
                indices[i] = best;
                out[i] = best_val;
            } else {
                indices[i] = -1;
                out[i] = 0.0;
            }
        }
        free(used);

    }

    void topk_double_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        int k,
        double* indices,
        double* out) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        double* buffer = (double*)malloc(shape[dim] * after * sizeof(double));
        topk_dim_ctx ctx = {out, indices, k, shape[dim], after, 0, buffer};
        for_each_cell_along_dim(a, shape, ndim, dim, topk_dim_callback, &ctx);
        free(buffer);
    }

    void log_softmax_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        double* out) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int dim_size = shape[dim];
        double* buffer = (double*)malloc(dim_size * after * sizeof(double));
        double* max_vals = (double*)malloc(after * sizeof(double));
        double* sum_vals = (double*)malloc(after * sizeof(double));
        log_softmax_dim_ctx ctx = {out, dim_size, after, 0, buffer, max_vals, sum_vals};
        for_each_cell_along_dim(a, shape, ndim, dim, log_softmax_dim_callback, &ctx);
        free(buffer);
        free(max_vals);
        free(sum_vals);
    }

    void for_each_cell_along_dim(
        const double* data,
        const int* shape,
        int ndim,
        int batch_dim,
        void (*callback)(const double*, int, int, void*),
        void* user_data) {
        // Compute strides
        int* strides = (int*)malloc(ndim * sizeof(int));
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        int before = 1, after = 1;
        for (int i = 0; i < batch_dim; ++i) before *= shape[i];
        for (int i = batch_dim + 1; i < ndim; ++i) after *= shape[i];
        int batch = shape[batch_dim];
        int cell_len = after;

        // Iterate over all cells
        for (int b = 0; b < before; ++b) {
            // Compute base offset for this cell
            int base = 0;
            int rem = b;
            for (int i = batch_dim - 1; i >= 0; --i) {
                int idx = rem % shape[i];
                base += idx * strides[i];
                rem /= shape[i];
            }
            for (int i = 0; i < batch; ++i) {
                int cell_offset = base + i * strides[batch_dim];
                callback(data + cell_offset, cell_len, i, user_data);
            }
        }
        free(strides);
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
        ffi.memmove(t.buffer, tensor.buffer, tensor.size * ffi.sizeof("double"))
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
        if end is None:
            n = start
            start_val = 0.0
            step_val = 1.0
        else:
            n = int((end - start) // step)
            start_val = float(start)
            step_val = float(step)
        out = CTensor((n,))
        C.create_arange(start_val, step_val, n, out.as_c_ptr())
        return out

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

        shape = tensor.shape
        if dim < 0:
            dim += len(shape)
        if dim < 0 or dim >= len(shape):
            raise ValueError("dim out of range")

        out_shape = shape[:dim] + shape[dim + 1 :]
        out = CTensor(out_shape if out_shape else ())
        shape_arr = ffi.new("int[]", list(shape))
        C.mean_dim(tensor.as_c_ptr(), out.as_c_ptr(), shape_arr, len(shape), dim)
        if not out_shape:
            return out.buffer[0]
        return out

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
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

        rows = list(indices_dim0)

        if isinstance(indices_dim1, slice):
            start, stop, step = indices_dim1.indices(tensor.shape[1])
            cols_range = list(range(start, stop, step))
            row_arr = [r for r in rows for _ in cols_range]
            col_arr = cols_range * len(rows)
            out_shape = (len(rows), len(cols_range))
        else:
            cols = [indices_dim1] * len(rows) if isinstance(indices_dim1, int) else list(indices_dim1)
            if len(rows) != len(cols):
                raise ValueError("Index lists must have same length for element-wise selection")
            row_arr = rows
            col_arr = cols
            out_shape = (len(cols),) if not isinstance(indices_dim1, int) else (len(rows),)

        row_buf = ffi.new("int[]", row_arr)
        col_buf = ffi.new("int[]", col_arr)
        n_pairs = len(row_arr)
        out_buf = ffi.new("double[]", n_pairs)
        C.gather_pairs_2d(tensor.as_c_ptr(), row_buf, col_buf, out_buf, n_pairs, tensor.shape[1])

        return CTensor(out_shape, out_buf)

    def log_softmax(self, tensor: CTensor, dim: int) -> Any:
        """Compute log softmax along ``dim`` using C routines."""
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        ndim = len(tensor.shape)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")
        c_shape = ffi.new("int[]", list(tensor.shape))
        out = CTensor(tensor.shape)
        if ndim == 1:
            C.log_softmax_1d(tensor.as_c_ptr(), out.as_c_ptr(), tensor.size)
        else:
            C.log_softmax_dim(tensor.as_c_ptr(), c_shape, ndim, dim, out.as_c_ptr())
        return out

    def pad(self, tensor: CTensor, pad: Tuple[int, ...], value: float = 0) -> Any:
        """Pad ``tensor`` with ``value`` according to ``pad`` specification."""
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")

        dims = len(tensor.shape)
        num_pad_dims = len(pad) // 2
        if num_pad_dims > dims:
            raise ValueError(
                "Padding tuple length implies padding more dimensions than tensor has."
            )

        left = [0] * dims
        right = [0] * dims
        for i in range(num_pad_dims):
            left[dims - num_pad_dims + i] = int(pad[-2 * (i + 1)])
            right[dims - num_pad_dims + i] = int(pad[-2 * (i + 1) + 1])

        new_shape = [
            tensor.shape[i] + left[i] + right[i] for i in range(dims)
        ]

        out = CTensor(tuple(new_shape))
        shape_c = ffi.new("int[]", list(tensor.shape))
        new_shape_c = ffi.new("int[]", new_shape)
        left_c = ffi.new("int[]", left)
        C.pad_double_nd(
            tensor.as_c_ptr(),
            out.as_c_ptr(),
            shape_c,
            new_shape_c,
            left_c,
            dims,
            float(value),
        )
        return out

    def topk(self, tensor: CTensor, k: int, dim: int) -> Tuple[Any, Any]:
        shape = tensor.shape
        ndim = len(shape)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")

        if k > shape[dim]:
            k = shape[dim]

        c_shape = ffi.new("int[]", list(shape))
        out_shape = list(shape)
        out_shape[dim] = k
        values = CTensor(tuple(out_shape))
        indices = CTensor(tuple(out_shape))
        C.topk_double_dim(
            tensor.as_c_ptr(),
            c_shape,
            ndim,
            dim,
            k,
            indices.as_c_ptr(),
            values.as_c_ptr(),
        )
        return values, indices

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
        # NOTES: Stub pending full CTensor infrastructure.
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

