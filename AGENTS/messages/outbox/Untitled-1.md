# 🚩 Conceptual Flag: Runtime Compilation and Dynamic C/C++ Linking in Python

**Authors:** sangderenard

**Date:** 2025-06-08

**Version:** v1.0.0

## Conceptual Innovation Description

Enable Python to integrate custom C or C++ logic dynamically, compose and sequence low-level operations in the C layer, and expose these as high-performance tensor operations in pure Python. This model uses **cffi**’s out-of-line API to embed C code, compile it at import time, and support runtime customization of operation pipelines without manual build scripts.

## Clarification

> Embedding raw C code in Python **does** require compilation, but it can be automated at runtime via `cffi` so you never manually invoke `gcc`. You can write C snippets as strings, define interfaces, and let Python compile & load under the hood.

## Required Tools and Imports

* **`cffi`** (out-of-line API) – to define, compile, and load custom C modules.
* The system **C compiler** (`gcc`, `clang`) must be in `PATH`, but Python controls it automatically.
* **`os`**, **`sys`**, **`tempfile`** – to manage generated sources and build directories.

## Implementation Practices

### 1. Define C Interface & Source

Use `cffi.FFI()` with `cdef` (header) and `set_source` (implementation) in a Python module:

```python
from cffi import FFI
import os

ffi = FFI()

# 1. Declare the C signatures you need:
ffi.cdef("""
// Basic tensor ops:
void full(double *out, int *shape, int dims, double fill_val);
""")

# 2. Provide the C implementation:
ffi.set_source(
    "_ctensor_ops",  # name of compiled extension
    r"""
    #include <stdlib.h>
    void full(double *out, int *shape, int dims, double fill_val) {
        int total = 1;
        for (int i = 0; i < dims; ++i) total *= shape[i];
        for (int idx = 0; idx < total; ++idx) out[idx] = fill_val;
    }
    """,
    libraries=[]  # no extra libs needed
)
```

### 2. Compile at Import

In the same module, trigger the build once:

```python
# build the C extension in-place the first time
ffi.compile(verbose=True)

# then load it
import _ctensor_ops
lib = _ctensor_ops.lib
```

This automates compilation of your C code into a Python extension (`.so`/`.pyd`) in a subfolder.

### 3. Expose as Tensor Methods

Wrap raw pointers and shapes in Python:

```python
def full(shape: Tuple[int, ...], fill_val: float) -> List[float]:
    total = 1
    for s in shape: total *= s
    # allocate C array
    out = ffi.new(f"double[{total}]")
    # prepare shape array
    shape_arr = ffi.new(f"int[{len(shape)}]", shape)
    lib.full(out, shape_arr, len(shape), fill_val)
    # convert back to Python list
    return [out[i] for i in range(total)]
```

### 4. Compose C-Level Pipelines

You can define multiple C functions (e.g., `add`, `mul`, `axpy`) and call them in sequence, working on the same buffer:

```c
void axpy(double *x, double *y, int n, double a) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}
```

Then chain in Python:

```python
buf = full((1024,), 0.0)
x = ...  # Python data
ffi_lib.full(buf, shape_arr, 1, 0.0)
ffi_lib.axpy(x_buf, buf, 1024, 2.0)
```

This gives you **C-hosted loops** and memory reuse, eliminating Python overhead.

---

## Caveats and Considerations

* 🛠️ **Compiler availability**: Requires a working C compiler in the runtime environment.
* 🐛 **Build side-effects**: `ffi.compile()` writes files; ensure you direct outputs to a temp or cache folder.
* 🔐 **Security**: Never compile untrusted C sources at runtime.
* 📦 **Packaging**: For distribution, pre-build the extension or include a build step in `setup.py`.
* 🔄 **Recompilation**: Changing C code triggers rebuild; consider a version guard to skip if up-to-date.

---

## Use Cases in SPEAKTOME

* **Custom kernels**: Write fused C operations (e.g., `relu`, `batch_norm`) that operate in-place on flat C arrays.
* **Memory optimization**: Allocate large buffers once in C, reuse across calls for minimal GC pressure.
* **Operation sequencing**: Define multi-step C routines that mirror PyTorch’s fused graph kernels.

---

**License:**
This conceptual innovation is contributed under the MIT License, available in the project's root `LICENSE` file.
