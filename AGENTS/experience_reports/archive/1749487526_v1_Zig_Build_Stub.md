# Extraction of C Source and Zig Stub

## Prompt History
- User: put zig in the speaktome toml under the ctensor group and begin the process of extracting the c code to c source files and then offering either the current method or compiling our own binaries in the venv live on first use of the c backend instead of cffi we could offer the use of precompiled binaries. If there is a fundamental problem with this plan, if it would be nearly impossible to offer either and there would be no benefit from compiling, point that out, but I suspect we might get more power if we use zig and a precompile if necessary stage of the class loading

## Notes
- Added `ziglang` as optional dependency under the `ctensor` group.
- Moved the embedded C string in `c_backend.py` to `tensors/c_backend/ctensor_ops.c`.
- `c_backend.py` now loads this file if present and supports `SPEAKTOME_CTENSOR_LIB` for precompiled libraries.
- Introduced stub `build_ctensor_with_zig` for future Zig compilation.
- Tests fail due to missing torch dependencies.
