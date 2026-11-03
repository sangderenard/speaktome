# Turing native C-shell I/O handoff

The adjacent `turing` repository now generates a working dependency-free
Windows display shell around emitted whole-program Fortran. Start with:

- `src/compiler/fortran_c_shell.py`
- `src/compiler/shell_io.py`
- `src/compiler/python_native_shell.py`
- `tests/test_fortran_c_shell.py`
- `build/columnar_fortran_c_shell/columnar_multifluid_native.exe`
- `build/columnar_fortran_c_shell/verification.json`

## Proven boundary

`ShellIOManifest` binds the final ABI parameters. For the current demo the
required capability is `display_double_buffer`, the format is
`rgb_f64_planar`, and the resources are `display.red`, `display.green`, and
`display.blue`. The generated C shell owns conversion to a top-down 32-bit DIB
and presents through `StretchDIBits`. It does not inspect application names.

The compiler-selected control entry must be called. Do not substitute the
numerical entry to make a demo run once. The fix in `ssa_fortran_backend.py`
ensures the API sidecar describes transitive extents on the final control
signature.

## Work remaining

1. Add a native SPSC input-event ring using `ShellIOABI.input_events`.
   Translate Win32 keyboard, pointer, wheel, and close messages into the shared
   record fields; do not call back into Fortran from the window procedure.
2. Add the native file broker using `file_requests` and `file_completions`.
   Keep handles shell-owned and paths/payloads as offset-length spans.
3. Support packed `rgb8`, `rgba8`, and `indexed_float32` display bindings.
4. Introduce a stable native build entrypoint which accepts a compiled module,
   manifest, state-feedback table, specialization extents, and initial state.
5. If presentation is moved to another thread, retain explicit front/back
   generation ownership and measure compute completion separately from blit.

## Verification baseline

Run from `C:\dev\Powershell\turing`:

```powershell
python -m pytest -q tests/test_fortran_c_shell.py tests/test_fortran_control_target.py tests/test_fortran_fidelity.py tests/test_ssa_fortran_and_optimizing_llvm.py tests/test_process_graph_shell.py -x
```

Baseline: 34 passed. The full-size fidelity report records 30 matching output
arrays, 45,056 elements per plane, and maximum absolute error
`2.842170943040401e-14`.

