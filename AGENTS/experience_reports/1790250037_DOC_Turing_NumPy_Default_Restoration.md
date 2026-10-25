# Turing NumPy default restoration

**Date:** 2026-09-24
**Title:** Make Nodus opt-in at the AbstractTensor backend boundary

## Summary

Changed `AbstractTensor.check_or_build_registry()` so its implicit backend
order is now:

```text
numpy, nodus, torch, pure_python
```

It was previously `nodus, numpy, torch, pure_python`. The
`ABSTRACT_TENSOR_BACKEND` environment override, `use_backend()` scope, and
`set_default_backend()` behavior are unchanged, so Nodus remains available as
an explicit backend.

Updated `tests/test_nodus_backend.py` to assert that connecting the Nodus arena
does not change the default. Tests that prove Nodus arena execution now select
Nodus through `AbstractTensor.use_backend("nodus")`, keeping their existing
backend-specific coverage without relying on an implicit global preference.

The rational timing document now describes its earlier Nodus discovery as
historical context. Its recorded command still selects its backend explicitly
for reproducibility.

## Verification

```text
py -3.11 -m pytest -q tests/test_nodus_backend.py tests/test_backend_scope.py
14 passed in 1.28s

py -3.11 -m pytest -q tests/test_precision_surface.py tests/test_complex_precision.py tests/test_rational_precision.py -k "not source_compiler and not torch"
45 passed, 2 deselected in 36.40s

ABSTRACT_TENSOR_BACKEND unset; AbstractTensor.get_tensor([1.0])
NumPyTensorOperations
```

## Existing working-tree state

Unrelated pre-existing edits in
`src/transmogrifier/graph/graph_express2.py` and
`tests/test_abstract_tensor_topological_reducer.py` were not modified.

## Next Steps

- Begin rational descriptor and compiler lowering work from the canonical
  AbstractTensor/Precision substrate.
- Retain explicit Nodus tests and repair its unsupported complex fallback
  independently of the process default.

## Prompt History

> can you find the hard coded nodus default and change it to numpy, then we'll start iterating on compiling
