# Turing rational feature timing through AbstractTensor

**Date:** 2026-09-24
**Title:** Repeated-division cost table across the feature lattice

## Summary

Added `turing/tools/benchmark_rational_features.py` and recorded a reproducible
run in `turing/docs/RATIONAL_FEATURE_TIMING.md`.

The workload applies six nonzero tensor divisions to 256 elements and observes
the value once at the end. It covers all eight subsets of complex, rational,
and precision, separating construction, division-chain, and final-collapse
time. Results are checked against an ordinary AbstractTensor evaluation of the
same expression.

The first draft directly instantiated `NumPyTensorOperations` and assigned its
`.data`; this bypassed AbstractTensor construction and was rejected. The final
tool contains no backend-class import or storage access. It selects `numpy`
through `AbstractTensor.use_backend("numpy")`, constructs exclusively with
`AbstractTensor.get_tensor`, and performs the workload exclusively with the
public overloaded operators. The rational test helper was corrected in the
same way.

Attempting the unspecialized default path resolved to Nodus and exposed that
native `complex128` operations currently raise `NodusUnsupported`; therefore
the full feature lattice is measured on the canonical backend selected through
the abstraction, rather than by bypassing the abstraction.

## Verification

```text
py -3.11 -m pytest -q tests/test_rational_precision.py
11 passed in 15.57s

py -3.11 -m py_compile tools/benchmark_rational_features.py tests/test_rational_precision.py
```

Measured with:

```text
py -3.11 tools/benchmark_rational_features.py --backend numpy --size 256 --steps 6 --limbs 2 --warmups 1 --repeats 3
```

The full table is in `turing/docs/RATIONAL_FEATURE_TIMING.md`. On this eager
run, real Rational and Precision[2] had similar chain time (155.116 ms and
161.131 ms); RationalPrecision[2] took 416.972 ms plus 25.257 ms to collapse.
The complex+rational combinations were the most expensive current paths.

## Existing working-tree state

Unrelated pre-existing edits in
`src/transmogrifier/graph/graph_express2.py` and
`tests/test_abstract_tensor_topological_reducer.py` were not modified.

## Next Steps

- Keep this expression and measurement boundary fixed while reducing repeated
  full-tensor limit inspection and adding compiler lowering.
- Make the documented Nodus complex fallback effective before using Nodus for
  the complete feature-lattice benchmark.
- Add exact common power-of-two rational rebalancing.

## Prompt History

> can we make any kind of timing table on some task through the various features so we can maybe get an idea of the cost of things

> there shouldn't be any numpy involved, where did you use numpy

> it is acceptable to use the backend. it is not acceptable to bypass your way to backend only functions, bypassing all the features of abstract tensor
