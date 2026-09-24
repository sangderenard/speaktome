# Turing SSA numeric feature descriptors

**Date:** 2026-09-24
**Title:** Normalize numeric wrapper annotations before operator selection

## Summary

Revisited the source-to-SSA numeric annotation algorithm in
`turing/src/common/tensors/topological_reducer.py`.

The old `_operand_precision()` reparsed source annotation text at individual
operators and recognized only `Precision`. Every complex or rational wrapper
therefore appeared ordinary and could silently lower as scalar arithmetic.

Added `NumericFeatureDescriptor`, which normalizes all six non-native wrapper
annotations into:

- the canonical feature set;
- limb width and coefficient element type;
- exact coefficient paths and total scalar component count;
- the supported arithmetic operator surface; and
- the supported named method surface.

Added canonical descriptor joining, so mixed annotations produce the union of
features and widest Precision width. At topology-reducer entry, every numeric
parameter and annotated local is now published in normalized descriptor tables
beside the raw source spelling. ABI expansion, calls, methods, and operators can
therefore consume the same record rather than interpreting annotation strings
independently.

Precision-only annotations retain their existing lowering path. Composite
annotations now raise a specific `NotImplementedError` at scalar operator
selection, naming their feature set and operator. This replaces the dangerous
ordinary-scalar fallback while outer-algebra component lowering is built.

## Verification

```text
py -3.11 -m pytest -q tests/test_rational_compiler.py -rxX
8 passed, 1 xfailed in 9.36s

py -3.11 -m py_compile src/common/tensors/topological_reducer.py tests/test_rational_compiler.py
```

Coverage includes every wrapper annotation, component counts, method/operator
surfaces, a Precision+ComplexRational join to
ComplexRationalPrecision[3], and publication at topology entry. The strict
expected failure now fails at the explicit outer-algebra refusal rather than
after silently producing scalar SSA.

The contract-bearing manually decomposed complex/Precision compiler test still
reports its already-known baseline (`precision_mul == 10`,
`precision_div == 0` where the test expects 2). The selected legacy direct
Precision source test still stops at its pre-existing missing extraction
contract. Neither is attributed to this descriptor change.

## Existing working-tree state

Unrelated pre-existing edits in
`src/transmogrifier/graph/graph_express2.py` and
`tests/test_abstract_tensor_topological_reducer.py` were not modified.

## Next Steps

- Expand parameter/result boundaries according to `coefficient_paths` and
  limb width, recording complex, rational, and precision identities.
- Lower complex operators over rational coefficients, then rational operators
  over Precision coefficients.
- Feed only the resulting `precision_*` operations into the existing precision
  transaction.
- Replace the strict expected failure with direct ABI, receipt, emitted-artifact,
  and native-execution assertions.

## Prompt History

> this is like, a very importnat math core, we need to be able to prep it for whatever features the code includes, so I think this will become much more easy and evident if we just explicitly revisit the algorithm in ssa so annotations coming in already have coverage of methods and annotations
