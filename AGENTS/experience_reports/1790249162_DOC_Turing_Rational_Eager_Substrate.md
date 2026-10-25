# Turing rational eager substrate

**Date:** 2026-09-24
**Title:** Checked rational composition and canonical eager promotion

## Summary

Implemented the first eager slice of the rational composition design in
`turing/src/common/tensors/extended_precision.py`.

Added:

- `RationalLimitError` and reason-carrying `RationalLimitDecision`;
- `RationalLimits` with exact `Fraction`-based eager component bounds,
  denominator-nonzero/finite proofs, dtype limits, quotient bounds, and
  collapse admission;
- `Rational`, `RationalPrecision`, `ComplexRational`, and
  `ComplexRationalPrecision`;
- structural reciprocal and numerator/denominator arithmetic with no implicit
  quotient evaluation;
- pre-operation overflow/destructive-underflow checks;
- identity-and-nonzero-proven factor cancellation before products;
- canonical feature-set union promotion for ordinary, complex, precision, and
  all rational-containing types; and
- AbstractTensor reflected dispatch for every composite wrapper.

The change also repairs the pre-existing operand-order asymmetry:
`Precision <op> ComplexPrecision` now promotes to `ComplexPrecision`, matching
the reverse order.

The eager implementation deliberately does not yet include exact common
power-of-two rebalancing or compiler wrapper lowering. Unsafe component work
that cannot cancel is refused before arithmetic. Compiler integration remains
the next substrate layer.

## Verification

New focused suite:

```text
py -3.11 -m pytest -q tests/test_rational_precision.py
11 passed in 15.62s
```

This covers basic rational algebra, repeated integer division against
`Fraction`, special/zero refusal, pre-operation overflow refusal,
structurally-valid-but-not-collapsible values, cancellation before overflow,
the repaired precision/complex order, every ordered pair across the empty and
seven non-empty feature sets for all four basic operators, and delayed wide
division in `RationalPrecision`.

Existing eager sister surfaces:

```text
py -3.11 -m pytest -q tests/test_precision_surface.py tests/test_complex_precision.py -k "not source_compiler and not torch"
34 passed, 2 deselected in 21.01s
```

The broader `tests/test_precision_pipeline.py` result was 30 passed and 4
failed. A clean detached Turing worktree at HEAD reproduced both failure
families checked: the live-wrapper test expects `exp()` to reject an argument
that the current implementation range-reduces, and the source compiler test
omits the now-mandatory extraction contract. These are pre-existing. The
temporary detached worktree was removed after comparison.

## Existing working-tree state

Unrelated pre-existing edits in
`src/transmogrifier/graph/graph_express2.py` and
`tests/test_abstract_tensor_topological_reducer.py` were not modified.

## Next Steps

- Implement exact common power-of-two rational rebalancing under proved
  exponent windows.
- Add direct source descriptors and component/limit identity receipts.
- Lower complex composition, then rational structure, then reuse the existing
  precision transaction.
- Add call, return, and native execution regressions for every rational
  composite.

## Prompt History

> begin implementation
