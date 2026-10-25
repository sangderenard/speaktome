# Turing composite compiler probe

**Date:** 2026-09-24
**Title:** ComplexRationalPrecision is silently erased before SSA lowering

## Summary

Compiled a minimal repeated-division function annotated with
`ComplexRationalPrecision[2]` through the sanctioned
`lower_ast_source_to_ssa` entry and a declared scalar extraction contract.

The compiler currently accepts the source but treats both composite arguments
as ordinary scalars. The planned region has two arguments, two ordinary `Div`
instructions, and one scalar output. It has no precision pipeline receipt and
no complex, rational, or precision component identity metadata.

**Follow-up:** `1790250828_DOC_Turing_SSA_Numeric_Feature_Descriptors.md`
replaced this silent fallback with normalized source descriptors and an
explicit refusal at the outer-algebra seam. The observations below describe
the probe before that repair.

For two width-two `ComplexRationalPrecision` inputs, the direct wrapper ABI
requires sixteen scalar input components: two complex coefficients, two
rational components per coefficient, and two precision limbs per component.
The returned composite requires eight components.

Added `tests/test_rational_compiler.py` as a strict expected-failure regression
that pins those boundary counts and the three required component-identity
records. It will become an XPASS failure if only the marker is accidentally
satisfied; the marker should be removed when the actual lowering lands.

The existing complex compiler coverage is not direct composite lowering. It
manually authors four `Precision[2]` parameters (`ar`, `ai`, `br`, `bi`) and
therefore proves the inner precision substrate rather than ComplexPrecision
wrapper ingestion or ABI expansion.

## Verification

```text
py -3.11 -m pytest -q tests/test_rational_compiler.py -rxX
1 xfailed in 3.85s
```

The observed unlowered region was:

```text
args: 2
ops: Div, Div, Ret
precision receipt: absent
precision_lowered_values: absent
```

## Next Steps

- Add authored feature-set and limb-width descriptors at the source boundary.
- Expand direct composite parameters/results into complex identities, then
  rational numerator/denominator identities, before the precision pipeline.
- Lower complex algebra over rational coefficients, lower rational algebra
  over precision coefficients, and feed the resulting precision operations
  into the existing transaction.
- Refuse unsupported composite sections loudly during the transition; never
  retain the current ordinary-scalar fallback.

## Prompt History

> before we compile did we try complex precise rational

> let's compile a little test and see the obvious failing then we'll work on the ssa engine for these type combinations presently only serving precision and complex maybe
