# Turing validation diagnostic concordance

**Date:** 2026-10-17
**Title:** Validation diagnostics publish exact source identities

## Overview

A fresh Turing provenance bootstrap failed while lowering `_validate_fact` and
`identity_spill._replace_cells`.  Both failures tried to materialize `repr` for
an unclassified value inside an exception message even though native control had
already represented the terminal raise as a `ValidationBlock` and had no runtime
consumer for that message.

## Steps Taken

- Read the Turing baseline, compiler-convergence continuation/census, failure
  receipts, unit logs and scoped graph products.
- Confirmed that `Any`/`object` already use the reference/`opaque_ref` ABI and
  that `opaque_ref` deliberately does not provide generic native formatting.
- Added exact diagnostic value ids to `ValidationBlock`.
- Published those ids as cells on the existing `control_block` concordance fact.
- Changed resident glyph materialization to read that concordance and omit only
  `JoinedStr` recipes owned by compiled validation diagnostics.
- Preserved the strict missing-producer refusal for every other formatted value.
- Added direct concordance and end-to-end lowering regressions.

## Observed Behaviour

- `tests/test_glyph_sequence_provenance.py`: 57 passed.
- `tests/test_control_source.py`: 24 passed.
- Focused lexical-validation, orphan refusal and control-backend selection: 7
  passed; one unrelated benign orphan-control test retained its existing missing
  lexical-read-binding failure.
- A broader loop-composer run stopped on the pre-existing empty-`Derived`
  `formal_shape` publication for a mutable public tensor.
- Frozen resolved-unit inputs correctly cannot be replayed after a compiler
  fingerprint change.  A fresh provenance bootstrap remains the authoritative
  verification of units 038 and 045.

The subsequent fresh run confirmed unit 038 clears SSA lowering and reaches
finalization.  Unit 045 remained because its guard is nested in a retained loop:
the loop composer creates that `ValidationBlock`, while the original repair
only populated diagnostic ids in ordinary validation discovery.  Diagnostic-id
extraction is now shared by both planner lanes.  A membership-guard regression
proves the retained-loop block publishes the exact `JoinedStr` cell that the
glyph reader consumes.  The updated combined gate passes 82 tests; unit 045
awaits the next fresh run.

The unit 038 final-accounting shortfall is not the diagnostic exception call:
value 18 is the Boolean result of dynamic
`isinstance(fact, page.fact_type)`.  `Unresolved` already has explicit admitted
semantics.  The target class currently crosses ProgramABI only as `opaque_ref`,
which is transport identity rather than native subtype evidence.  The existing
Python-shell fallback regression reproduces the shortfall, and the compiler's
full-native contract forbids using its CPython ABI as the compiler-product
answer.  Do not settle this shortfall or reinterpret the opaque handle as a
type tag; the open work is a concorded native class/type-identity contract.

## Lessons Learned

The necessary general distinction is not another representation of `Any`.
There are two independent facts: the value's transport/storage representation
(`opaque_ref`) and the semantic consumer of the value.  Control planning is the
authority for the latter.  Recording diagnostic ownership there avoids both a
function-specific exemption and an invented native `repr` implementation.

## Next Steps

- Run a fresh provenance bootstrap and confirm that unit 045 advances past SSA
  lowering.  Unit 038 already clears SSA; its next boundary is the dynamic
  native class/type-identity contract described above.
- Classify any remaining non-validation unknown glyph consumers from their own
  use contracts; do not broaden the validation exemption.

See `todo/1792262097_turing_validation_diagnostic_bootstrap.stub.md`.

## Prompt History

> "examine the dirty state, understand the process, fix the failure to use the concordance that caused this"

> "do we have unresolved any handling at all or do we need to have like a reference handle object for it"

> "what do you mean those exact, like a hack? what do we need in general to identify"

Relevant workspace instruction:

> "Never \"find a better way.\" If the task names a mechanism, use that mechanism as it is spelled."

> "When something seems wrong, the system is right until proven otherwise."
