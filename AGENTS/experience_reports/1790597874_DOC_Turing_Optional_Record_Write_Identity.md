# Turing optional record write identity

## Work

Repaired two related receiver-field classification losses in the Turing
compiler without running the Woodshop outer-native probe.

`_field_slot_ops` previously recognized a constructed record stored in a
receiver field only when the immediate value carried `class_ref`. Specialized
call results carry the established `result_class_ref` fact instead. The field
coordinator now accepts either exact record-instance fact, so a specialized
`Metrics` value is retained as a nested record rather than degrading to an
untyped scalar slot.

The scalar-slot precompiler also treated a ProgramABI `reference` field as a
scalar when the method did not contain a static-reference literal write. It
now derives reference slots from the declared storage contract as well as
static-reference writes, gives those slots `opaque_ref` physical dtype, and
keeps them out of scalar dtype completeness validation.

Added graph-level and SSA-level regressions for both cases. Existing optional
record presence/payload behavior remains the representation; no Python object
boxing or fallback runtime was introduced.

## Verification

- Focused record-identity, declared-reference, and optional-presence batch:
  6 passed.
- `python tools/audit_identity_concordance.py mapping`: 24 rows across two
  functions, zero findings.
- `git diff --check`: clean apart from Windows line-ending notices.
- `tests/test_native_optional_record_constructor.py`: 3 failed, 1 passed on
  both the modified tree and an untouched HEAD worktree. The identical
  failures are the existing `id_scale` audit finding and are unrelated to
  this repair.

The user explicitly reserved the Woodshop probe run for themselves, so it was
not executed.

## Prompt History

User:

> the shape of the fix must transparently and without sacrificing optimization, recognize this situation. i'm not certain it's as easy as it might seem so lets think first before we act

User:

> what if we made none type able to be uninitialized none with type

User:

> how complicated is the contract for this trial script? also what do you feel is the best industry standard enough and elegant enough way to resolve this

User:

> try it out, don't run the script though I'll do that myself
