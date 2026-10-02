# Turing runtime iterable-bound retention

## Work

Fixed the retained-loop composer in `turing/src/compiler/loop_composer.py`.
A `for` loop's runtime bound is owned by its iterable identity, but the
composer additionally required the target to publish one or more value
bindings.  When normalization legitimately produced no target binding, the
body regions were scheduled and the owning loop was refused as
`unresolved-loop-bound` even though the established
`__iterable_extent_<id>__` lowering was available.

The blocker now means exactly that neither a stop value nor an iterable
identity/constant exists.  The runtime iterable-extent spelling no longer
depends on target bindings.  No new loop representation or fallback was
added.

Added a public-entry regression using an empty destructuring target, which
has no target binding but retains a runtime iterable.  It asserts that the
lowered function contains the complete loop CFG and that its extent operation
reads an authored parameter identity.

## Verification

- `python -m pytest tests/test_orphaned_loop_refusal.py -q`: 4 passed.
- `python tools/audit_identity_concordance.py mapping`: 24 rows across two
  functions, zero findings.
- Broader loop/control/SSA batch: 173 passed, 17 failed.  An untouched HEAD
  worktree produced the identical 173/17 result, so no new failure was
  introduced by this repair.
- `git diff --check`: clean apart from the repository's Windows line-ending
  notices.

The user's generated `build/woodshop/_outer/_native_probe.py` was not present
in the working tree, so that exact command could not be rerun.  The focused
reproduction reaches the same `CompilationSubdivisionRequired` blocker before
the repair and lowers through the sanctioned public compiler entry after it.

## Prompt History

User:

> I need you to fix the compiler, it's fussy about unknown loop bounds despite that being it's job any time there isn't a way to tell how to finish a loop, just leave it a fucking loop

The user also supplied the traceback ending in
`blockers=('unresolved-loop-bound',)` for Woodshop loop node 174.
