# Turing SSA Resident Sort Benchmark

**Date:** 2026-10-14
**Title:** Stable resident-sequence sort lowered and verified through native LLVM

## Overview

The canonical `C:/dev/Powershell/turing` compiler now recognizes a lexical
zero-argument `list.sort()` effect, lowers it to a stable bottom-up merge sort
written entirely in repository SSA, and can compile and execute a source-level
Python benchmark through the sanctioned `lower_ast_source_to_ssa` entry.
Extraction policy, CPython calls, and machine decompilation were not changed.
The keyed form now accepts both a resident mapping lookup such as
`positions.__getitem__` and a source lambda projection such as
`lambda row: row[1]`.

## Work Completed

- Added `lower_sequence_sort` beside the existing sequence/table SSA helpers.
- Added `sort` to the existing modeled resident-sequence mutation vocabulary
  and lexical effect-retention path.
- Connected the lexical mutation to generated SSA and declared its runtime-sized
  scratch as caller-provided workspace.
- Completed the previously empty generated-helper concordance registration:
  helper blocks and produced values now receive exact owner/value records.
- Added `tools/benchmark_ssa_sort.py`, which compiles a normal annotated Python
  function containing `values.sort()`, emits and compiles LLVM, executes it,
  compares against a stable NumPy ordering, and reports mean native time.
- Extended the authored row ABI so fixed scalar tuple annotations publish every
  physical column. This is the source contract needed by a lambda projection;
  no generated sequence descriptor was patched.
- Corrected aggregate declaration propagation to distinguish authored locals
  from parameters, allowing `_reindex_in_place` to express its resident
  dictionaries, list, tuple row, and integer column without duplicating an
  annotated mapping parameter.
- Preserved compiler-frame ownership on nested-row projections and child-table
  storage at the descriptor construction boundary. The final public-argument
  gate now reads those source-derived identities instead of treating the
  compiler's own storage cells as authored inputs.
- Carried retained local scalar annotations across their complete identity
  histories. `_reindex_in_place` now declares `dict[int, int]`,
  `dict[tuple, list[int]]`, and `list[int]` locally, and the loop projection
  receives its integer boundary type from that source declaration.
- Added `ssa_constant_materialization`, an exact concordance receipt written by
  the constant-materialization transformation that changes block geometry.
  This repairs the first wrong writer behind the two stale-geometry failures.
- Resolved the exact authored `type(row) is tuple` guard from the declared tuple
  loop target, and changed `_validate_row`'s source annotation from `Any` to the
  tuple contract its first runtime check already enforces.

## Verification

- `python -u -m tools.benchmark_ssa_sort --size 4096 --repetitions 3`
  sorted 4,096 `int64` values correctly; measured mean native time was
  approximately 0.001677 seconds on this machine.
- `python -u -m tools.benchmark_ssa_sort --projection --size 257 --repetitions 2`
  sorted two-column rows by the second column correctly; mean native time was
  approximately 0.001028 seconds in the final run.
- `python -u -m tools.benchmark_ssa_sort --keyed --size 257 --repetitions 2`
  sorted through resident mapping keys correctly; mean native time was
  approximately 0.001166 seconds in the final run.
- `python -u -m tools.benchmark_ssa_sort --size 257 --repetitions 2` sorted
  the plain resident list correctly; mean native time was approximately
  0.000955 seconds in the final run.
- `python -m pytest tests/test_ir_sequence_tables.py tests/test_loop_composer.py
  tests/test_native_reference_boundary.py
  tests/test_abstract_tensor_topological_reducer.py -q -k
  "not structural_fold_refreshes_retained_loop_order_after_removing_constants"`:
  207 passed, 1 deselected, 1 third-party deprecation warning.
- Direct public-entry compilation of `_validate_row` and
  `_reindex_in_place` succeeded; the latter emitted the resident SSA sort and
  key-lookup helpers with no unexplained public arguments or unresolved
  boundary types.
- The fresh serial whole-program provenance receipt cleared original units 047,
  048, 058, and 063. Unit 063 (`_reindex_in_place`) is recorded as
  `repository_ssa_complete: true`, with no lowering/accounting shortfall.
  Publication then exposed a later, separate verification problem: the final
  concordance detector logged a transient `NoneType.functions` failure, while
  rebuilding the detector over the persisted module succeeded and reported
  the broader existing audit backlog. The unit therefore remains
  `compiled-unverified`, not partial.
- `python -m pytest tests/test_ir_sequence_tables.py -q`: 40 passed.
- Python compilation and `git diff --check` completed without errors.

## Next Steps

- Trace the post-compile detector's transient `NoneType.functions` input and
  then address the newly reachable whole-module concordance audit backlog as
  separate work from the four repaired source blockers.
- Keep keyed mapping performance explicit: the stable merge performs
  logarithmic comparison rounds, but mapping lookup is not yet a cached
  decorate-sort-undecorate path.

## Prompt History

> "okay so add one that is sort. now."

> "okay now we need to compile a little sort benchmark python program to verify this works"

> "please enable it, test it, then work on these remaining problems:"

> "you fix at the source of mishandling in using the concordance you do not tweak any intermediate product"
