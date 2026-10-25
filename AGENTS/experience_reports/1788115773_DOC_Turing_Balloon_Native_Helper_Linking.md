# Turing Balloon Native Helper Linking

**Date:** 1788115773
**Title:** Canonical non-fused balloon program reaches native C and LLVM

## Overview

The canonical balloon-tire Python program now lowers through
`lower_ast_source_to_ssa` and emits complete native C and LLVM modules. The
failure was not an absent helper body or a missing tensor operator table entry.
The helper and its planned region already existed in repository SSA; the
shared source-call linker could not express a caller selecting an exact subset
of a repeated multi-output `Ret`, so it left `__plan_callsite_718__` behind.

The investigation also exposed two identity/ABI defects that had been masked
by the first failure: fresh linker values did not reserve operand-only tensor
ids, and consumers of semantic PlanCall aggregate placeholders were not
rebound to the actual native aggregate container.

## Steps Taken

- Confirmed the supported source entry is `lower_ast_source_to_ssa`; retained
  `compile_ast_aot` only as a documented compatibility planning adapter.
- Audited the unresolved call record, its 39 resolved frame bindings, 20
  result bindings, and the specialized callee's 50 authored / 38 unique
  return identities.
- Added shared partial-aggregate correlations: caller ids, callee ids,
  authored return positions, and canonical native output slots.
- Updated shared aggregate analysis plus C, LLVM, and Fortran partial-output
  handling.
- Reserved every caller argument/result/operand id, graph id, and pending
  call-binding id before linker temporary allocation.
- Rebound exact semantic aggregate placeholder objects to native Call results
  and removed those placeholders from the caller ABI.
- Added native pointer-table materialization to LLVM `PointerArray` emission
  and to C aggregates consumed whole.
- Replaced `emit_balloon_tire_python_c`'s legacy fused route with the canonical
  repository-SSA entry and added the equivalent LLVM emitter.
- Added focused regression coverage and expanded
  `tools/TRANSLATION_DEBUGGING.md` with the new diagnostic signatures.

## Observed Behaviour

Before repair, C first reported three missing addresses, then one unknown
`__plan_callsite_718__`, then a 14-address cascade after partial linking. The
first address in that cascade, `%2331`, was both a real operand-only tensor and
the accidentally reused id of a later synthetic aggregate container.

After repair, the canonical module has no raw PlanCall markers, `%467` and
`%718` are absent from root arguments, C emission has zero shortfalls, LLVM
emission has zero shortfalls, and both artifacts compile to Windows DLLs.

Fortran reaches source generation but remains incomplete for an explicit
backend coverage set: `PointerArray`, imported `llvm.memcpy`,
`index_assign_double`, and `matmul_double`. These are not Python fallbacks,
autograd calls, or unresolved source helpers. Native C and LLVM already satisfy
the program's immediate native execution boundary.

## Lessons Learned

The repository is organized around several real but differently aged
contracts: ProcessGraph/hierarchy planning, repository SSA, shared aggregate
ABI analysis, and backend tables. Their entrypoints are physically far apart,
and older compatibility functions remain prominently named, which makes it
easy to mistake a historical adapter for the product route. The reliable
method is to follow the artifact: authored PlanCall record, linked Call,
aggregate projections, shared ABI analysis, then backend emission.

A late “unknown helper” can mean the opposite of a missing helper: the helper
may be completely lowered while the linker's result convention is incomplete.
Likewise, a backend address cascade should be reduced to its first missing
value and audited by object identity before changing scheduling or storage.

The correct abstraction for repeated/partial tuple returns is an explicit
four-way correlation, not positional guessing. The correct abstraction for a
fresh value is “unoccupied across every presently authored occurrence,” not
“larger than current instruction results.”

## Next Steps

- Bring Fortran's `PointerArray` and imported repository-kernel call table to
  parity without weakening the shared SSA or treating ordinary buffers as
  pointer tables.
- Run the canonical native artifact through the vehicle runtime buffer binder
  and qualification/drive loop; compilation is now complete, runtime physics
  equivalence remains the next gate.

## Prompt History

> "Don't make it fused"

> "right, you're fixing the previously bespoke validator to be a proper product of this repo, proceed"

> "missing operators is usually a table not entirely filled, there's a backend tensor compat table somewhere"

> "LITERALLY, HOW IS LLVM OR FORTRAN DOING IT. THEY WORK. WHAT IS YOUR PROBLEM."

> "when you finish and it works document your experience in the decision tree and summarize your findings on the confusing nature of repo organization"

> "sounds like the problem is not lowering the helper from ssa"
