# Turing formal-consumer provenance

**Date:** 2026-10-20
**Title:** Provisional SSA formals retain planner concordance receipts

## Overview

Continued the bootstrap work from Turing compiler-convergence report 22. Unit
034 previously exposed two unexplained public formals and an unresolved starred
tuple return because downstream consumers did not read the planner's
`executable_node` facts.

## Steps Taken

- Read the workspace, Turing, testing-hazard, report 22 and related provenance
  guidance before editing.
- Reproduced the starred variadic tuple return through the sanctioned compiler
  entry.
- Rejected and reverted a false-green classifier experiment after inspecting
  its emitted SSA: the slice was treated as one scalar.
- Made the provisional-formal collector read `executable_node` and retain the
  exact source node, construct, text, span, verdict and rule.
- Preserved receipts in function metadata so sequence frame expansion cannot
  erase them.
- Joined unexplained-public-formal audit details and native-publication refusal
  messages to those receipts.
- Added regressions for the exact starred return and unit-034-shaped audit row.

## Observed Behaviour

- Native-formal provenance tests: 12 passed.
- Focused provenance/project/precompile gate: 10 passed, 230 deselected.
- A targeted direct compilation stopped earlier on a pre-existing missing
  source identity row for `IdentityPage._resolved` node 78, so it did not
  validate unit 034's final lane.
- The subsequent user-authorized normal bootstrap produced fingerprint
  `3c1b8a2900c4c0e207bd047aa912370655102cb259ef6afb37b6905708e9a172`.
  Its fail-fast stage admitted 29/330 units: 26 compiled-unverified, 3 partial,
  0 failed and 0 verified. Unit 034 retained the known variadic tuple gap and
  now carries the exact concordance receipt. Concurrent workers also exposed
  newly catalogued partial units 081 (`_result_layout`) and 115
  (`_source_page_declaration_findings`). See report 23 for their exact
  shortfall inventories and the qualification that they are not proven new
  regressions.

## Lessons Learned

Planner classification was already correct and already recorded. The missing
feature was a consumer read, not a new classification. Receipts must survive
physical ABI expansion independently of transient SSA objects. Also, marking a
starred tuple wrapper executable is not a lowering; inspecting emitted SSA is
necessary to reject such false greens.

## Next Steps

- Extend the same read-before-use rule to return slots, region feeds and display
  leaves.
- Build the real variadic tuple sequence lane with resident clear, append and
  append-slice operations.
- Trace unit 081 through the incumbent compiler-object ABI and partition unit
  115 by the existing Identity Book/Page/Ref, string/tuple and generator
  contracts. Do not introduce substitute object machinery.
- Keep tuple-valued mapping work atomic across reducer, ABI and helpers.

See `todo/1792510251_turing_variadic_tuple_and_provenance_consumers.stub.md` and
`turing/docs/compiler_convergence/23_formal_consumer_provenance_continuation_2026_10_20.md`.

## Prompt History

> "I did not fully trust the agent who did the last section of work, but here we
> are, trying to move the needle on the bootstrap."

> "can you do any of the rest? did i not send you a path to a continuation report"

> "can you run the bootstrap for me and catalog the new problems"

Relevant workspace instruction:

> "Never \"find a better way.\" If the task names a mechanism, use that mechanism
> as it is spelled."
