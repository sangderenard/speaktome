# Turing Woodshop Newton concordance reaches zero

**Date:** 2026-09-27
**Title:** Final identity transitions close every Woodshop Newton finding

## Scope

Continued compiler regulation from Turing commit `b8d47465`. The concrete goal
was to compile the Woodshop Newton dt-system through the sanctioned source
compiler and LLVM backend with no identity-concordance findings.

## Methodology

Read the existing regulation report, reproduced the real Newton lowering, and
inspected the shared identity-book pages, final function metadata, record and
sequence descriptors, SSA definitions, formals, and uses for every remaining
finding. Each change was guarded by a small identity-table regression before
the full Woodshop compile was repeated.

## Detailed Observations

The starting audit had 13 findings: nine missing alias targets and four
sequence-descriptor member findings. Four missing targets came from treating
semantic output history as physical storage. The output page was already the
authority and already had a separate durable-agreement audit.

Two duplicate sequence descriptors had exact planning aliases from their
nonphysical handles to resident handles and identical arena columns. Their
only differing members were private extent/status cells used solely by dead
compiler bookkeeping. Consuming the existing descriptor-role relation retired
the duplicate rows and updated the record-table and formal-accounting views.

Four source projections had already advanced to provisional aggregate slots
when positional result legalization selected emitted output slots. Advancing
only the authored field occurrence left the provisional terminal behind. The
positional proof now advances both identities.

The last planning edges named occurrences absent from final SSA and every
surviving descriptor. Finalization now tombstones only terminal dead edges,
preserving their page history and emitting explicit retirement receipts.

The measured progression was 13 → 9 → 4 → 0 findings. The final module has
6,629 audited rows across 445 functions.

## Verification

```text
Woodshop Newton LLVM compile:
identity concordance: 6629 rows across 445 functions, 0 finding(s)
  id groups: legacy=2668, minted=3961

Focused sequence/identity tests:
4 passed, 15 deselected in 2.49s

Focused linking tests:
2 passed; the third reached and passed its new concordance assertion, then
failed the pre-existing minted-id-below-1e9 assertion.

Complete sequence-concordance file:
17 passed, 2 failed. The two keyed-tensor planned-region failures reproduce
identically at b8d47465 in a clean C:\tb8 worktree.
```

## Recommendations

- Keep semantic output history distinct from physical storage aliases.
- When exact aggregate legalization replaces a slot, advance the already
  concorded terminal as well as the authored occurrence.
- Retire descriptor aliases only through exact handle/column and role proofs;
  do not merge by spelling, dtype, or numeric proximity.
- Preserve page tombstones and retirement receipts so final cleanup remains
  auditable rather than becoming silent metadata deletion.

## Prompt History

> b8d47465 <- this commit should include a report of some kind on the progress of fixing the compiler, with the goal of being able to compile the woodshop's newton block with no reports from the concordance

> in turing sorry not the root
