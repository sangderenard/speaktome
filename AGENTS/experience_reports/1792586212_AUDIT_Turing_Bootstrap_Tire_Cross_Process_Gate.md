# Turing bootstrap/tire cross-process gate audit

## Later authorization and work in progress (2026-10-21)

The audit below is historical, not the current authorization state. The user
subsequently explicitly requested a goal: implement the three proposed
provenance-core additions first, then advance BOTH full compilations. The active
goal covers recursive structural views, boundary transport receipts, and exact
revision-dependent truth maintenance in the existing IdentityBook/concordance.
Compiler source and tests have now been edited in the canonical Turing repo.
The original finding that the two terminal errors differed remains true; it
must not be rewritten as proof that those errors were identical.

Current core verification: 157 tests passed across structural concordance,
call-input adapters, and precompile-to-SSA. Broader optional/audit tests were not
all green. The first full pair did not advance: bootstrap 4/337 admitted
(3 compiled-unverified, 1 partial), with a return-merge self-use and two stale
aggregate-to-scalar feeds; tire again failed at region24 formal295, float64
despite its bool physical storage and exact returned-value origin. Both expose
consumer views surviving producer publication. Full evidence and subsequent
core integration are recorded in Turing report32. The second paired run has
finished; neither result demonstrates correct progress. Prior generated run
artifacts were removed after preserving these receipts; fixes are source edits.

### Additional prompt history

> can you study until you can repair it right what am i failing to ask for that justifies limited to no progress all month and the only thing, the one and only thing we need, is for the algorithm to not be so fucking bad it has no claim to order at all

The second bootstrap labels unit027 compiled-unverified, but its saved Ret
still drops the second tuple member and optional presence. This is not a
semantic success. It stops at unit028's missing nested record descriptor.
The paired tire failed in the new transport checker before the old immutable
input check: Metrics.error_present's leased SSA fact has shape (), while its
actual producer value183 and record descriptor establish a (15,) span.
The new core's SSA-only physical view is incomplete. Subsequent work traced
publication, priority, and acceptance ordering without another compiler edit.
The terminal errors are not identical; shared incomplete publication is not
permission to synchronize from an arbitrary mutable consumer. Report32 retains
the exact source/result IDs, exceptions, and bounded complexity observations.
Second-run cleanup was attempted with validated in-repository paths, but the
execution policy rejected the PowerShell deletion command before execution.
Those generated artifacts remain; do not claim cleanup succeeded. Guestbook
validation passed without applying the suggested archival operations.

> when you keep reinforcing that the caches have one location you're reinforcing in the conversation the implicit idea that your fixes are landing in the cache files which would be a flagrant rejection of your task and regression

> list three structural pieces that, if we added them to the provenance core, we would see the greatest simplification of problems and elimination of all the problems that keep cropping up somewhere any time elsewhere they are fixed because the surface is too large for you to conceive of

> set a goal to get farther in both compilation processes AFTER FIRST implementing each of your identified desires as part of the provenance core closely associated with the concordance and existing machinery

---

**Date:** 2026-10-21
**Title:** Bootstrap and managed-tire failures are not yet the same failure

## Scope

Audit the canonical whole-compiler bootstrap and the complete managed
balloon-tire build as a coupled pair. No compiler repair was permitted unless
the same failure was demonstrated in both processes.

## Methodology

- Read `C:/dev/Powershell/turing/AGENTS.md`, the current opening receipt in
  `TEST_BASELINE_AND_HAZARDS.md`, and compiler-convergence report 31.
- Ran the canonical bootstrap in its default reusable output location:
  `python -u -m tools.bootstrap_compiler_concordance --lower-only`.
- Ran the complete managed tire in its default reusable output location:
  `python -u tools/build_balloon_tire_native.py --managed-dt`.
- Continued the same cached bootstrap census with
  `--continue-after-failure` so all 331 real source units were attempted.
- Compared per-unit bootstrap worker failures with the tire's full identity
  history and physical-call conflict.
- Removed the reusable bootstrap cache, three compiler logs, and all 162
  identity logs created during this audit. No source or test file was edited.

## Detailed Observations

The first bootstrap gate admitted 21 of 331 units: 20 compiled-unverified and
one partial. Unit 034, `IdentityBook._virtual_edges_into`, remained partial.
Its exact shortfalls were the unresolved tuple source value for
`(origin, *ref.row[1:])`, unexplained public values 19 and 13, and the
consumer refusal for value 19.

The managed tire lowered all 41 shells, the full 396-region
`balloon_tire_vector_step`, and its 369-function local repository with zero
shortfalls. Whole-program call-frame settlement converged after six rounds;
physical-call adaptation converged after two rounds. Final lowering then
refused one immutable storage mismatch: the returned
`Metrics.hard_failure` value was `bool`, while
`step_with_dt_control_used__planned_region_24` formal 295 remained
`float64`.

The complete bootstrap census attempted all 331 units. Saved receipts included
50 compiled-unverified and 55 partial units; other workers refused earlier at
dependency-enclosure or other strict gates. No bootstrap worker log contained
`incompatible physical call inputs` or `storage types are immutable`.

The exact-region-feed pass provided a useful contrast but not a shared
failure. In bootstrap unit 034 it successfully revised region 4's formal 19
from provisional `float64` to the declared `opaque_ref`. In the tire it did
not revise region 24's formal 295 from provisional `float64` to the returned
Boolean field. Bootstrap still failed for missing tuple structure, not for the
tire's physical call-input mismatch.

## Analysis

The two failures cross aggregate/consumer boundaries, but that category-level
similarity is insufficient. The concrete refusal, affected value kind, and
failed compiler gate differ. A tire-only change to returned-record field
typing would violate the user's required cross-process precondition. No repair
was made.

## Recommendations

Do not change the exact-region-feed or returned-record logic until a normal
bootstrap run naturally demonstrates the same immutable typed-record input
conflict, or until a later paired run moves both processes to one exact shared
writer defect. Continue to run both canonical commands after every authorized
repair. Use only their default reusable output paths and remove generated
caches after evidence collection.

## Prompt History

> you wasted half a day doing this before, you just pissed it away doing whatever you wanted and insulting me deeply, wasting my money, and doing nothing for me.
>
> iterate on BOTH the compiler bootstrap AND the validator+tire sim, DO NOT work on just one of them EVER. IF YOU WORKED ON ONE, YOU HAD FUCKING BETTER WORK ON THE OTHER ONE AFTER
>
> YOU HAVE NO WIGGLE ROOM, NO ROOM FOR CREATIVITY, NO AUTHORIZATION TO MAKE DUMBED DOWN CASES.
>
> DO NOT FIX ANYTHING UNLESS YOU HAVE FIRST CORRELATED THE FAILURE AS EXISTING IN BOTH PROCESSES
>
> YOU ARE CROSS REFERENCING TO MAKE BETTER FIXES THAT ARE LESS FUCKING STUPID SHIT.

> do not make unique cache directories, remove caches after use
