# Turing native compiler frontier repairs

**Date:** 2026-09-14
**Title:** Scalar adjoints, broadcast storage, control ABI, and optional records

## Overview

Worked in the independent `C:/dev/Powershell/turing` repository. The detailed
technical record is `turing/docs/NATIVE_FRONTIER_REPAIRS_2026-09-14.md`.
No full suite, package installation, commit, push, or subagent delegation was used.

## Steps Taken

- Read the requested frontier/hazard documents and relevant workspace guidance.
- Reproduced scalar-loss adjoint failure and repaired single-input Phi recovery.
- Reproduced a broadcast allocation mismatch; fixed shape settlement before
  tensor storage registration and removed the recurrent loss workaround.
- Compared native recurrent history gradients and Adam updates with independent
  finite-difference/NumPy calculations across all parameters and moment buffers.
- Replayed the validator's pre-frame checkpoint and repaired control uniform
  storage dtype precedence, stale insertion positions, and scalar/sequence
  confusion. Extended call-edge shape propagation and specialization checks.
- Added optional record constructor payload/presence support, corrected the
  existing Metrics ABI declarations, and verified native absence versus zero.
- Added reusable inspection/replay helpers under `AGENTS/tools/`.

## Observed Behaviour

Seventeen cross-frontier tests passed in 117.35 seconds. Seven optional native
record/scalar tests passed in 19.07 seconds. Validator v7 produced repository SSA
with 13 formal-parity groups, three dominance findings, and one optional result
contract; it is not an executable validator artifact. Full source v8 incorporates
the new optional declarations. The separate real-engine run uses 64 transitions,
two training experiences, one validation experience, and two native Adam epochs.
Their logs/artifact paths and final status belong in the turing technical record.

## Lessons Learned

A successful first gradient element can conceal undersized native storage.
Poisoned buffers and independent numerical references exposed that defect.
Compiler-issued IDs starting at one billion invalidated an older magnitude
heuristic; the guard is diagnostic only and never allocates identities.
Optional fields require distinct presence facts, including inside sequence rows.

## Next Steps

Follow `todo/turing_native_frontiers_20260914.stub.md` and the turing technical
record. Continue from existing run results rather than duplicating quiet builds.

## Prompt History

> Work in C:\dev\Powershell\turing.
>
> Read these first:
>
> - AGENTS.md
> - TEST_BASELINE_AND_HAZARDS.md
> - docs/FRONTIER_REPORT_2026-09-14_NATIVE_COMPILER.md
> - docs/CONTINUATION_2026-09-14_NATIVE_COMPILER_FRONTIERS.md
> - docs/CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md
> - docs/CONTINUATION_2026-09-14_NATIVE_DT.md
>
> There are two active native-compilation frontiers, and both should be used as pressure tests for the compiler:
>
> 1. Validator DT native simulation: keep the Python viewer, but compile the coupled physics and adaptive DT controller underneath it into a native DLL that owns the whole viewer-tick cycle without repeated Python dispatch for adaptive inner substeps.
>
> 2. Perforated engine training: get the engine/perforated network training path working as native-compiled, history-respecting forward graph -> backward graph -> update execution.
>
> Do not treat either frontier as a one-off workaround target. Use their failures to improve the compiler until both compile and run correctly. If one frontier exposes a compiler defect, fix the compiler generally, add focused coverage when useful, then retry the frontier. If the other frontier exposes a different defect, do the same there. Move between the two as needed to keep finding real compiler limits.
>
> A recent important repair changed compiler-created representational IDs to come from one central issuer: `src/compiler/monotonic_ids.py`, `GLOBAL_MONOTONIC_IDS.mint()`. Do not reintroduce local counters, range reservations, or max-scan “next id” allocation for new compiler-created IDs. Source-side IDs may be references or provenance, but they are not authority to allocate new SSA identity.
>
> Be patient with compiler runs. These builds often take 10 minutes to more than an hour. Do not spend turns reporting that a quiet compile is still running. Wait for concrete results: a native artifact, a specific compiler exception, a numerical mismatch, a focused test result, or a necessary question.
>
> Use targeted tests and checkpoint replays. Do not run the full suite as a default gate; `TEST_BASELINE_AND_HAZARDS.md` explains why.
