# Turing Woodshop ctypes Newton execution

**Date:** 2026-09-27
**Title:** The Python Woodshop now advances Newton state through its compiled DLL

## Overview

Connected the Python Woodshop demo to the complete native Newton dt window and
closed the compiler defect exposed by its first linked, singleton-output law.
The pygame client compiles the C shell plus three linked LLVM laws before it
opens its display, then each simulation step crosses the pointer-table ABI
through `ctypes`. Python still owns Machine/contact state and copies the
authoritative written ProgramABI buffers back into those Machines after the
native window.

## Steps Taken

- Added native-system attachment, batch invalidation, compilation, execution,
  telemetry readback, and Machine-state synchronization to `engine_toy/woodshop.py`.
- Made `engine_toy/woodshop_pygame.py` compile and attach the Newton DLL before
  opening pygame, and documented the behavior in `WOODSHOP_README.md`.
- Made `NativeSystem.prepare` use the established allocator for either C or
  LLVM artifacts and made `state_field_ids` prefer the compiler-designated
  written ProgramABI slot over callsite aliases.
- Added exact projection receipts to destructuring normalization. The receipt
  follows the source through compilation-unit extraction, links a flattened
  singleton native result to its authored `temporary[0]` target, and performs
  a final pre-gate SSA reconciliation if a later call-frame fixed-point round
  restores the obsolete PlanCall result identity.
- Preserved exact callee-formal identity for frame-aliased results so final
  signature pruning can rederive the emitted argument position.
- Added focused regressions for normalization receipts, authoritative state
  slot selection, mutable snapshot state carried through a native while/call,
  and singleton destructured results.

## Observed Behaviour

The complete Woodshop build finished normally inside the expected several-
minute window. Its final artifact was:

```text
build\woodshop_ctypes_verify_5\step_2__dt_system_over.dll
```

One native `1/120 s` window produced:

```text
before z = 0.019049999999999997 m, momentum_z = 0
after  z = 0.018368034567037634 m, momentum_z = -0.30414999110331353
advanced telemetry = 0.008333333333333333 s
singleton-result concordance receipts = 1
full-native completeness failures = 0
```

This proves that the ctypes call is not only returning telemetry: the linked
gravity result reaches the live `force_z` state, momentum consumes it, and
position consumes the updated momentum. Every declared state/control value in
the real compile was dynamic ProgramABI storage; the failure was not static
parameter registration. It was an SSA identity transition lost between
singleton destructuring and a later call-frame fixed-point round.

Focused validation passed 13 tests in 41.48 seconds. The final real compile and
execution also passed the full-native gate and state-motion check.

## Headless Profile

The pygame-free `WoodshopSimulation.advance(1 / 120)` path was profiled with
`cProfile`; compilation and import/warm-up time were outside the runtime
sample. The comparable measurements were:

```text
eager:  100 frames, 24.439674 s, 244.397 ms/frame, 4.09 fps
native: 120 frames, 26.091918 s, 217.433 ms/frame, 4.60 fps
native compile: 255.952 s
```

The native lane saved 26.964 ms/frame (11.0%) end to end. The Newton window
fell from 41.56 ms/frame in eager execution to 13.93 ms/frame at the complete
native boundary. Of the latter, `NativeSystem.prepare` cost 6.88 ms/frame and
`_managed_native_feeds_by_id` cost 5.23 ms/frame, so Python ABI preparation is
now a material part of the remaining Newton cost.

The dominant whole-simulation work is elsewhere: `woodshop.step` cost about
167 ms/frame in the native sample, `_resolve_pair` about 133 ms/frame, and
`part_bounds_xyz` about 120 ms/frame. Moving Newton native therefore cannot by
itself make this scene real-time.

Profiling also exposed a repeated-window compiler defect. Eager execution
publishes `dt_next = 1 / 120` and a nonzero `max_vel`; the compiled window
advances the physical columns and publishes the correct `advanced`, but its
remaining telemetry is zero, including `dt_next`. Feeding that zero into the
next round makes the second native frame advance only
`0.0008333333333315205` of `0.008333333333333333`. The native timing sample
therefore pinned `_newton_dt_next` to the already-authored requested step in
the benchmark harness before every frame. Production code was not changed to
hide the defect. Generated C inspection shows distinct `dt_next` and `dt_used`
loop-carried values and the correct outer argument order; the remaining fault
is inside the compiled controller metrics/sidechain publication path.

## Repeated-window compiler repair

The follow-up traced the zero `dt_next` to invalid SSA placement, not to a
static/dynamic parameter omission. An optional record-field projection was
materialized in an `if_merge` block even though a Phi consumed that value on
the predecessor edge. The C backend's old operand-shortfall message did not
identify this temporal relationship, so its diagnostic now reports the exact
consumer, producer candidates, formal accounting, dtype, shape, and
provenance.

The completed-module repair now treats each Phi operand as a use on its named
predecessor edge. It moves a pure projection closure only when the concordance
proves either one semantic use or one read-only ProgramABI field identity, all
dependencies dominate the selected placement, and same-block ordering is
valid. The move is recorded in `phi_edge_projection_placement_concordance`;
this is a dominance rule over finished SSA, not a Woodshop-specific fallback.
Indexed-to-GEP/Load lowering also retains the source projection provenance.

The final identity audit then exposed four additional, related findings. The
`PieceState.pub_limits` and `pub_limits_present` formals in both
`run_superstep` and its sequence-append helper had acquired their correct
public ProgramABI identities while still carrying older anonymous linked-frame
leases. Those were two incompatible owners for one physical value. Once a
declared field identity is proven, the compiler now retires the provisional
lease and matching shell-allocation declaration together and records the exact
transition on the shared `program_abi_frame_transition` page. The detector was
not relaxed or given an exception.

The final whole-program build produced:

```text
build\woodshop_record_fix_26\step_2__dt_system_over.dll
ProgramABI frame transitions concorded: 4
detector still reports 0
frame 1: ok=True, dt_next=0.008333333333333333
frame 2: ok=True, dt_next=0.008333333333333333
```

Both telemetry rows also report `advanced = dt_next = 1 / 120`; their measured
`max_vel` values were `0.5335964756198482` and `0.46956489854546646`.
Consequently the benchmark-only pin described above is no longer necessary:
the native two-frame path now advances the requested time without Python
intervention. The final compile, native C build, and two headless frames took
about four minutes in this verification run. This check did not by itself
establish complete eager/native physical-state parity; the longer comparison
below subsequently showed that distinction matters.

## Long paired profile and deviation audit

`time_trials/profile_woodshop_native.py` ran two independently constructed,
initially bit-identical Woodshop worlds through 10 warm-up frames, 300 ordinary
timed frames, and 30 separately profiler-instrumented frames per lane at
`dt = 1 / 120`. Compilation and warm-up were excluded from timing, and eager
and native execution order alternated each frame. Every frame compared all 17
objects' centers and linear momenta plus `dt_next` and all nine Newton telemetry
channels (38,080 scalar comparisons total).

```text
compile: 264.522441 s, final concordance findings: 0

eager, 300 frames:
  mean 163.273 ms, median 161.864 ms, p95 172.059 ms
  stddev 7.014 ms, min 157.517 ms, max 247.463 ms, 6.125 fps

native, 300 frames:
  mean 151.317 ms, median 149.535 ms, p95 159.372 ms
  stddev 8.232 ms, min 145.617 ms, max 255.334 ms, 6.609 fps

native end-to-end speedup: 7.323%
```

The separate 30-frame cProfile sample attributed 1.295 s to eager Newton
(43.2 ms/frame) and 0.426 s to native Newton (14.2 ms/frame). Whole-frame
cProfile time was 7.849 s eager versus 6.932 s native. In both lanes the main
remaining cost was collision work: `_resolve_pair` consumed about 4.3 s and
`part_bounds_xyz` about 3.9 s over those 30 frames.

The deviation audit did **not** establish physical parity. Fresh eager and
native simulations had identical initial item state (`max_abs = 0`), but the
first post-advance comparison already differed at
`jig.sawhorse-bracket.001.momentum_z`:

```text
eager   0.008052647514029721
native  0.03649799893239763
abs     0.028445351418367907 kg m/s
```

The largest observed difference occurred during warm-up frame 7
(`frame = -4`) at `tool.bar-clamp.002.momentum_z`: `0.9763996169862907 kg m/s`
(scaled difference `0.5516040976670129`). There were no non-finite mismatches.
After all 340 paired advances, the largest residual state difference was
`0.040289239414344286 kg m/s` at `tool.bar-clamp.001.momentum_z`; position
differences were much smaller (the largest reported clamp z difference was
`0.00021730981304098063 m`). `advanced`, `dt_next`, and every final telemetry
channel except `max_vel` agreed; final `max_vel` differed by
`0.02632824859957067 m/s`, consistent with the already-diverged momentum.

Therefore the timing result is a valid performance measurement of the current
native path, but the native path cannot yet be called numerically equivalent to
eager. The next diagnostic should compare eager and native outputs immediately
after `_advance_newton_dt_system`, before floor and pair contacts amplify the
first difference.

## Value trace and deterministic correction

`time_trials/compare_woodshop_newton.py` now compares every Newton column lane,
`dt_next`, and every telemetry field immediately after the dt-system boundary.
The C backend's opt-in full trace records every shaped result lane and, for
side-effecting calls, the argument values after the call. This located the
first corruption without collision or Python-world noise.

The linked gravity LLVM function produced all 17 correct values. The following
`state.force_z[...] = gravity_result` repository call had originally frozen its
`value_count` operand at 1, so `index_assign_double` broadcast lane zero. The
underlying defect was temporal: indexed-store extent constants were created
before final whole-module call/shape settlement and were not restamped
afterwards. Broadcast kernels already had such a restamp, but indexed stores
did not.

The repair extends the existing static-extent restamp to
`index_assign_double` and `index_set_double`, and runs it at the final module
seam after physical call adaptation has published exact region-feed receipts.
Those receipts are essential. A scalar actual used by `state.dt[...] = dt`
has source count 1 even though its callee-local view is shaped `(17,)`; the
linked gravity result has source count 17. Choosing either local shape as an
incumbent breaks the other case. Following the exact call edge preserves both
contracts:

```text
step_2__advance_pieces__planned_region_120  dt source count       1
step_2__advance_pieces__planned_region_121  gravity source count 17
final concordance findings                                    0
```

A fresh three-step run at `dt = 1 / 120` then compared 744 scalar values with
zero differences, including zero IEEE-754 ULP distance:

```text
step 0: 248 comparisons, 0 nonzero, max_abs 0, max_ulp 0
step 1: 248 comparisons, 0 nonzero, max_abs 0, max_ulp 0
step 2: 248 comparisons, 0 nonzero, max_abs 0, max_ulp 0
```

The machine-readable result is
`turing/build/woodshop_value_compare_verified.json`; its corresponding complete
native trace is
`turing/build/woodshop_value_compare_verified/native-values.log`.

## Corrected sustained profile

After the indexed-store correction, the paired headless profiler ran 20 warm-up
frames, 600 ordinarily timed frames, and 60 separately cProfile-instrumented
frames per lane. Compilation took 268.660 seconds and was excluded from all
step averages.

```text
eager, 600 frames:
  mean 157.164 ms, median 156.468 ms, p95 162.308 ms
  stddev 4.661 ms, min 151.747 ms, max 211.943 ms, 6.363 fps

native, 600 frames:
  mean 144.923 ms, median 144.227 ms, p95 150.643 ms
  stddev 2.948 ms, min 139.984 ms, max 160.347 ms, 6.900 fps

native whole-frame speedup: 7.789%
```

The complete 680-frame comparison made 76,160 scalar comparisons. All were
exact (`max_absolute = max_scaled = 0`) and there were no non-finite
mismatches. Over the 60 profiled frames, eager Newton consumed 2.595 seconds
(43.25 ms/frame) while native Newton consumed 0.844 seconds (14.07 ms/frame).
Pair-contact and bounds work remained dominant in both lanes.

The result is `turing/build/woodshop_sustained_profile.json`. The profiler now
saves a source-fingerprinted `native-system.pkl` in its build directory and
reuses it on subsequent compatible runs. `--rebuild-native` explicitly
invalidates that convenience; changes anywhere under `turing/src`, to
`examples/llvm_dt_system.py`, or to Woodshop source invalidate it
automatically.

## Lessons Learned

The compiler cannot reconstruct normalized destructuring after graph reduction:
that graph may have legitimately discarded the synthetic `temporary[0]` node.
The normalizer must publish the exact temporary/index/target relation, and the
linked-call transaction must retain it through compilation-unit boundaries.

Likewise, a semantic value-ledger lookup is insufficient when it can return a
physical alias with a different SSA id. A direct native result defines the
exact caller binding. The pre-native gate is the correct final enforcement
seam because all call-frame and signature fixed points have completed there.

The long compile was real compiler work over the dt-system source catalogue,
not accidental ingestion of `re`, `networkx`, or another unrelated package.
Repeated runs completed in roughly five to six minutes, below the stated
eight-minute healthy expectation and far below the twenty-minute fault line.

## Next Steps

- Repeat the long whole-world profile with the corrected indexed-store
  contracts if updated performance and post-contact drift numbers are needed;
  the direct Newton boundary itself is now bit-identical for the verified run.
- Reduce or cache `_managed_native_feeds_by_id` work; it currently consumes
  nearly half of the native Newton boundary.
- Profile collision broad-phase/bounds caching separately. It dominates the
  full headless simulation after Newton is native.

## Prompt History

> woodshop has a python demo. can you make it use ctypes to call out to the binary for the newton step?

> If anything is broken in a static way my first thought is to wonder if all the necessary parameters are being registered as dynamic/symbolic

> you may be running up against a need to define state machine capability on the ssa level like the tables for general programmatic needs

> when it goes forever either there's a loop bug in the compiler or someone put something in to compile like re or networkx or something

> oh dear no you can expect 8 minutes without a bug

> 20 minutes is a problem

> but continue your work dutifully

> woodshop can run headless right cane profile it
