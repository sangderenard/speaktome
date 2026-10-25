# Original Python validator native integration

## Work

The user accepted the measured tire/controller trajectory precision for now
and requested integration into the original validator. Located
`tools.run_vehicle_native_assembly._run_dually_python_profile` in Turing.
Its coupled `_DuallyDTState` and `advance` closure are different from the
verified standalone managed tire. Started the canonical full-native project
lowering, preserving the authored simulation and no-Python-callback policy.
It published 108 connected units and an exact reduced graph of 22,376 nodes.

Two read-only stack samples found the deployment metadata classifier spending
time recounting graph edges for each node's cache query. Added a classifier
factory whose lifetime is one stable scan, preserving existing per-query
behavior for isolated callers and cache invalidation between changed graphs.
This change does not modify numerical/controller behavior.

## Validation and state

The focused classifier group passes 3 tests. The adjacent group passes 38
with one failure reproduced using the HEAD module in an isolated process:
`test_tensor_method_candidate_requires_tensor_receiver_value` is missing
the expected `tensor` attribute. The fixed classifier processed the saved
real graph in 2.470 seconds and matched all unchanged uncached classifications.
No working-tree reset, dependency install, commit, or push was made.

The initial full compile finished naturally after 2720.846 seconds, exit 1,
at four Pygame drawing effects. It loaded the pre-classifier-change compiler.
The user then explicitly selected keeping the Python viewer and compiling
the simulation. The coupled simulation entry and opt-in runner integration
are being implemented and validated under that revised scope. See
`turing/docs/CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md` for terminal updates,
artifacts, commands, and remaining integration work. Do not treat source
discovery or the provisional numerical acceptance as a working native validator.

The extracted simulation's eager state/feedback comparisons now pass for both
inactive and active tires (1e-12 tolerances). Snapshot restoration and the
existing viewer frame acknowledgement test also pass: four tests in 280.82 s.
The eight-lane simulation-only build started at 19:26 local time; native
execution and the final viewer launch remain unverified. The continuation
document names the exact build log and the optional consecutive-window test.

That build stopped naturally after 798.786 s on cyclic atomic regions.
The user required fixing the compiler rather than rearranging an intermediate
graph. An isolated HEAD-fusion diagnostic reproduced the identical error in
`balloon_tire_vector_step`: a direct tensor-to-reshape edge masked the parallel
path through `tensor.shape`, an index, and the shape tuple. Fusion now retains
the coordinator-path provenance even when the endpoint pair also has a direct
edge. The unchanged saved function graph passes atomic ordering after this
repair, and two focused regressions preserve both the minimal parallel path
and the real reshape pattern. The adjacent suite passes 84 and fails 14;
all 14 reproduce with HEAD's fusion module in an isolated process.
The fresh eight-lane rebuild is `build/validator_simulation_native_v2_20260914`
with a same-named `.log`; native execution still awaits that build.

The fresh rebuild passed the first failure, then stopped after 1146.769 s on
a different control-region cycle in `vehicle_tire_recurrence`. An exact replay
captured the specialized graph and proved that synthesized tuple-call result
projections were misclassified as numerical tensor indexing. The classifier
now respects their explicit `authored_call_result_projection` marker and its
serialized cache includes a semantic schema version. The unchanged saved
specialized graph passes control replanning; ordinary tensor indexing remains
numeric. The combined suite passes 86 with the same 14 baseline failures.
A fresh source build with both repairs is in progress under
`build/validator_simulation_native_v3_20260914`, with a same-named log.
The v3 source build passed both original failure stages and lowered its
regions, including the real controller update / snapshot / restore functions.
It saved `pre-frame-link.pkl` and is assembling complete repository SSA.
There is still no verified coupled native library or viewer launch command.

## Prompt History

> that's acceptable precision for the moment, can we use this now in the code that inspired the need to make the dt controller compile, i think it might have been for the validator python version

> oh you'll need to give it up to 20 minutes

> this is normal, relax it will come through soon or within an hour, don't get too excited

> Keep Python viewer; compile simulation

> what's a command to see the new mostly native version , or is it not ready yet

> you'll need to fix whatever caused the compiler to make such a mistake even if you can fix the work manually in an intermediary state

Preserved historical user scope:

> your task, is to compile a totally pythonic program end to end and compile the entire thing together, so that a generic shell is adequate representation

> that means, crucially, that the native insertions in the pythonic validator are not seen by the compiler ingesting the pythonic representation
