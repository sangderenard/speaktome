# Turing scheduled edge contraction force repair

**Date:** 2026-08-15
**Repository:** `C:\dev\Powershell\turing`

## Overview

Rebalanced the CUDA-resident BoundSpring schedule pulse so an activated edge
has an observable geometric contraction. CLI contraction settings were
reaching the simulation and dynamic rest lengths were changing, but the active
geometry could continue expanding because edge stiffness and accumulated node
repulsion were badly mismatched. Adaptive timestep normalization at a target
velocity of `1` also suppressed stronger forces.

## Steps Taken

- Reproduced the failure with 200 nodes in the sampled-repulsion regime: with
  every edge active, 95% contraction, and yank `1`, mean edge length increased
  from `1.584` to `1.972`.
- Exposed existing force coefficients as `--spring-stiffness` and
  `--spring-repulsion-strength`, tuning their defaults to `64.0` and `0.005`.
- Kept the existing smooth rest-length contraction and smooth yank-force
  envelope; discarded proposed controller/impulse rewrites.
- Set the default adaptive target maximum velocity to `100` as requested.
- Removed presentation-only `compiler-phase` markers from physical projection;
  telemetry still retains them in the compiler census.
- Restricted camera fitting to nodes participating in visible edges and added
  monotonic physics-sequence/topology-revision guards to observation consume.
- Closed two deeper one-frame race windows: video publication now retains the
  exact immutable topology page used by its physics step instead of re-reading
  a possibly newer page, and resident topology writes publish a CUDA completion
  event that physics waits on device-side before using the revision.
- Removed an invalid projector-side interpretation of `component-spawn.sources`
  as physical dependencies. Only explicit `component-link` and
  `component-handoff` events create visual springs.
- Removed the detached visualization scheduler and its topology-derived
  ASAP/ALAP groups. `ProcessGraph.compute_levels()` now publishes the exact
  embedded-`ILPScheduler` result as `graph-schedule-finalized`; SSA retains the
  same authored level/group that determined instruction emission, and CUDA
  merely replays those ordinals on incoming explicit edges.
- Mirrored already-planned ProcessGraph levels without rescheduling, following
  DualIR's read-only composition pattern. Closed SSA/backend graphs retain a
  finalized presentation state, and WebGL shader nodes are indicated without
  receiving an invented schedule.
- Added a geometry-level regression test rather than checking only the hidden
  rest-length tensor.
- Tested the actual SymPy zeta graph against an identical pulse-off run.

## Observed Behaviour

- Exact zeta-graph counterfactual probes established that the parameter balance
  produces real geometry influence rather than illumination-only activation.
- The final focused spring suite passed `5` tests after setting target maximum
  velocity to `100`.
- The completed zeta projection now has 49 nodes, 111 edges, and zero isolated
  nodes. The full precompiled visualization file passed 19 tests and the camera
  matrix/filter file passed 7 tests.
- After the topology/observation ownership repair, 29 combined precompiled
  visualization, camera, and threaded-renderer tests passed.
- The final compiler-authored projection passed 39 combined evolution,
  ProcessGraph, GPU visualization, and WebGL finalization tests, plus five
  focused extraction-contract/ILPScheduler/WebGL tests. No size clamp was
  restored: the demo's
  default contract is `extraction_contracts/program_extraction.yaml`, whose
  Python limits are unbounded; alternate search entry points can still omit or
  override that contract.

## Lessons Learned

- A changing rest-length buffer proves only that an actuator target exists; it
  does not prove that integrated geometry responds in the intended direction.
- Porting a force topology without its force-scale ratios can leave the code
  structurally connected but physically ineffective.
- Adaptive timestep targets can counteract coefficient tuning; their configured
  scale must match the intended visible velocity regime.

## Next Steps

None required for this repair.

## Prompt History

> "no matter what I set these settings to, there's no evidence that any force is feeding in at all to the integration of the system on the activation of any edge, they simply do not contract"

> "it has to be smooth, physically real, second order or w/e behavior, you just need to tune it right"

> "put target max velocity at 100"

> "one is sometimes there's a aquamarine lone node not connected to anything in the sympy example, and the other is that sometimes the view flashes like we're getting a frame that used entirely different positions or hugely different zoom like it just flashes to the camera somewhere else or to a different graph just for one frame and then goes back"

> "I'm still getting flashes that seem like they're different moments or different projections"

> "did you put size clamps back on by default or something? the nodes in the compile don't match the nodes in the system, there are so many nodes without connections"

> "okay audit if you fundamentally broke something badly, because \"python implementation machinery discovered during traversal\" was all the semantic program i intended to schedule. in what way were tapes not intrinsic, deeply, to the program in the graph"

> "THE VISUALIZATION IS NOT TO HAVE ITS OWN SCHEDULER, THE SCHEDULING COMES FROM THE SCHEDULER PROCESS GRAPH USES. THE ONE THAT ACTUALLY DETERMINES THE PROGRAM."
