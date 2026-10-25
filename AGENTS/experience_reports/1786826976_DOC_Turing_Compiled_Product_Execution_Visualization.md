# Turing compiled-product execution visualization

**Date:** 2026-08-15
**Repository:** `C:\dev\Powershell\turing`

## Overview

Expanded the live compiler visualization from a replay of unrelated local
ProcessGraph levels into a compiler-authored view of the accepted compiled
product. The authoritative replay is now the prepared `ControlProgram`
composed with selected/remapped dispatch regions, deployment evidence, and
later SSA/backend lineage. ProcessGraph levels remain available only as a
local-DAG fallback before a complete execution plan exists.

## Steps Taken

- Added `compiled-execution-plan-v1` evolution events containing ordered
  symbolic execution frames and exact cross-IR component memberships.
- Recorded sequence, conditional alternatives, calls/returns, loop
  headers/latches/backedges/exits, while-loop condition mechanics, state
  dispatch, parallel deployments, lane joins, dispatch entry/instruction/exit,
  and durable `ControlDeploymentRegion` evidence.
- Preserved uncertainty honestly: branches are alternatives and retained loops
  contribute a recurrent structural iteration; no runtime predicate outcome or
  trip count is fabricated.
- Used the selected/remapped `aot.region_programs` table rather than colliding
  shell-local region indices.
- Extended frame membership through explicit representation handoffs after SSA,
  packaging, and backend emission.
- Retained deployment-frame, membership, region, join, and scheduling metadata
  when control instructions enter the SSA evolution surface.
- Made CUDA/CPU BoundSpring prefer global compiled execution frames once
  available. Nodes outside the accepted product remain visible but do not
  masquerade as executable schedule members.
- Added semantic visual channels: dedicated hues and sizes for branches, loops,
  deployment lanes/joins, calls, and dispatch boundaries, plus role-specific
  line colors for backedges and control/deployment relationships.
- Added HUD execution coverage (`covered / resident nodes`) so a shallow plan
  cannot be confused with coverage of a much larger compiler ledger.
- Final backend targeting now emits every accepted numerical region and returns
  the complete mapping as `AutogenesisCompilation.final_artifacts`; the legacy
  `final_artifact` remains the first region for compatibility.

## Observed Behaviour

- 67 focused evolution, SSA, WebGL, visualization, and physics tests pass.
- The exact XOR source compile completes with 40,555 evolution events, 11,480
  visible nodes, 23,502 explicit edges, and 693 compiler-authored execution
  frames.
- Its accepted execution membership covers 1,431 nodes: 467 structured
  execution/control nodes, 482 accepted precompile nodes, and 482 corresponding
  SSA nodes.
- The 693 frames include 100 calls, 100 returns, 21 loop headers, 21 latches,
  21 exits, 102 dispatch entries, 226 dispatch instructions, and 102 dispatch
  exits. This replaces the former misleading 22-group global presentation.
- A pre-existing broader autogenesis failure remains in
  `glsl_deployment_strategy._fold_callsite_structural_values`: comparing a
  retained NumPy array with `!=` raises the ambiguous-truth-value exception.
  It predates this change and blocks the array-fed test independently of the
  execution-plan recorder.

## Lessons Learned

- ProcessGraph schedule depth is a local dependency coordinate, not a deployed
  program timeline.
- Deployment is not absent from the compiler; it was absent from the observer
  schema. Recording the accepted product fixes that boundary without creating
  a visualization-owned scheduler.
- Compiler discovery/provenance surfaces and executable product membership are
  both valuable, but they must remain visibly distinguishable.
- Static compiled control can show exact loop mechanics without claiming to
  know runtime decisions. Runtime iteration counts require execution telemetry,
  not inference from IR.

## Next Steps

- Optionally feed backend runtime predicate, iteration, and deployment-lane
  telemetry into the existing symbolic frames, preserving the static plan as
  the baseline and marking observed execution separately.

## Prompt History

> "there has to be something amiss, is the ilpscheduler unable to handle control flow? a 17k node compile became a swarm of not connected or activated nodes and then a center 22 stage plan. maybe that's possible? under some circumstances? but how does it correlate to those 17k nodes"

> "the scheduler also doesn't know how to handle deployments either, and I don't know if we're using deployments right in our process here or we're cutting it out"

> "okay well can we by any chance gracefully expand the visualization and physics graph to become more completely and honestly the compiled product of all the subgraphs, recognizing that a process graph can tell us an execution but not any execution, for which we need to draw in the full semantic richness of the actual prepared compiled fully accepted code that we can display, illuminate, and derive physics from actual loop iteration mechanics and deployments and conveying with color lines and dots more information density relevant to the system we have"
