# Turing live translation growth and emergency clamp

**Date:** 2026-08-14
**Repository:** `C:\dev\Powershell\turing`

## Overview

Replaced the reduced hidden-context graph-growth renderer left by
`HANDOFF_LLVM_LANE_AND_OOP_2026-08-14.md` with a compatibility entry into the
existing visible `MultiNetworkFluxSpring` + `LiveVizGLPoints` surface. Added
source class/scope attribution, live top-K expansion reports, renderer-only
signed urgency rings, and compiler-thread emergency ceilings for branch size,
depth, and height.

A later clarification made the central contract stricter: the visualizer now
consumes the compiler's append-only evolution events individually. One
`component-spawn` reveals one node; one `component-link` or
`component-handoff` reveals its edge. A bounded queue backpressures the
compiler rather than dropping or coalescing unseen mutations, while FluxSpring
continues stepping every display frame.

The hierarchy diagnostics do not alter spring anchors, forces, activation,
clustering, or scheduled physics. They remain an observational channel beside
the emergent system.

## Steps Taken

- Located the working surface in `src/rendering/precompiled_graph.py` and the
  living physics in `autoautograd/fluxspring/evolution.py`.
- Annotated AST nodes with lexical class/scope provenance before normal graph
  ingestion and carried that metadata through `EvolutionMetaGraph` events.
- Added incremental ownership propagation through component links and IR
  handoffs, SCC-safe depth/height measurement, per-stage censuses, relative
  branch size, and smoothed nodes-per-second reports.
- Added a separate cool-colored OpenGL ring pass driven by observational haze
  values. Immediate before/after tests prove applying haze leaves spring
  position and velocity unchanged.
- Added `ExpansionEmergencyClamp`, subscribed in the compiler's event thread.
  It raises `ExpansionLimitExceeded` with culprit and stage census. CLI limits
  default to depth 512, height 512, and 50,000 nodes per attributed branch;
  relaxation requires `--growth-limit-boost` or explicit limit flags.
- Extended `precompiled_graph_demo` so `--source ... --entrypoint ...` runs the
  full evolution recorder rather than topology-only ingestion.
- Replaced `compiler/graph_growth_display.py` with a wrapper around the one
  canonical visible surface.
- Added `LiveEvolutionEventBuffer`, exact event tracing, a configurable reveal
  cadence, and an on-window sequence/kind/queue display.
- Added a boundary restart cascade. A clamp writes a `.growth_flags` receipt,
  leaves the failed physical graph visible, waits while physics runs, and
  restarts into the same event ledger only after a `*.node.json` rule changes.

## Observed Behaviour

- Focused verification: `18 passed in 7.00s` across evolution metagraph,
  precompiled graph visualization, and FluxSpring evolution tests.
- A bounded real pygame/OpenGL run opened and closed cleanly while compilation
  continued and emitted live growth reports.
- An intentionally tiny CLI branch ceiling aborted compilation and reported:
  `scope spectral_route: nodes=3/1 ... stages=[process-graph:3]`.
- A 12-second XOR `abstract_nn` launch did not reach useful class output before
  the outer command timeout. This is not a substitute-program failure; the
  launcher remains aimed at the real source and should be observed in a longer
  interactive run.
- Final focused verification: `31 passed in 6.63s`. A real OpenGL
  launch printed and displayed ordered `component-spawn` and `component-link`
  events beginning with sequence `#0` while live growth reports updated; the
  compiler completed normally. This also caught and fixed a replay/live startup
  race which could previously enqueue `#1` just before replayed `#0`.
- Added and exercised `examples/live_compile/spectral_route.py` as a small
  file-driven entrypoint. Its three-second launch visibly traced graph open,
  individual AST node spawns, and individual links while physics advanced.
- Isolated full compilation in a spawned process after the XOR compiler was
  observed starving the pygame thread under the GIL. Compiler-assigned events
  cross a bounded multiprocessing queue and are replayed without renumbering;
  queue pressure still blocks compilation while renderer physics remains live.
- Exposed the real pre-ingestion compiler phases and AOT dependency search as
  nodes/edges. A 60-second XOR run produced 48 `scope train` nodes and 47 edges
  at depth/height 47, naming `Model`, `Adam`, `MSELoss`, their methods,
  unresolved builtins, and unbounded upward searches.
- Proved the emergency path with ceilings 15/12/12: compilation stopped at 16
  nodes, the retained 16-node/15-edge graph settled to zero expansion rate, and
  a `python/train` boundary receipt was written. Fixed a post-clamp crash by
  keeping spectral-inertia FFTs on the spring's NumPy histories instead of an
  unrelated globally selected Nodus tensor backend.
- Corrected the extraction boundary after the user clarified that domain
  classes are meant to be pursued and CPython implementation code is not.
  Added an exhaustive YAML extraction contract with dispositions for authored,
  repository, third-party, stdlib, builtin, native-extension, DLL, and unknown
  callables. Source ingestion, intrinsic lowering, interpreter host calls,
  native reuse, explicit bounded decompilation, and rejection are distinct.
- Wired the contract into ProcessGraph parent expansion, call receipts, AOT
  checkpoint identity, the live CLI, Python source file/byte/work/depth limits,
  and transitive PE function/byte/depth limits. A governed XOR trace continued
  through `Model`, `Adam`, `MSELoss`, and `Linear` while no longer pursuing
  `range`, `float`, or `print`; progress reported `contract depth 1/512`.

## Lessons Learned

- Compiler hierarchy is valuable diagnostic metadata but must not be turned
  into a force or anchor when the physics' emergent clustering is itself the
  signal under study.
- Source ownership must be captured before graph lowering; reconstructing a
  class from later numeric topology is unreliable.
- Advisory visualization and load-bearing safety are different layers. A halo
  can warn, but only a subscriber executing in the compiler event thread can
  stop runaway expansion in time.

## Next Steps

- Run the real XOR training source interactively long enough to capture its
  first class-attributed top-K expansion census and tune the default ceilings
  from measured behavior.
- Consider persisting periodic growth reports as a small machine-readable
  artifact after the live format stabilizes.

## Prompt History

> "this agent struggled and made compromises and false starts, maybe didn't find the right spring graph version , and really let me down. I need the visualization working AND for it to track height and depth so that it can show me the branches that have become exploded subgraphs above or below the program, so we'd see it building live, and see live a report of subgraphs getting out of hand, and then we can go back from that and say, my translation leaks like a seive right here on this class I can probably fake"

> "heirarchy ordering must not damage the physics of schedued activation clustering, the emergent value of the system, so we should actually just put haze effect or something colored for + or - relative to subgraph size... that's overly complex, something simple maybe, like cool colors increasing in urgency in a circle with a little margin letting the node be the node... but you see, we don't want to damage the physics just to visually see heirarchy, visually we will see massive clustering and multiple huge cluster entities and then the list of subgraph sizes the topk or size and rate of expansion or something I don't know, would identify the leak then you'd go back and spoof that node"

> "this system should have an additional clamp for emergencies that you'd have to cli boost, something that will kill depth or height that's gone astronomical"

> "that's not precisely what I said, I said live changes viewed, like node by node edge by edge tie-in with the compile and the visualizer"

> "can you run it on the training file and see it safeguard and process to an analyzeable system of nodes"

> "there needs to be detailed contracts for extraction specifying the choice in all cases, where decompile lines are drawn, where python calls are accepted, where dlls are just used in place, etc, like a big yaml sheet of the desired parameters for program extraction"
