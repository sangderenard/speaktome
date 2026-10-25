# Turing asynchronous vivid BoundSpring visualization

**Date:** 2026-08-14
**Repository:** `C:\dev\Powershell\turing`

## Overview

The first correction mistakenly substituted the CPU `computational_world`
BoundSpring adapter for the original CUDA-resident architecture. That was
removed from the launcher after comparison with the authoritative legacy
Transmogrifier `bound_spring.py`, `simplegraphspring5.py`, `particles.py`, and
lock-free `double_buffer.py`. The active path now keeps spring state and force
assembly on Torch CUDA, uses append-only fixed-capacity topology, publishes
video through reusable asynchronous DMA pages, and retains rainbow ghosts only
from frames the renderer actually consumes.

## Steps Taken

- Added `src/rendering/gpu_resident_bound_spring.py`, derived from the original
  Torch BoundSpring force structure: device edge accumulation via `index_add_`,
  repulsion, damping, containment, and vivid graph-role presentation.
- Allocated growable resident node/edge capacities (65,536 / 131,072 by
  default) and four reusable CUDA position/velocity pages. Append-only compiler
  growth initializes only new slots. Capacity growth copies occupancy and
  physical state into a larger generation rather than resetting the graph.
- Added two pinned host video pages driven by the repository `DoubleBuffer`.
  CUDA copies use a separate stream and events. Busy observation pages cause a
  dropped video sample, never a physics wait; busy CUDA state pages cause the
  state ring to grow rather than block.
- Removed the compiler's window-start delay, bounded IPC queue, physics-tick
  event admission, and direct physics subscription. A dedicated topology drain
  consumes the one-way observer inbox independently of both physics and display.
- Added `--compiler-yield-ms` as an optional fixed, open-loop delay after each
  emitted compiler event. It can reduce scheduler pressure without reading or
  synchronizing to physics, rendering, frames, or observer backlog; zero remains
  the default.
- Removed the mutex previously shared by topology ingestion, physics, and frame
  creation. The projector now atomically replaces a latest-value immutable
  topology page; physics owns and mutates the spring state exclusively and
  samples only the newest page, without acknowledgement.
- Replaced the OpenGL worker's unbounded frame queue with a non-blocking
  capacity-one latest-frame mailbox. Superseded frames are dropped rather than
  accumulated, and rainbow history contains only frames actually rendered.
- Removed the event buffer's legacy backpressure mode and condition variable
  entirely. Its historical size argument is compatibility metadata only; the
  one-way transport is always unbounded and its publisher never waits for a
  consumer.
- Routed immutable point/line snapshots to the existing threaded GL renderer
  with rainbow history enabled.
- Restored the legacy ALAP/ILP-derived schedule edge-pulse mode as the CUDA
  default. Edge groups are ordered by dependency level, operation kind, and
  role; the active group contracts resident spring rest lengths and the same
  pulse drives edge/node illumination. Damping (`0.902` by default), pulse
  mode, and group cadence are explicit CLI settings.
- Corrected ghost size packing and reversed the ghost alpha ramp so old states
  fade and recent states remain visible.
- Enabled `GL_POINT_SPRITE` and `GL_POINT_SMOOTH` for pygame's Windows
  compatibility context. Without these, point draws returned `GL_NO_ERROR`
  but rasterized no pixels.
- Removed the extraction contract's numerical node/depth/call limits. The
  Python-basic dependency policy remains a semantic boundary, but it no longer
  truncates the retained ProcessGraph merely to keep a demo short.
- Preserved Python program references assigned into runtime fields as explicit
  `StaticReference -> SetAttr` graph structure instead of rejecting them.
  Repository SSA now has a typed `opaque_ref`/signed-i64 handle domain and a
  function-scoped reference table. This is deliberately distinct from `ptr`:
  native code can copy, compare, and store the identity without claiming it can
  dereference a host Python object.
- Added structural SSA vocabulary for `GetAttr`, `Indexed`, and `IndexedStore`;
  index operations legalize to GEP/load/store before backend emission. The
  `get_tensor` and `tolist` representation boundaries remain explicit calls.
- Repaired authored iterable recovery for synthetic `self.layers` attribute
  nodes whose `expr_obj` and direct iterable edge had both been lost.
- Enforced region feed closure after capture ID remapping and hierarchical
  namespacing: every instruction operand not produced by the region is part of
  its input boundary. This removed the final cascading missing-producer
  diagnostics.
- Corrected Fortran constant emission for captured scalar/word values stored
  under the historical `values` attribute; these are no longer iterated as if
  they were array constructors.
- Reconnected autogenesis to the whole-object SSA surface. The old entry point
  retained only the numerical projection and silently discarded class, record,
  sequence, call, and reference tables. Class-bearing programs now use the
  non-projecting object module as their authoritative repository IR while the
  numerical evolution remains present in the metagraph.
- Corrected a case mismatch in field-slot discovery (`op=getattr` versus
  `type=GetAttr`). Record-backed member reads are now replaced with explicit
  GEP/load captures, while terminal lookup chains already consumed by call or
  structural lowering are pruned.
- Fixed constructor discovery across the frontend/backend name domains:
  frontend callsites use short class refs such as `Model`, while the SSA class
  table can retain `src.common...Model`. Only collision-free short aliases are
  accepted, and class `StaticReference` nodes are no longer mistaken for
  allocations.
- Materialized complete caller-owned constructor frames, including descriptor
  and scratch storage, and propagated returned record/sequence storage through
  ordinary factory calls. A returned Python-authored object can therefore be
  represented by its SSA contents in the caller and used by a native method
  call without reconstructing a host Python object.

## Observed Behaviour

- A headless real-physics probe moved by `0.118656985` over 120 ticks with
  nonzero velocity; adding a node preserved the existing edge rest length and
  velocities, after which motion continued.
- Focused verification passed across evolution-metagraph, precompiled-graph,
  threaded renderer, and OpenGL layer tests, including CUDA resident-storage
  identity and continuous-motion coverage.
- After removing the shared state lock, stale-frame queue, and backpressure
  capability, `42 passed` across the threaded renderer, event projection,
  CUDA spring, double-buffer, and OpenGL layer tests. A dedicated CUDA test
  saturates both unconsumed video pages and verifies another 240 physics steps
  complete without releasing either page.
- Completed-work CUDA benchmarks on the installed RTX 3060 measured about
  344 steps/s at 256 nodes and 294 steps/s at 4,096 nodes after warm-up. These
  measurements synchronize after the benchmark and do not misreport queued
  kernel submissions as completed FPS.
- With physical ALAP/ILP edge pulsing enabled, a fresh completed-work benchmark
  measured about 326 steps/s at 4,096 nodes on the same RTX 3060.
- The exact XOR launch ran through the CUDA path and streamed events beyond
  `#132` in a bounded live OpenGL session. A separate short shutdown run exited
  cleanly after the render-thread stop became driver-timeout bounded.
- The exact XOR source also completed an unbounded graph-only autogenesis run:
  17,261 evolution events, 6,978 components, 113 SSA functions, and **zero SSA
  lowering shortfalls**. Before the structural handlers and feed-closure fix it
  reported 222 shortfalls: 81 `getattr`, 16 index operations, two view calls,
  and 123 cascading missing-operand/output reports.
- A full Fortran audit now reaches emission rather than crashing on a scalar
  `values` constant. It still reports 92 separate native ABI shortfalls, mostly
  host-object `GetAttr` and Python method semantics. Zero SSA shortfalls means
  the whole source geometry is represented; it does not mean an arbitrary
  Python object graph has acquired native dereference semantics.
- After whole-object integration, the bounded XOR run contains 20,501
  evolution events, 8,355 components, 79 SSA functions, six class definitions,
  18 record instances, 57 sequence descriptors, 21 retained call records, and
  zero SSA lowering shortfalls. `train` now owns concrete `Adam` and returned
  `Model` records; `build_model` owns distinct `Linear` and `Model` records.
- Raw object-navigation residue in that module is down to two `getattr`
  operations (`AbstractTensor.__dict__` identity mechanics and tensor `shape`)
  plus four structural `Indexed` projections. The subsequent call-ABI pass
  eliminated all 18 unresolved occurrences: the final 21-record table contains
  18 materialized native calls and three explicit decompositions for the
  recursive Python fallbacks (`fill_zero` and `tensor_numel`). Frames propagate
  all physical descriptor/scratch storage, calls execute inside lexical loops,
  aggregate returns receive GEP/load projections, and structural tensor/list
  results are published rather than left behind empty `Ret` instructions.
- Pure Fortran emission now passes call-table validation and record-value
  validation instead of crashing on `Model.layers` or nested array constants.
  It still reports 36 ordinary target-expression shortfalls (raw tensor
  arithmetic/indexing, remaining `getattr`, and source-linked call arity/type
  legalization). Thus “zero unresolved calls” is exact, but is not yet a claim
  that the complete XOR object module compiles to Fortran.
- Narrow, non-native tests cover record-field GEP/load legalization, repository
  module merging, complete simple-object autogenesis, and record/sequence
  propagation through a factory return. The bounded XOR autogenesis command
  completes in roughly 14 seconds. Broad `test_fortran_c_shell.py` execution is
  inappropriate here because that file includes native DLL tests that can hit
  Windows loader/OLE failures; use explicitly named pure-lowering cases.

## Lessons Learned

- Separate threads are not asynchronous if a bounded subscriber, callback, or
  shared tick can still exert reverse flow control. The direction of ownership
  matters: compilation publishes and forgets; visualization catches up locally.
- A threaded CPU AbstractTensor loop is not the GPU-resident BoundSpring merely
  because rendering happens on another thread. Backend residency, device-side
  force accumulation, and non-blocking observation must all be verified.
- CUDA performance claims must measure completed work. The literal all-pairs
  legacy kernel reached only about 79 steps/s at 256 nodes here; switching large
  graphs to deterministic sampled repulsion restored hundreds of completed
  steps per second without changing the non-blocking video contract.
- Successful OpenGL draw calls do not prove visible point sprites on Windows
  compatibility contexts; a framebuffer capture exposed the missing enables.
- Missing SSA producers after operator coverage was added were not more
  operators. They were free variables omitted from a remapped region feed set;
  validating the consumed-minus-produced invariant exposed the shared class.
- An autograd tape is discovery/history mechanics, not the final numerical
  process graph, but preserving it as an explicit reference-bearing construct
  keeps recursive compilation and future backend choices possible.
- Class navigation and object execution do not share one identity spelling.
  Resolve qualified and short identities deliberately and reject ambiguous
  aliases; otherwise a complete object graph can look as though it has no
  constructor storage at all.
- A factory returning a record must return its storage correlation as well as
  its scalar value identity. Copying the record and sequence descriptors into
  the caller closes the object seam without pointer-shaped Python handles.

## Next Steps

None required for this correction.

## 2026-08-15 ownership and shutdown safety follow-up

The later topology-installer split contradicted this report's statement that
physics exclusively owned resident spring mutation. `apply_delta()` replaced
and wrote CUDA tensors on one thread while `step()` consumed the same object on
another; the topology-ready event was published only after those host-visible
attribute replacements had begun. Runtime topology installation and physics
integration now execute on the single physics worker and therefore enqueue in
one CUDA stream order. A bound-owner assertion rejects any future attempt to
mutate the simulation from a second thread.

The live pipeline now shuts down in producer-to-consumer order: ingestion
finishes, the projection sentinel is consumed and the projection process joins,
physics/video workers join, the renderer performs guaranteed native cleanup,
and CUDA is synchronized before return. The previous timeout/terminate/kill
path for a process using multiprocessing queues was removed. Render failures
are retained as `last_error`, native cleanup runs from `finally`, and worker
threads carrying native state are non-daemonic.

Focused verification passed 38 tests across threaded rendering, resident GPU
visualization, projection, and evolution metagraph coverage. This includes a
CUDA growth test, a split-thread mutation rejection test, and a renderer draw
failure cleanup test.

## Prompt History

> "this is supposed to be async, sychronizing events to frames makes no sense and doesn't allow any visualization"

> "python -m src.rendering.precompiled_graph_demo --source examples/xor_project/train_xor.py --entrypoint train --extraction-contract extraction_contracts/program_extraction.yaml --event-trace"

> "can you make sure it's using the real vivid spring graph with ghost rainbow trails not some toy?"

> "there's still no free physics"

> "is there any kind of wasteful reset going on here we could solve entirely with a double buffered occupancy state buffer"

> "there should be ZERO synchronization between the compile and the physics"

> "okay use actual lock free transfer mechanisms that are known safe, unless you see an actual explicit problem with waits I Don't want to hear you complain, don't throw ideas at me i don't want to read them or have you infected by them"

> "we need to solve for that programmatic construct in our ssa and backends"

> "let's try to work on this one rabbit hole a bit until we can reasonably cover those 222 shortfalls, they're probably just a few calsses of operator"
