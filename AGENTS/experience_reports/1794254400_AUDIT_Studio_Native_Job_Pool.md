# Native Studio job-pool qualification, 2026-11-09

Inspected local Nodus 6f3183e49d74d2392d889bfde740bafb72a8f680. This is a source-backed integration plan and bounded scheduler fix, not a completed service cutover.

## Existing authority to use

- `include/graph_runtime.h`: GraphRuntime owns a 1:1 ThreadManager and native plugin-row state. Do not replace this with a Python scheduler.
- `include/nodus_headless_abi.h` and `src/headless/nodus_headless_abi.cpp`: create graph, register plugin rows, configure typed FIFO edges, exchange named-port tokens, and step the actual ThreadManager without a renderer. The header's old process-global dispatch caveat is partly superseded by GraphRuntime ownership; topology still includes Canvas context.
- `include/edge_tensor_fifo.h`: fixed-size, single-writer, multiple independently subscribed readers. Multiple readers broadcast samples; they do NOT implement exclusive competing-consumer work claims. `top_k=0` refuses a full write without advancing readers. Subscribe before publishing; subscribe-at-head skips prior samples.
- `include/edge_tensor_fifo_transaction.h`: existing binary transaction snapshot preserves backing bytes, slot tags, writer binding and reader cursors. Graph-definition/Canvas topology save alone omits live queue state. Snapshot is not automatically a durable disk journal or a transaction across plugin lease state.
- `camera_software/nodus_runtime_bridge.py` in spectral-analyzer wraps the native edge ABI, but using that alone would not satisfy native ThreadManager ownership.

## Bounded change

`src/thread_manager.cpp` now requires acyclic topology for predecessor-count parallel dispatch in both modes. Previously FreeSpinning allowed upstream A -> B <-> C into the parallel loop: A completed, the ready queue emptied, and remaining B/C could never complete. The fix reuses the existing cyclic sequential ordering; it introduces no new scheduler.

Regressions added to `tests/thread_manager_test.cpp` for both modes and `tests/nodus_headless_runtime_test.cpp` for the real headless graph route. CTest timeouts bound hang failures. These tests prove completion only when executed; they do not establish token-ready KPN semantics or job durability.

## Next native slice

Use custom native admission/lease/return/reclaim rows in the existing graph. One authoritative native lease owner assigns work; browsers must not be independent competing FIFO readers. Carry job identity, owner/account reference, scene revision, work range, lease generation, worker/device identity and product identity in typed tokens. Python retains authenticated HTTP ingress and account library presentation, not pool scheduling. Reuse existing continuation/recovery protocol rather than invent new ray state.

A lease return must match its generation and scene revision, accept at most once, and leave timed-out work eligible for reassignment without losing its original retained rays. Checkpoint native FIFO state together with lease-owner state at a real quiescent transaction boundary before claiming crash recovery. Keep queue admission and GPU residency admission distinct; a yielded live GL process still holds VRAM.

## Remaining correctness gates

Current `thread_manager.cpp` ignores typed input/output transfer return values around lines 894 and 1429. Scalar inputs may fabricate zero without an available token. A one-line early return is insufficient: earlier input rows may already have consumed samples, and earlier outputs may already have published. Define and test the module transaction using the existing transaction machinery rather than hiding partial effects. Full-output, empty-input, stale-return, duplicate-return, lease expiry, restart and two-account isolation need real native graph tests.

## Verification

`git diff --check` passed. Current-source build command:
`cmake --build build --config Release --target nodus_headless_runtime_test test_thread_manager --parallel 2`
reached MSBuild and Checking File Globs, then execution was blocked by network policy for `https://dc.services.visualstudio.com:443`. No MSBuild/cmake process remained afterward; existing test executables retained October 23 timestamps. No runtime test pass is claimed and no source changes were deployed. Do not use the old Release binaries as evidence for this patch.

## Prompt history

User: "you could future proof us to the moon if you used nodus fifo for the job storage, be set up and imply that load management is probably going to be nodus handled eventually"
User correction relayed by parent: "it's not optimized for python ... what would matter more would be looking at whether you can run an instance of thread manager from nodus with the fifos from nodus with some custom nodes specifically for our job stealing pool"
