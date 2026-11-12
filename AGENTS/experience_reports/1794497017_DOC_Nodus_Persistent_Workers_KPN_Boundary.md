# Nodus persistent workers and the KPN boundary

**Date:** 2026-11-12 UTC
**Title:** Persistent Nodus workers and the remaining free-process contract

## Overview

The owner clarified that Nodus workers should be permanent entities, with
per-worker result FIFOs collated on the ThreadManager thread. The intended end
state when all tables run FREE is a genuine Kahn process network. This report
records the current code boundary and the first scheduling repair; it does not
claim that FREE is a KPN yet. The collision engine is outside this lane.

## Steps Taken

- Read the canvas shell tick controller, ThreadManager dispatch, table-edge
  FIFO implementation, existing KPN design draft, and scheduler tests.
- Confirmed that `ThreadManager::run_loop` owns a persistent manager thread,
  while its parallel dispatch created and joined pooled workers per tick.
- Added a persistent worker pool to `ThreadManager`, default enabled, with a
  runtime debug toggle that selects the previous per-tick construction path.
- Exposed the toggle through canvas and headless APIs and the SDL shell
  `--persistent-workers=0|1` option.
- Added a worker-identity regression and headless toggle checks.

## Observed Behaviour

- `ctest --test-dir build-service -C Release -R '^(thread_manager_test|thread_manager_affinity_test)$'`
  passed 2/2 after the persistent-pool change.
- The Nodus `build-service` static library and shared canvas DLL built with the
  new scheduler and exported canvas toggle.
- The main CMake cache restored its vcpkg dependencies, and direct compilation
  of `frontend_shell.cpp` then passed. The full SDL shell target did not link:
  four running Python processes hold the existing `nodus_tensor_core.dll`,
  causing LNK1104 when the build tries to replace it. No server process was
  stopped manually. The shell option has not been qualified in a linked shell
  executable.
- The current worker output rows still publish directly to edge FIFOs. The
  manager still submits and completes discrete ticks. Thus persistent workers
  alone do not implement independent free-running table processes or
  manager-collected result FIFOs.

## Lessons Learned

- The SDL shell has its own persistent canvas tick thread and a separate
  `--canvas-config=IDX:DT:RATIO:FREE` setting. That shell flag decouples canvas
  steps from display frames; it does not make each table a free process.
- The canvas's per-module FREE choice currently falls through to pooled
  per-tick execution. `ThreadManager::Mode::FreeSpinning` also enters the
  same `run_scheduled_tick` pipeline as Scheduled.
- Existing edge FIFOs are bounded; `top_k > 0` drops old unread data, while
  `top_k == 0` rejects writes when full. A true KPN path must explicitly use
  lossless channels and blocking reads, and must define how bounded
  backpressure affects cycles.
- Existing tests require a producer's output to reach its dependent module
  within the same scheduled tick. Any manager-side result collation must
  preserve that scheduled-mode behavior.

## Next Steps

See `AGENTS/todo/1794497017_nodus_free_process_kpn.stub.md`. In particular,
move FREE table execution off the canvas tick barrier, let persistent table
workers block on their input channels, give each worker a result FIFO, and let
the manager greedily commit results and wake dependents. Verify long-running
production without new `gp_canvas_step` calls, fan-out ordering, cycles with
seed tokens, pause/shutdown, and graph changes at quiescent boundaries. Do not
present the current worker-pool change as completion of the KPN objective.

## Prompt History

> "nodus can run things in different threads, I believe it uses mailboxes not sure, anyway, our audio can be made more realtime if it ends up contained in a table with free run permission, because if it isn't realtime it isn't appreciated"

> "nodus is getting a collision engine, just fyi, i've interrupted you a lot please try to achieve all goals, but the nodus collision engine isn't your lane"

> "Try to make a truly free thread, see what happens, I mean, we have to do mailboxing somewhere for the fifo for it and a skip on that is like zero time"

> "no, don't make new thread messaging. what is the thread manager tick actually doing to the workers, they aren't dispatching and joining one moment to the next, and if they are, where are the parameters to specify the tick ratio of whatever that work is or the no cost little sync check"

> "The workers were supposed to mitigate this by being permanent worker entities taking work and never initializing or joining because that would take forever"

> "Make keep workers alive a toggle so it can bea debug option"

> "The workers would all have result fifos i think to a results pool that greadily collates results and offers them on the thread manager thread i think is how i would want it"

> "The end result needs to be we have a genuine khan process network when all tables run free using worker threads that don't die they just pump results"
