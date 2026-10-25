# DT mode conformance and dewar target simulations

**Date:** 2026-09-20

## Result

Added one focused test battery for the existing scientific and realtime dt
execution modes, and made every top-level simulation in the live dewar scene
cross the standard managed-time boundary. No third execution mode or scheduler
was introduced.

The distinction established in code and tests is:

- scientific mode rejects a causal-ceiling violation without advancing;
- realtime mode clips to the engine's causal ceiling and reports the remainder
  as `time_slip`;
- participant contracts (`BIND`, `DILATE`, `SUBCYCLE`, `HOLD`) are claims used
  inside controller reasoning, not execution modes;
- realtime processing cost and penalty are allocation inputs, while accumulated
  slip is still telemetry and does not yet drive catch-up dispatch.

## Target declarations

- The atmosphere `ChamberSim` now implements the existing
  `DtCompatibleEngine` boundary directly. Its callable ceiling is the most
  recently published chamber-law limit; its complete package snapshot includes
  both managed clocks.
- A general `MachineSystem` boundary advances one complete `MachineSim`, with
  the machine's existing whole-state snapshot. It publishes `HOLD` rather than
  inventing a mechanical tau.
- `CycleEngine` publishes the engine simulator's real 1 ms fixed interior step
  as `SUBCYCLE` and exposes the fixed accumulator's 50 ms retained-window limit
  as its realtime causal ceiling.
- Fluid and electrical systems publish explicit `HOLD` rows. Both are fully
  transactional, but neither currently has a defensible universal physical tau.
- Thermal retains its geometry/material stability ceiling and energy/power
  `BIND` publication.

The live dewar's atmosphere and mechanical callbacks still perform their
boundary exchange work, but their physics calls now go through
`step_with_state`; they no longer bypass managed clock, clipping, or slip
accounting.

## Realtime allocation repair

`MetaLoopRunner` already measured each scheduled call, but the allocation
ledger was updated by optional wrappers under human-readable labels while
`compile_allocations` looked up object identities. Those records could not
influence one another. The runner now writes measured milliseconds and the
normalized scientific penalty under the exact `id(AdvanceNode)` used during
allocation and writes measured `proc_ms` into the returned metrics.

## Verification

- Turing dt mode, time-contract, time-accounting, and allocation batch:
  25 passed in 3.31 s.
- Engine-toy machine, cycle, fluid, and thermal batch: 13 passed in 9.03 s.
- Spectral-analyzer electrical batch: 10 passed in 5.31 s.
- Real assembled dewar mode tests: 2 passed in 12.91 s. In the forced-slip
  case, atmosphere, machine, engine, fluid, and electrical reached 0.020 s;
  thermal reached 0.005 s and reported 0.015 s slip.
- A two-step headless live run completed. Every top-level dt participant
  reported 0.040 s in the ordinary configuration; cooling and electrical/fluid
  coupling remained active.

The full capability table and test entry points are recorded in
`turing/docs/DT_MODE_CONFORMANCE.md`.

## Prompt history

> "could we use probabalistic step firing in dt system weighted by tao slip so we stop working where it works fast to work where it needs catching up, are we not already sort of \"buffering\" between sims with tau?"

> "now, correct me if I'm wrong, but the time slip tau currently has no pressure one way or the other, there is never a \"reason\" experienced for altering time difference, or can stability or individual sims and limits actually create a slipping scenario, because our test right now is in lock step"

> "we need to set up a battery of tests for the different configurations of the dt system and we need to outfit all our target sims with the necessary levers to work with all the different modes by supplying the right values or functions to derive values for each mode's system of reasoning"

> "okay you're right they're different, try to use discretion"
