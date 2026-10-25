# Turing balloon appendage native boundary

## Summary

The native Mechanical Creature deployment formerly packaged the new balloon
tire scalar authority but still called the legacy analytic torus/contact law
inside `vehicle_native_graph_tick`.  This session replaced that live boundary
with the persistent four-wheel balloon skin assembly, closed bead and fixture
reaction wrenches into the canonical vehicle graph, and compiled a new native
DLL/viewer bundle at `turing/build/vehicle_native_balloon_connected`.

The bead Kelvin--Voigt junction was the immediate free-energy source under the
explicit timestep.  It is now a symbolic backward-Euler junction kernel with
an equal/opposite rim wrench.  Membrane Kelvin forces retain their authored
law but receive an impulse passivity limiter.  The native host uses sixteen
microsteps at the existing 4096 Hz outer tick because the nonlinear elastic
skin remains unstable at a single 4096 Hz explicit step.

A generic `GraphAppendageReplacementContract` now records the exact graph cut,
candidate inputs/state/outputs, external JSON parameter transport, novelty
channels, and exact-teacher invariants.  Duty telemetry is read-only and the
scientific viewer renders a hub attachment ring plus `EXACT SKIN`/`GPU DUTY`
HUD state.  No network is silently claimed in the current exact native build.

## Verification

- `python -m pytest tests/test_graph_appendage_replacement.py tests/test_vehicle_native_deployment.py -q --tb=short`: 12 passed.
- `python tools/build_vehicle_native_teaser.py --output build/vehicle_native_balloon_connected --no-launch`: completed all four milestones and emitted both DLL and EXE.
- The launched viewer exposed one Mechanical Creature render window. Windows
  Graphics Capture failed with `SetIsBorderRequired ... 0x80004002`, so no
  screenshot was fabricated or captured through an alternate path.

## Remaining scientific risk

The exact skin is stable only with the high-rate microstep currently used;
that path is expensive.  The learned GPU operator is already trained and
profiled separately but is not yet called by the native viewer.  The next
step is to bind the selected GPU operator at the generic appendage contract,
run periodic exact microstepped trials, and publish its real duty telemetry.
A long fixed-hub/ground quiescence test should gate claims beyond the short
standalone probes already run.

## Prompt History

> "yes, let's swiftly but methodically in a master stroke connect the dots and compile, and do so knowing this sets the reference for network duty replacement for any local graph appendage with complex effect"

> "a great nod to what we're doing would be if it were apparent hub attachment was active when the network was in use, even if it flashed on and off with some like, statistical whatever to not slow it down, it would explictly show the balloon model, hence the hub actuator, was in use"

