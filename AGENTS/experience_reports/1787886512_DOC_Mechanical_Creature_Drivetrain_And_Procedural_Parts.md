# Mechanical Creature drivetrain and procedural parts

**Date:** 1787886512
**Title:** Mechanical Creature drivetrain conservation, audio, and procedural parts

## Overview

Continued the bespoke MechanicalCreature page in `turing`. The work corrected
nonphysical drivetrain limits, preserved torque reaction paths, connected the
resident engine-audio kernels, and expanded the vehicle's data-driven mechanical
presentation without putting DOM work into physics authority.

## Steps Taken

- Removed the clutch launch cap, reflected engine inertia during clutch slip,
  the wheel-speed saturation, and the duplicate generic throttle slew.
- Restored omitted tire, braking, and rolling torque reactions in the chassis
  pitch wrench balance.
- Added engine architecture/energy metadata and a resident PCM kernel bank.
- Added a routed throttle cable/table actuator and medium-rate energy paths.
- Added procedural tires, helical coilovers, engine cylinders, fuel/battery
  storage, ignition pieces, and differential brake rotors/calipers.
- Declared part masses, centers, principal inertias, and differential-brake
  polar inertia without double-counting the existing lumped vehicle mass.
- Ran focused symbolic conservation tests, Python compilation, and JavaScript
  syntax validation.

## Observed Behaviour

The focused drivetrain conservation and slipping-clutch tests pass. The
generated JavaScript source parses under Node. The differential brake torque
already acts before each axle differential and returns its reaction through the
differential housing; rotor angular momentum is now declared in the graph but
is intentionally marked for a later coupled driveline mass-matrix pass rather
than approximated with a fake per-wheel inertia.

## Lessons Learned

The original low-speed feel was not one problem: it combined a clutch torque
cap, double-counted reflected engine inertia, and an explicit wheel-speed
saturation. Presentation meshes can remain inexpensive when generated from the
same graph metadata, while mass/inertia declarations must distinguish future
part aggregation from the currently authoritative lumped mass.

## Next Steps

- Complete the coupled driveline mass matrix so differential-brake rotor
  momentum participates in axle acceleration without an incorrect per-wheel
  approximation.
- Promote the declared wheel-bearing/knuckle/rotor/caliper stiffness and
  failure contract into the structural solver so the one free spin axis never
  becomes an artificial floppy wheel-end joint.
- Add a leveling pose bank (neutral, climb, descent, side-hill, articulation,
  service/jack, and user poses) whose chassis and four hub targets are solved
  through bounded linkage/actuator motion rather than coordinate teleporting.
- Add topology-preserving wheelbase morphology edits using tagged chassis
  cross-sections, axle-local subgraphs, extensible longitudinal members, routed
  guide paths, and mass/inertia recomputation.
- Recompute coilover presentation metadata (preload collar position, wire
  radius and active turns) from live rest-length/stiffness controls. The hub
  solve and WASM spring law already consume the live parameter without SymPy
  recompilation, and A-arm member lengths correctly remain fixed.
- Replace the current declared reservoir manifold hook with an authoritative
  medium-rate carrier/air/knock/flow model before mixture controls affect torque.

## Prompt History

> "thrashing javascript and assembly or shaders is not acceptable, nor is pretending async is a thread"

> "pitch should only be emergent torque response can we not have fake forces please"

> "also it's really disappointing how you can floor the revs and no matter what engine the wheels are locked at slow speed"

> "can you get our engine synth connected for the next version too or is it too late"

> "can I suggest a nice candy red for fuel tanks, and electric blue for batteries"

> "make a note while you do this sometime that the diff brakes need their flywheel momentum as a real part of the powertrain graph, and the calipers will go in wheels next"

> "make a note that we need the mechanical graph to include the wheel bearing knuckle and rotor caliper system so we don't leave an artificial flimsy joint there"

> "giving the LVL system a pose bank and pose control that makes it easier to set the location of each wheel and the chassis"

> "make sure we can identify someplace to stretch wheelbases and maintain connections correctly, make sure the positions of things are graph relative"

> "when the rest length of the spring in the suspension is edited, the a arms remain their usual size, so does the shock ... can we make sure the presentation and the physics both align"
