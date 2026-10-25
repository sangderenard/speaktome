# Production gun on the arch full reference solve

**Date:** 2026-09-13

## Outcome

The arch-carried station now imports the existing production gun records from
`turret_production.build_gimbal_cannon_station` rather than authoring a second,
simplified weapon in `sled_reference`. The selected 20 mm liner assembly keeps
the production 120 mm outer tube, six 305 mm square recoil-tank/journal modules,
four-sided journal shoes, original spring/oil/MR absorber pack, chambered
washers, and 80 coolant-jacket chambers. The old `weapon.*` gun records are no
longer emitted by the active build path.

Only the mounting interface is new: the production fine-rig base, trunnion and
elevation anchor are bolted symmetrically to the upper arch platform. The gun's
declared case-ejection clear volume exposed three conflicting platform members;
those were removed, leaving an open chute and a moment-framed platform. The
combined standalone graph passes `ProductionGraph.check()` with no findings.

`LiveStructure` now treats the complete six-DOF Timoshenko system as the
reference solve. It advances all free physical coordinates with an implicit
average-acceleration Newmark step. The full mass-orthonormal eigenbasis is used
as the baked effective-inverse matrix: no elastic or mechanism vector is
discarded. The full station has 577 nodes, 1,408 edges, 1,187 beam members,
3,414 free DOFs, 3,232 positive-stiffness vectors, and 182 mechanism vectors.
The reference assembly/eigendecomposition took about 54 seconds and then ran
the OpenGL viewer at about 2 fps on this host.

The live viewer is shot-driven at idle. SPACE fires the actual 20 mm liner;
holding G supplies an explicit 15 kN user proof load through the same nodal
force boundary. R resets dynamic beam and joint state. Strain coloration is
computed from the current complete displacement. The obsolete scripted fold
composition was removed, and the fake 20/120 mm hot swap was removed because
changing the projectile without changing the installed liner was incoherent.

The production recoil edges were given explicit constitutive ownership:
spring-recuperator, oil-orifice, and MR-yield. `GraphJointForces` remains a pure
force evaluator; `LiveStructure` owns integration. `joint_bank_for` now accepts
the production record's `recoil_stroke_m` field.

## Validation

```text
python -m pytest tests/test_station_reference_motion.py tests/test_structure_native.py -q
14 passed, 1 warning, 3 subtests passed in 85.90s

python -m pytest tests/test_station_reference_motion.py -q -k live_beam
1 passed, 7 deselected in 31.88s

python -m pytest tests/test_station_reference_motion.py -q -k baked_reference_inverse
1 passed, 8 deselected in 19.04s

standalone sled graph: 353 nodes, 825 edges
square recoil modules: 6
coolant jacket chambers: 80
production absorber edges: spring, orifice, magnetorheological
old weapon.* nodes: 0
ProductionGraph.check(): []

full station reference assembly:
577 nodes, 1408 edges
3414 free DOFs = 3232 elastic + 182 mechanism
viewer reached live loop and reported 2.0 fps at idle
```

A later attempt to repeat the production-gun test while the viewer remained
open was blocked before physics setup because Windows would not overwrite the
native `rot` kernel DLL already loaded by that viewer.  The viewer was left
running.  A separate two-node reference-operator test then verified that the
complete baked basis reproduces `M^-1` on every free coordinate at zero step
size.

The complete station still reports two pre-existing underconstrained outer
stand-leg anchor warnings during the station-level check. These are outside the
gun transplant and remain to be resolved rather than suppressed.

## Prompt History

The user required:

> under all running conditions every moment goes through beam evolution.

> there are not stages it needs to just solve the physics

> you will use it you will not use it to model you will use it. you will put
> the former sled and actuators in place as they were and put that on the arch
> system's gun platform

> this repo is about reproduction at multiple scales through baking a matrix,
> consider this the full reference solve of the system

The workspace instructions required checking prior experience reports, adding a
new report, and validating the guestbook.

## Next steps

Use trajectories from this complete reference system to validate any smaller
baked reproduction matrix. Do not let a reduced solve redefine the force laws,
droop equilibrium, moment transfer, or joint state. Resolve the two outer stand
anchor warnings. Then return to the original EngineCycleSim full-native SSA
state-ABI task; its first genuine blocker remains the managed Python object
fields for hole emitters, bursts, and ordnance.
