# Station first-shot frame freeze audit

**Date:** 2026-09-13

## Outcome

The live station was reproduced with one idle frame, a synthetic SPACE event,
and a required second display flip.  The second flip never occurred because
the recoil-loaded call to `LiveStructure.step(0.05)` diverged while evaluating
`turret.equilibrator`; execution had not yet reached `ProjectileField.step`.

The joint-bank scheduler reported the following sequence across successive
beam substeps:

```text
velocity_m_s       requested joint substeps
 9.10384911                 10
-370.342153                386
-15629.4715              16281
-4377290.96            4559679
```

The reproduction aborted diagnostically at 4,559,679 requested iterations.
This is runaway explicit force/body coupling, not an SDL deadlock and not a
ballistic compilation pause.

The immediate semantic mismatch is in `structure_native.joint_bank_for`.
The graph declares the equilibrator with `stiffness_n_per_m`,
`preload_force_n`, `compression_damping_n_s_per_m`, and
`rebound_damping_n_s_per_m`, but the loader reads `spring_rate_n_per_m`,
`spring_preload_n`/`preload_n`, and an orifice.  It therefore ignores the
declared 8 kN/m spring, 400 N preload, and 300/420 N s/m damping and silently
synthesizes a generic gas-over-oil strut.  At the measured velocities its
quadratic default-orifice force escalates from about 78.7 kN to 130 MN,
232 GN, and 18.2 PN.  The scheduler then subdivides only the strut's internal
compression update while the beam velocity is held fixed, so it cannot
stabilize the coupled body response; it only turns the divergence into
millions of Python iterations.

Separately, the pending station cooling work was checked.  Physical frame
mounts were added for all eight remote heat-rejector cores and blowers.  The
cooling-focused tests passed (21), the station powerplant integration test
passed, and `ProductionGraph.check()` now has only the eight pre-existing
outrigger-anchor/bump-stop findings rather than sixteen new rejector findings.

## Validation

```text
python -m py_compile station_cooling.py station_powerplant.py hcu.py station_reference.py
python -m pytest tests/test_station_cooling.py tests/test_station_powerplant.py tests/test_hcu.py -q
21 passed

python -m pytest tests/test_station_reference_motion.py::StationReferenceMotionTests::test_each_powerplant_has_four_beam_mounts_on_its_bay_platform -q
1 passed

full station ProductionGraph.check(): 8 pre-existing findings only
```

One concurrent station test initially failed because the diagnostic viewer had
the cached `rot` DLL loaded and Windows denied overwriting it.  After the
diagnostic process exited, the same test passed.

## Prompt History

The user narrowed the investigation to:

> the only thing you have to be involved in is finding out why the freeze happened this is a fuckign quagmire

While the long reference solve ran, the user asked to continue the already
specified isolated cooling and remote heat-rejector work:

> work on the above while waiting for any compiling

## Next Step

Do not touch ballistics to repair this freeze.  Make the `spring-damper`
constitutive path consume its declared linear stiffness, preload, and
directional damping names, and couple/substep the joint force with the beam
state rather than internally repeating a fixed-velocity force evaluation.
Add a regression requiring the actual SPACE path to reach the next display
frame with finite equilibrator velocity and a bounded substep count.

## 2026-09-13 continuation: stops, telemetry, and structural participation

The first-shot path now advances the compiled interior-ballistics pressure+force
history in impulse-preserving 50 microsecond segments through the complete
beam/joint state. The projectile carrier is visual/flight state only and does
not apply a second recoil force. A full hidden OpenGL acceptance reached a
finite second frame after SPACE, although the complete reference frame remains
slow.

Both parallelogram swing stages now have graph-declared, four-face-equalized
stops: an adjustable full-forward preload stop and a fixed full-aft maintenance
stop. `GraphJointForces.command_platform_preload_force` accepts total stage
force in newtons, divides it over the two parallel hydraulic rams, and closes the
forward stop by the installed (not maximum) preload setting. Contact step size
is resolved at sixteen force/body exchanges per local contact time constant;
no clipping or reduced stiffness was introduced.

`LiveStructure.inject_solver_stats` was added. Every solver node receives
position, displacement, rotation, velocity, acceleration, and their rotational
counterparts. Beam edges receive axial/bending strain, stress, and yield
utilisation; managed joints receive constitutive force. The same records feed
strain colour, engine render objects, HCU analysis, and sustained-push extrema.

A nonstructural/gestalt boundary was also added. `ProductionGraph.node` now
declares `structural_participation` (`auto` by default); a baked subobject may
declare `solver_condensed_into` and optional `solver_condensed_mass`. Its own
beam DOFs disappear, its mass and parallel-axis inertia can be transferred to
the gestalt, and it inherits the gestalt's solved rigid kinematics for display.
This removed routed channels, drum motor internal rotors/service hubs, HCU
service fittings/regulators/sensors/bottles, the latched blast panel, and
metaconduit heads as independent solver bodies while retaining their render and
machine state.

An attempted broad interpretation of `beam_solvable=False` was rejected after
it increased zero modes from 88 to 344: that legacy flag is also used on exact
rigid construction ties. Node participation/condensation is now the explicit
boundary, and those rigid ties remain assembled. With the corrected pass the
full 670-node station has 608 structural nodes, 62 excluded internal/routed
nodes, 1,320 physical members, 3,588 free DOFs, 3,586 positive elastic modes,
and two negative generalized eigenvalues (-860.069 and -195.126). Their diffuse
shapes are led by the filler sleeve/utility housings and fine platform/cooling
plant respectively; they are not the two parallelogram motions. The first
positive elastic frequency is 1.144498485 Hz.

Focused validation after the pass:

```text
python -m py_compile frame_solver.py graph_columns.py live_scene.py graph_physics.py sled.py station_reference.py surfaces.py hcu.py station_cooling.py turret_production.py
python -m pytest tests/test_station_reference_motion.py -k "explicit_nonstructural_node_flag or baked_subobject_mass_and_motion or live_solve_injects or each_swing_has_equalized or platform_preload_can_be_commanded or triple_recoil_pack" -q
6 passed, 22 deselected
```

The exact remaining blocker is the indefinite full-frame generalized stiffness
reported by `FrameSolver.modes` near `frame_solver.py:648`: the two negative
eigenpairs are currently classified as zero-frequency mechanisms and would be
integrated without restoring force. Do not damp or threshold them away. Trace
the conditioning/assembly contribution (especially approximate rigid links and
single-housing mounts), preserve the positive-semidefinite beam law, and only
then rerun the sustained barrel-push and live shot.

## 2026-09-13 continuation: exact surface condensation and lower-room framing

The negative generalized eigenpairs were numerical conditioning, not negative
element energy.  Their scale was driven first by hundreds of 0.4 kg prism
surface ports solved as independent six-DOF bodies. `FrameSolver` now performs
rigid kinematic endpoint condensation: a surface point follows its gestalt by
`u_port = u_body + theta_body x offset`, while attached member stiffness,
preload, gravity and external force are transformed back to body force **and
moment**.  This retains screw-in washer and mount load paths at their actual
surface offsets without retaining the port as a separate body.  Prism ports,
annulus rim ports, drawing discs, barrel coolant chambers, and the filler
sleeve now declare the appropriate gestalt boundary.  Drawing-only disc ties
use the new unambiguous edge-level `structural_participation=False` declaration.

The full station spectrum is positive after this change: zero mechanism or
negative eigenpairs.  The initial 457 MHz weapon-port ceiling fell to 13.4 MHz;
condensing annulus drawing bodies/rims reduced it to 3.06 MHz.  A subsequent
high mode revealed that the physical 4340 arch legs, crowns, cross-ties and pin
bosses were incorrectly tagged as ideal rigid links despite carrying real
sections and damage laws. Removing those contradictory flags makes every arch
moment traverse its Timoshenko beam section and reduced the ceiling to 363 kHz.
The outrigger deployment lug was then found to be a point on the upper leg but
modeled as an independent body and a rigid triangle to both top and telescoping
pad. It is now an offset/mass-condensed upper-leg point; the hydraulic deploy
ram still applies at that 1.6 m lever. The spectral ceiling is now 133 kHz and
is led by the drum thrust bearing, with no negative eigenpairs.

The previously absent lower-room structure is now explicit: four light
vertical hangers and fore/aft end crossmembers around the open workroom, plus
four light vertical hangers, two end crossmembers and two side rails around
each open engine-lowering bay. These hangers end on neither pads nor outrigger
pivots and explicitly declare `not_ground_support=True`.

Cooling tanks are now instances of parametric `CylindricalServiceTank` rather
than fixed boxes. Capacity determines cylinder length and total filled mass.
The two 500 L potable tanks lie horizontally on floor cradles beneath 0.92 m
service counters along the room sides, face inward, and leave a 1.51 m central
aisle. Their cradle forces enter lower-room side rails rather than one long
diagonal member to one corner. Four-point lower-frame racks carry the smaller
cooling reservoir, pump and exchangers. This cut the false potable-tank gravity
deflection from 1.01 m; condensing the barrel filler sleeve then moved the
largest gravity displacement to the actual gun/elevation structure at 0.157 m.
This is elastic/static droop and is separate from the roughly metre-scale
permitted recoil/parallelogram articulation.

Focused checks pass (six structural-boundary/framing checks, three cooling and
tank-sizing checks, and three recoil/condensation checks). A full 34-test
`test_station_reference_motion.py` run passed its first twelve tests and remains
inside the intentionally long `test_force_push_settles_the_complete_unlocked_rig`
at the time of this note; do not report the full file green until it exits.

## 2026-09-13 continuation: component-mode atlas boundary

`component_mode_atlas.py` now provides the first deliberately separate
reduced-scale structural ABI. It implements Craig--Bampton component mode
synthesis: component interface freedoms remain literal physical coordinates,
static constraint modes preserve interface equilibrium, and only internal
elastic modes may be truncated. A full-retention atlas is an exact change of
basis and is the mandatory comparison reference. An internal mechanism is
rejected rather than pseudo-inverted or discarded; it must be promoted into
the explicit interface/articulation state.

The numerical tolerances were audited while adding this boundary. Beam and
node participation remains categorical (`structural_participation`) and there
is no strain-based member culling. The unused `beam_candidates` slenderness
cutoff therefore does not alter the solve. The atlas's internal-zero test is
only 64 machine epsilons relative to its local eigenvalue scale, so a very soft
positive mode remains physical and test-covered. Atlas truncation occurs only
when a caller supplies an explicit `internal_mode_count`, and its first omitted
mode produces a reported flexibility bound.

Multiple component atlases can now be scattered onto shared interface DOFs.
A two-component regression proves that their assembled reduced mass and
stiffness reproduce the equivalent monolithic matrices exactly when all
internal modes are retained. The station graph has 29 assigned structural
assembly labels, no unassigned structural members, and 156 shared nodes. Some
are intentional three-way interfaces, so the next step is per-member component
matrix assembly; slicing the already-summed global matrix is not valid.

Validation:

```text
python -m pytest tests/test_component_mode_atlas.py -q
5 passed

python -m pytest tests/test_component_mode_atlas.py tests/test_station_reference_motion.py -k "component_mode_atlas or explicit_nonstructural or live_reference_basis_is_complete" -q
7 passed, 32 deselected
```

Prompt history:

> check the tolerances on the exclusions from beam vs strain

> if the math works like a sum of modes, can you solve each piece's mode atlas and assemble them with speed?

> proceed carefully

Next step: make `FrameSolver` emit stiffness and mass contributions grouped by
edge `assembly`, allocate each nodal/condensed mass exactly once, promote every
graph-joint endpoint to an interface, then compare a full-retention assembled
station atlas against the current complete reference inverse before permitting
any modal truncation in the viewer.

## 2026-09-13 continuation: shaped floor grillage and drum columns

The lower-room plate had remained a 10 mm A36 plate hung from four 24 mm A36
tubes. It is now a continuous 20 mm HY-80 welded plate (about 1.413 tonnes),
supported by two 420 x 240 mm fabricated HY-80 I girders along the bore axis
and five segmented 300 x 200 mm transverse diaphragms. Every intersection is a
shared plate-grid node, so the beams form a graph load path rather than merely
crossing in the render. Four fabricated HY-80 columns rise from the spine /
diaphragm intersections to cap nodes beneath four quarter points of the static
drum seat ring. Stout front and rear cap diaphragms give every column head a
third physical connection. Existing upper seat posts remain as a redundant
parallel path.

This required fixing an older solver fiction before using shaped members.
Plate emission had calculated anisotropic strip properties but discarded them,
and `FrameSolver` collapsed every member to one circular second moment with
`J = 2I`. Production edges can now merge explicit section properties into the
same damage record consumed by frame/material physics. The 3-D Timoshenko
element carries distinct `Iy`, `Iz`, and Saint-Venant `J`, and `section_up`
fixes the authored strong-axis orientation. Static and live strain recovery use
the appropriate extreme fibres. The renderer builds a welded I section as its
actual web and two flange plates. This is section-aware beam analysis, not full
AISC rolled-member, local-buckling, lateral-torsional-buckling, or warping
design; those checks remain separate work.

Validation:

```text
python -m pytest -q tests/test_station_reference_motion.py tests/test_component_mode_atlas.py
43 passed, 1 warning in 56.71s

python -m pytest -q tests/test_station_reference_motion.py -k "oriented_i_section or welded_i_section or lower_room_and_engine_lowering_frames or drum_has_four_columns"
4 passed, 35 deselected, 1 warning in 21.34s
```

The complete station remains finite in planted and raised gravity solves (996
physical members, 0.09004 m maximum displacement at the remote heat-rejector
diffuser). The new drum columns carry up to 11.73 kN in the reference gravity
pose. Under a deliberately sustained 416 kN rearward barrel load, the new
HY-80 shaped members peak at about 3.98% of yield. The unlocked mechanism moves
about 1.07 m under that sustained load; this was retained as articulation, not
hidden by a support lock or a solver clamp.

Prompt history:

> is this defaulting to mild steel? there's no pillars under the drum anymore.
> Is there any way to put pillars on the drum underside to the floor and ensure
> that floor is continuous and strong, strong enough to bare weight through the
> colums under the drum if the floor is elevated. investigate I beams or other
> shaped beams instead of just tubes,

> it would not be a bad idea AT ALL to have two grand beams along the long axis
> and stout cross members for the floors and for ballast weight for the large
> gun above

Next step: run the time-resolved real recoil profile through the revised full
station and record peak shaped-member stress and column buckling margin in both
planted and raised firing poses. Do not infer that check from the static 416 kN
witness, and do not add warping stiffness to the existing Timoshenko element.

### Live colour control repair

The documented `1 / 2` yield/assembly colour control was dead: both live loops
had no key handlers and unconditionally installed yield-band materials every
frame. They now start in the authored assembly/material palette, `1` selects
live yield utilisation, and `2` restores assembly colours. The shared material
selection is a pure tested function; only solved structural-member triangles
are recoloured. Focused validation: `3 passed, 37 deselected`.
