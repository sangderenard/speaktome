# Fabricated straps, external station fuel, and shared fluid routing

**Date:** 2026-09-13

## Outcome

The production gun and station powerplant now use explicit fabrication and
service graph records instead of decorative centreline stand-ins.

`engine_toy/fabrication.py` adds a convex outside-perimeter strap generator.
It produces two bent 4130 plate halves and two independently torqued side
closures; there is no hinge or pin.  The convex wrap deliberately bridges
narrow concavities that cannot be forged, fitted, or tightened.  The gun's
barrel-jacket-to-square-spine bands use this generator.  The old coincident
trunnion ties were replaced by six spatial filler-metal weld elements at the
actual touching outside seams.  `weld_touching` rejects remote centreline
connections presented as welds.  The station import filter now retains both
fabricated straps and welds.

The strap tensioners enter the existing spring assembly with their
torque-derived preload.  Strap contacts enter the linear solve in their
locked, below-slip state and publish axial/tangential slip capacity.  The
current nonlinear solve does **not** yet release a strap contact after its
capacity is exceeded; this is an explicit remaining mechanics boundary, not
a claimed feature.

The station no longer supplies either installed engine from a synthetic
`fuel.tank` or a synthetic aggregate pump bank.  Four parametric 50-US-gallon
drums are strapped behind the rear posts, two ULSD for the multifuel and two
Jet-A kerosene for the turbine.  Each has its own drop-in cap pump, dry-run
auto-off, check valve, water-separating pickup and quick disconnect.  Separate
fuel-typed manifolds prevent liquid blending.  `FuelControlUnit` (FCU) stages
only enabled, compatible, nonempty pumps and supports attaching/detaching a
newly rolled-up barrel/pump pair.  `EngineCycleSim.external_fuel_supply` and
the drivetrain graph consume the same four physical reservoir/pump identities.

`engine_toy/fluid_routing.py` introduces one shared routing contract:

- `rigid-orthogonal`: rectilinear pipe, 90-degree interior elbows, tee
  fittings derived from graph degree, and an obstacle-aware Manhattan search.
- `flexible-hose`: a sampled sagged path whose physical length includes
  declared slack.
- `trivial`: a thin conceptual trace that reports Manhattan distance without
  pretending to be installed pipe.

The route is stored as ordinary edge waypoints plus `route_length_m`, so the
vehicle mesh and fluid thermal/volume calculation consume the same path.
The full station currently classifies 323 fluid edges: 261 orthogonal and 62
flexible.  Four new barrel pickup hard lines are explicitly obstacle-audited.
The 257 legacy undeclared hard lines receive quick deterministic orthogonal
routing and `route_requires_obstacle_audit=True`; they are not falsely
certified clear.  The assembled route contains 156 interior 90-degree elbows,
237 edges incident to a tee port, and 395.41 m of reported line/hose length.

The service-detail filter was corrected to retain `station.fuel.*`, and the
still renderer now draws routed edge waypoints.  The detailed still was
successfully regenerated at
`C:\dev\Powershell\engine_toy\station_powerplant_detail.png` through the same
station graph, two baked engine meshes, and two engine-cycle simulation
objects used by the live viewer.

## Validation

```text
python -m py_compile fluid_routing.py vehicle_mesh.py drivetrain_graph.py
    station_reference.py station_powerplant.py

python -m pytest tests/test_fluid_routing.py tests/test_station_powerplant.py -q
10 passed

python -m pytest tests/test_station_reference_motion.py -q -k each_powerplant
1 passed, 39 deselected

python station_reference.py --powerplant-still
completed; both engines baked 24 moving frames; PNG written

full station route inventory
323 edges; 261 rigid-orthogonal; 62 flexible-hose; 4 audited hard lines;
257 pending legacy audits; 156 elbows; 237 tee-connected edges; 395.41 m
```

`git diff --check` reported only the repository's existing LF-to-CRLF notices,
not whitespace errors.  The full station dynamics suite and the long native
EngineCycleSim lowering were not rerun in this continuation.

## Prompt History

The user corrected the initially misunderstood strap closure:

> no pin, bolts on both sides, makes it truly adjustable and more in line with a forging slugs and reloading shells outfit

The user required external barrel fuel rather than hidden engine-local fuel:

> if you could, while doing everything you've been asked to do, also check on whether we're giving these engines fake gas tanks, and if so, make sure the engine engine doesn't do that, and give us some 50 gallon barrels strapped to back posts with fuel pumps

> I'm thinking we make barrel fuel pumps a think, auto-off drop in the cap good to go, keep several behind the mount, any barrel rolled up can be poured into the mounted barrels or just set around, slap a pump in and it's part of the system, going to a common manifold for fuel type

The user named the controller boundary:

> then like the HCU we have the FCU to manage multiple sometimes on pumps

The user requested the shared fluid-routing distinctions:

> let's make sure all our fluid edges can be straight and 90/tee, fluid edges that can flex, and fluid edges that are trivial, tubing or something, it's routing makes little difference to anything, then the 90 and tee would need an edge traveling algorithm to manhattan distance itself around things to get from A to B with minimal hanging stub anywhere, flexible ones would form hoses with some slack, and trivial ones would just be barely represented mostly conceptual "free" edges that only need to report a distance, which we can manhatan without needing the actual mesh edgefinding algorithm

## Next Steps

Migrate legacy fluid edges by subsystem, explicitly choosing hard pipe, hose,
or trivial routing and clearing `route_requires_obstacle_audit` only after the
obstacle-aware search succeeds.  Begin with the station service trunks and gun
cooling runs because those cross the largest moving/maintenance envelopes.

Then add a nonlinear strap-contact regime change: when resolved tangential or
axial contact demand exceeds the published friction capacity, release the
corresponding contact freedom without deleting the strap or its tensioner.
Regression coverage must prove locked below capacity and physical slip above
capacity.

Finally return to the original full-native objective.  Reproduce whole-module
`EngineCycleSim.step` lowering with `compile_contract.py`'s full-native
ExtractionContract and record the first genuine compiler blocker.  This
continuation changed construction-time external fuel graph state but did not
claim any additional part of `step` lowers natively.

## Placement and inlet correction

The initial barrel placement added positive Z to nodes on the negative-Z rear
post row, which put the barrels on the occupied-room side of the posts.  It is
now negative Z: each cylinder's frontmost surface is behind its post's rear
surface.  The fuel vessels also lacked `shape="drum"`; despite carrying drum
axis/radius/length, the renderer therefore fell back to misleading gold boxes.
They now render as vertical cylindrical 50-gallon drums with two strap bands.

The station manifold previously ended at `engine.<side>` and merely named an
`external_fuel_in` string in metadata.  Two real condensed surface-port nodes
now exist, one on each managed engine object.  Each typed manifold terminates
on its port.  The engine drivetrain graph retains that same identity and
continues from it to `powertrain.fuel_rail` (or the carbureted bowl where
applicable).  Focused drivetrain tests prove both physical cap pumps reach the
inlet and the inlet reaches the rail.

The old visible engine cuboid has not been retained.  `engine.port` and
`engine.starboard` are managed engine object boundaries for condensed
mass/inertia/state, declare `render_primitive=False`, and are displayed only
through their engine-owned baked meshes on the lowering pallets.  The other
powerplant objects are the separately requested generator, battery, hydraulic
pump, air compressor, refrigerant compressor, coolant reservoir and coolant
pump; none stands in for an engine.

The revised still completed at the same path.  A subsequent complete-station
test attempt encountered Windows `Permission denied` replacing the cached
Turing `rot.dll`, because another live process has that DLL loaded.  The live
process was not stopped.  The five fuel/FCU tests passed; the full graph had
already built and rendered successfully immediately before this lock-only
failure.

## Fully released gravity audit and nested pins

`O` in the live viewer was found not to be a structural unlock. It only
commands both platform packing rams to zero preload and withdraws their
adjustable forward stops. The four arch-crown positive height locks remained
assembled because their `lock_engaged` field had been descriptive. Structural
membership now treats `lock_engaged=False` as an actual exclusion, with focused
coverage.

`station_reference.unlocked_motion_document` makes a separate analysis pose
which releases the four crown locks, removes both platform-actuator preloads,
and withdraws the forward preload stops while retaining installation locks.
The initial audit solved zero-g and one-g linear beam endpoints and interpolated
their strain. It first appeared finite with 90.0357 mm maximum displacement,
but subsequent strengthening exposed the zero-stiffness mechanism that the
thin attachments had accidentally regularised. Therefore these endpoint
crossings are retained below as the evidence that found the bad parts, not as
valid certification of a freely moving one-g equilibrium.

The independent sand-standing heat rejector yields first at 0.396293436 g in
`station.heat_rejector.blower_inlet_mount.1/.2`. It remains in the total site
solve but is reported separately because it cannot explain arch-craft motion.
The arch craft first yields at 0.560638027 g, simultaneously in several end
segments of the 65 x 10 mm 4130 shaped straps joining the outer barrel jacket
to the square spine. The proportional intermediate result establishes that
this crossing is gravity-driven, not stored zero-g assembly energy. A real
post-yield/contact-slip transition is the next required physics step before a
truthful one-g dynamic settle. Resolved at the reported crossing, the strap
carried only -0.87 kN axial load; its approximately 263 MPa shear term, not
primary gun-path axial load, dominated. The old 1.77 m blower member likewise
carried only -4.1 kN axial while accumulating approximately 375 MPa shear.

The blower-to-inlet centroid sticks have now been removed. Each blower has
four separated condensed surface ports and four 80 mm flange standoffs into
its core casing. Production gun straps are now 100 x 20 mm 4130 forgings with
M20 10.9 closures at 430 N.m. The strap-tensioner retains axial preload as its
constitutive spring while the clamped lug faces and bolt bearing remain rigid
in shear and rotation below slip; releasing all rotations had left a genuine
free closure-lug mode. An isolated regression now requires zero strap closure
mechanisms.

`FrameSolver.applied_force_vector` is now the single construction of gravity,
authored loads and linear preload. `LiveStructure(settle_from_authored=True)`
can apply that same vector continuously from the authored pose so nonlinear
damper/joint laws participate instead of beginning from a singular linear
"settle". Live utilisation now includes shear in von Mises stress and excludes
ideal rigid/contact constraints from material-yield coloration. A diagnostic
dynamic run remained finite through 0.2 s (13.6 mm maximum motion), but was
stopped because the pre-fix utilisation display was assigning steel yield to
ideal strap-contact links. The corrected rerun remained finite through 0.1 s:
0.274 mm at 0.05 g and 1.834 mm at 0.10 g. The first actual material hot spot
was `turret.evacuator_port.4`, rising from 37.5% to 305.7% during that
dynamic ramp. Inspection showed this was not a branch tube or attachment at
all: it is the raked gas hole between the spine-housed evacuator passage and
the bore. Its old `rigid-distance` spelling had invented a structural member
and material capacity for a void. It is now an `exhaust-flow-path` with flow
area only, no damage record or beam participation. No nozzle or attachment
object was retained; the evacuator and outer-barrel bodies own the surrounding
metal.
Each passage now also declares clean/open initial service state,
`cleanliness_fraction`, `fouling_mass_g`, and `occluded_area_fraction`. These
are explicitly condition records only: stage 8 bore-gas recovery still lacks a
per-edge resistance consumer, so no flow effect is claimed yet.

The support-control path was also found to stop short of the beam engine. The
HCU drove `OutriggerSet` and obtained real per-leg hydraulic force, but those
forces previously affected only the standalone actuator state. `LiveStructure`
now accepts externally owned axial edge forces and distributes them as equal
and opposite endpoint wrenches. `apply_support_hydraulic_forces` maps the
existing `OutriggerSet.step` result to the eight authored `stand.leg.*` edges.
Outrigger positive-length collars now carry operational `lock_engaged` state;
`active_leveling_document` opens them while retaining the rest of the unlocked
gun pose. This is the required HCU -> hydraulic actuator engine -> beam engine
bridge for a levelled settle; it has focused force-direction coverage, but the
full HCU-controlled settle has not yet been rerun.

The lower parallelogram now uses four 55 mm radius x 180 mm long cylindrical
arch pins. The upper gun platform uses four inboard 40 mm radius x 120 mm long
cylindrical pins and a narrower link, so the second level nests inside the
first and outside the arch envelope rather than overlapping it.

Focused and neighboring regression results:

```text
python -m pytest tests/test_station_reference_motion.py tests/test_fabrication.py -q
45 passed, 1 warning in 63.10s

python -m pytest tests/test_fabrication.py tests/test_station_cooling.py -q
9 passed, 1 warning in 11.30s

python -m pytest tests/test_station_reference_motion.py -q -k \
  "live_authored_settle or live_utilisation or disengaged_lock"
3 passed, 41 deselected in 7.96s
```

## 2026-09-14 live-settle correctness continuation

The reported 12,665% peak was reproduced before any shot or preload release.
It was not supported by the energy: the complete initialized beam state held
only 2,527 J, while live recovery reported 57.1 GPa / 124.05 yield on
`turret.breech_to_barrel`. The frame assembly already condensed prism surface
ports through the correct rigid-offset MPC, but `LiveStructure.member_response`,
rendered positions, joint endpoint kinematics, and telemetry read the condensed
ports' unused fixed DOF slots directly. That made every surface attachment look
like it was tied to an invisible world anchor. `FrameSolver.endpoint_motion`
now materializes all graph points through their master transforms, and all live
consumers use it. The false breech and coolant-boss peaks disappear; the same
initial solve then reports 3.20 yield on a swing damper rather than 124.05.

The platform dampers and packing rams were also double-counted: their axial
spring/damping/command law was active, while the drawn telescoping cylinder was
simultaneously assembled as a fixed-ended transverse/bending beam. Those
clevis-ended force elements and their contact markers now have no beam
participation; their constitutive force and endpoint reactions remain active.
This exposes a genuine mechanism freedom instead of hiding it with cylinder
bending stiffness.

Static reference loads are no longer discarded along a released nullspace.
`LiveStructure.base_force` is now the exact `f - K u_static` residual. It is
zero for an exact static equilibrium and drives any mechanism component a
least-squares static solve could not balance. The constant modal projection is
baked once, avoiding a dense 3426-square projection on every constitutive
exchange without thresholding or changing the force. Beam/reference kinetic,
elastic, and constant-load potential terms are now injected into solver stats.

Each swing platform now has a 3x3 grid of 12 mm steel deck sheet strips welded
to all four perimeter corners. Overlapping upper/lower sheet stations carry
unilateral vertical contact laws, so gravity can seat the gun platform on the
lower platform and the two structural surfaces cannot pass through one
another. An interim abstract hard-forward stop was removed in favor of these
physical plate contacts.

Urgent HCU mode now builds the beam solve with only the outrigger positive-
length collars open and applies `OutriggerSet.step()`'s per-leg hydraulic forces
to the real support edges each frame. Weapon/crown locks are not opened by this
support-only leveling document.

New focused results:

```text
python -m pytest tests/test_station_reference_motion.py -q -k \
  "solver_stats or platforms_have_structural_sheets or support_leveling or static_nullspace or live_recovery or complete_reference"
5 passed, 44 deselected in 8.08s

python -m py_compile frame_solver.py live_scene.py graph_physics.py \
  structure_native.py sled.py station_reference.py tests/test_station_reference_motion.py
PASS
```

## 2026-09-14 adaptive component runtime and owned assembly

The first runnable paired-representation component now exists as
`AdaptiveGestaltRuntime`. It advances the same physical component with the
average-acceleration Newmark method in either its full elastic atlas or its
six-coordinate rigid atlas. At a sleep/wake transition it uses a
mass-orthogonal projection, so recovered rigid pose and every resultant
linear/angular momentum represented by the rigid basis are continuous.

The sleep policy was strengthened after identifying a zero-strain failure
case: an oscillating member crosses zero strain at maximum velocity. Low
stress alone could therefore erase genuine ringing. Sleep now also requires
low internally measured kinetic energy. The runtime calculates that energy
from velocity residual to the rigid subspace rather than trusting scene
telemetry, and a focused axial-expansion witness refuses to sleep even when
reported stress is zero.

`FrameSolver.assemble_component_matrices` now assembles a component from its
owned member contributions before slicing local coordinates. Distributed
member mass follows its member, while body mass at shared nodes requires an
explicit fraction. This avoids the invalid shortcut of slicing the already
assembled station matrix, which cannot say which adjacent component owns an
interface contribution. A two-member witness reproduces the whole reference
stiffness and lumped mass exactly.

An assembly inventory identified `central-storage-ballast` as a useful first
station candidate: 18 structural members, 16 solver nodes and four shared
frame interface nodes (24 interface DOFs). Its exact component atlas builds
successfully with all 72 internal modes retained; the smallest fixed-interface
eigenvalue is positive (`2950.6889946549054 (rad/s)^2`). No station component
has been allowed to sleep yet because the certified interface-wrench stress
envelope and global state scatter/gather are not wired.

```text
python -m pytest tests/test_component_mode_atlas.py \
  tests/test_station_reference_motion.py -q
67 passed, 1 deprecation warning in 88.01s
```

A broader station-building test can currently fail because the already-open
viewer holds Turing's cached `rot.dll` on Windows and another process cannot
overwrite it. The viewer was deliberately not killed and no cache was removed.

## 2026-09-14 urgent-support runaway correction

The live screenshot with assemblies hundreds of metres apart exposed two
coupled errors in the new support bridge. `OutriggerSet` bounded its private
`LinearActuator.position_m`, but the beam graph received only pressure force;
the released graph leg had no representation of the pumped oil volume and
could run past the actuator coordinate. Each of the eight support legs now
owns paired, unilateral, finite-compliance oil-column contacts on its actual
axial graph coordinate. `apply_support_hydraulic_forces` moves both contact
faces to `LinearActuator.position_m - initial_actuator_extension_m` before
applying the actuator's balanced endpoint force. The rod therefore advances
only with delivered fluid volume while pressure and all reactions still pass
through the complete beam graph.

The full graph exposed a second error that the local actuator witness did not:
with eight axial collars open, the linear stiffness matrix is singular because
the support laws are nonlinear. Using `FrameSolver.solve()`'s least-squares
displacement as a supposedly settled reference selected an arbitrary point
along those released coordinates. The first live tick began from that invalid
pose. Urgent live mode now starts at the authored, post-assembled geometry with
zero beam strain and lets gravity, oil columns, joints, dampers and beams settle
together. This is not a visual clamp and does not suppress load: the exact
gravity vector remains the runtime `base_force`.

The complete 1,176-node check under an intentionally severe 300 kN on every
support for 16.7 ms changed from 231.0 m maximum motion, 440 m/s support speed
and 36,254x yield to finite 2.512 mm maximum motion. It retained a 1.391x yield
result under that artificial 2.4 MN total support load; that possible strength
failure was not upgraded, clipped or normalized away.

```text
python -m pytest tests/test_station_reference_motion.py -q -k \
  "solver_stats or platforms_have_structural_sheets or support_leveling or support_oil_volume or pumped_oil_coordinate or static_nullspace or live_recovery or complete_reference"
7 passed, 44 deselected in 11.18s

python -m py_compile stand.py station_reference.py \
  tests/test_station_reference_motion.py
PASS
```

## 2026-09-14 exact sheet boundaries

`surfaces.emit_plate` previously placed a full visual cell at every structural
grid intersection. A boundary station was therefore the centre of a full cell,
and the rendered sheet extended half a cell (often several decimetres) beyond
the plate span on every side even though the authored grid coordinates were
correct. Boundary stations now own inward-shifted half-cells while interior
stations retain centred full cells. `vehicle_mesh` renders these cells in the
plate's authored local axes. The union ends exactly at the declared corner and
span boundaries, including rotated plates; the structural wrench grid remains
on the weld coordinates.

The gun-platform deck sheets are explicitly top-mounted: their bottom face is
at the perimeter pipe's upper tangent, their plan boundary terminates on the
four supporting pipe centre lines, and `overhang_m` is zero. This shared
surface-emitter correction also removes the invented boundary margin from the
lower-room floor and both powerplant pallet sheets.

```text
python -m pytest tests/test_station_reference_motion.py -q -k \
  "plate_render_cells or platforms_have_structural_sheets"
2 passed, 50 deselected in 8.76s
```

## 2026-09-14 exact/reference efficiency and adaptive gestalt boundary

The beam participation machinery remains operational, but two meanings are
now reported separately. `structural_participation=False` is a true exclusion
and is appropriate for render-only/routed objects. `solver_condensed_into`
keeps a subobject's geometry and interface wrench while mapping its motion,
mass and inertia into one rigid gestalt master. The new
`structural_participation_report` counts deformable beams, rigid-condensed
edges/nodes and actual exclusions independently so a generator cannot claim
an optimization by silently deleting structure.

Several exact hot paths were simplified without reducing the reference solve:
structural edge election is one cached NumPy mask; actuator endpoint forces are
batched and scattered through the same rigid-offset MPC used by beam assembly;
four public motion states are recovered in one gather; the duplicate static
matrix assembly in `LiveStructure` is gone; and displayed elastic energy is
computed from pre-baked static/modal terms rather than a dense physical-space
matrix product each frame. Focused tests compare the vector actuator scatter
against scalar endpoint-wrench assembly, including the moment from an offset
condensed port.

No strain-based real-time culling previously existed. The component atlas's
`64 * eps` relative eigenvalue test diagnoses arithmetic zero only. The frame
solver's `1e-6 (rad/s)^2` value is an absolute mechanism diagnostic (about a
0.000159 Hz mode), not a utilization cutoff; neither was repurposed.

`component_mode_atlas` now defines the dormant half of a reversible adaptive
lane. `RigidGestaltAtlas` maps every component point to six rigid coordinates,
condenses its matrices, and preserves physical load as resultant force and
moment. `AdaptiveGestaltState` starts elastic and can sleep only after a real
local stress solve plus its error bound stays below an explicitly supplied
stress threshold for an explicitly supplied dwell time. It wakes immediately
when a certified interface-wrench influence envelope crosses a larger wake
threshold. There are intentionally no default gameplay tolerances. Runtime
matrix/topology switching and momentum-conserving state transfer are not yet
wired into `LiveStructure`; the full-reference lane still solves all beam
modes.

```text
python -m pytest tests/test_component_mode_atlas.py \
  tests/test_station_reference_motion.py -q
63 passed, 1 deprecation warning in 111.58s

python -m py_compile component_mode_atlas.py graph_columns.py frame_solver.py \
  live_scene.py tests/test_component_mode_atlas.py \
  tests/test_station_reference_motion.py
PASS
```

## 2026-09-14 universal topology-derived partitions

The ballast assembly was only a real-graph validation candidate; there is no
tank-specific adaptive code. `discover_component_partitions` now derives
component ownership from an arbitrary graph field (`assembly` by default),
maps condensed endpoints to their gestalt masters, and identifies interfaces
from shared topology and external fixity. Completely free components retain a
six-DOF reference node so rigid-body motion cannot be mislabeled as an internal
mechanism. Node body mass follows its declared assembly owner; distributed
member mass follows the member.

Interface discovery also includes all non-routed physical force edges, even
when they are deliberately absent from beam stiffness. Thus a damper,
actuator, contact, or currently disengaged lock crossing a component boundary
remains able to transfer a wake-triggering wrench. A generic test uses arbitrary
assembly names plus a non-beam hydraulic connection; no part-role or identity
special cases participate.

`build_component_stress_envelope` now generates the wake metric generically.
It solves unit interface force/moment coordinates with inertia relief: the six
rigid resultants accelerate the gestalt, while only self-equilibrating load
produces local beam deformation. A KKT constraint removes arbitrary rigid pose
without pinning a real component node, and an additional internal mechanism is
rejected rather than numerically regularized. Member coefficients conservatively
bound normal and von-Mises shear contributions and are applied to the absolute
interface wrench, so cancellation can overpredict a wake but cannot hide one.

The unchanged generator also succeeded on the real
`central-storage-ballast` partition, producing a finite 18-member by
24-interface-coordinate envelope. Its largest coefficient was
`288751.6493625501 Pa` per corresponding unit force/moment coordinate.

`GlobalGestaltSubspace` now performs the corresponding global scatter/gather.
It replaces a dormant component's physical free rows with admissible rigid
columns only after the complete station K/M have been assembled, leaving all
neighboring beam contributions attached to the same moving interface points.
Fixed-coordinate rigid motions are removed by nullspace projection, and
overlapping dormant components are rejected until they are merged. The
projected system is diagonalized into a mass-orthonormal basis for the same
average-acceleration Newmark lane; no second time integrator was introduced.

`LiveStructure.configure_adaptive_partition` and
`update_gestalt_residency` connect one generic certified partition end to end.
The active representation supplies solved member stress and internally
measured non-rigid kinetic energy. The dormant representation estimates its
boundary wrench from component inertial/elastic effort plus absolute applied
loads, deliberately using triangle-inequality overprediction where shared-load
ownership is unknown. A transition swaps global bases by mass projection;
neighboring members stay attached. A synthetic graph now sleeps an arbitrary
assembly, applies a force through its graph node, and wakes it back into the
complete elastic basis. The configured component and current representation
are also injected into node/document solver telemetry.

## 2026-09-14 graph-native residency orchestration

Residency orchestration is now itself graph data. A
`ComponentOrchestrationGraph` stores two sparse CSR adjacencies over generic
component vertices. Weld adjacency means selected sleeping vertices must be
unioned into one gestalt; constitutive adjacency means the vertices remain
separate rigid bodies joined by the existing actuator/damper/contact law.
Standard connected-components traversal returns weld clusters, and
`merge_welded_partition_cluster` unions their members/mass ownership while
recomputing external interface nodes. No equipment identity participates.

The complete station currently yields 28 structural component vertices,
56 weld adjacencies and 21 non-beam constitutive adjacencies. This is now the
data structure from which multi-region sleep scheduling and wake propagation
can operate, rather than another set of scene-specific conditionals.

The live scheduler now accepts multiple component vertices. A deterministic
maximal independent set over weld adjacency prevents conflicting rigid maps;
constitutively adjacent bodies may sleep together because their joint edge
remains active. Global projections are cached by the dormant vertex set, so
waking one vertex retains the others and reuses any previously encountered
basis. A two-component witness sleeps both non-welded bodies and then wakes
only the loaded one.

`station_reference.py --live` now enables a printed adaptive live profile by
default; `--full-beam-live` is the explicit exact-runtime opt-out. Candidate
selection is generic: largest coordinate saving, at most 144 physical DOFs per
component, at most four components, and weld-independent. The accuracy budget
is printed: sleep below `2e-4` yield utilization, wake at `1e-3`, internal
velocity below `0.20 mm/s`, held for `0.05 s`.

A real headless startup validation completed the unchanged 3,534-mode full
reference solve and baked four selected vertices into a 3,174-coordinate live
subspace (360 coordinates removed, 10.2%). The projected eigensolve took over
a minute and therefore belongs in a persistent/offline bake eventually; the
dummy SDL driver then failed at the expected OpenGL-window creation boundary.
The structural/adaptive bake itself completed successfully.
