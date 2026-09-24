# Python woodshop Machine slice

**Date:** 2026-09-23
**Title:** Python two-hand woodshop with Machine objects and managed dt cutting

## Overview

Built the first playable woodshop loop entirely in Python. The Living Data Map
contributed interaction intent only; no JavaScript or compiler path was used.
The nominal pine stud and hand saw are existing `machines.Machine` objects,
their live instances are `MachineSim` objects, and all advancement crosses the
existing `DtCompatibleEngine.step_with_state` boundary with one shared
`StateTable`.

## Steps Taken

- Added generic `MachineWorkEdge` and `MachineAction` declarations to the
  established Machine graph export.
- Added a kiln-dried nominal 2x4x8 Machine and a hand-saw Machine containing
  one steel blade body, one wood handle body and one finite cutter edge.
- Added the Woodshop honorary family: L/R/T orthotropy, moisture strain,
  wood-fastener/bondline closures and executable saw/kerf work laws.
- Added a managed-dt wood-cutting engine which evaluates the honorary WO4
  pieces, retains kerfs, removes mass from the actual stock part and supports
  snapshot/restore.
- Added a Python pygame interaction layer with movement, pickup/drop, ten-slot
  hotbar, independent two-hand equipment, crosshair cut projection and held
  left/right actions.
- Added focused construction, geometry, dt, conservation and rollback tests.

## Observed Behaviour

- The requested focused and adjacent compatibility gate passes 40 tests.
- A broader 113-test machine/fabrication selection passed 112 tests. Its only
  failure is the existing turret-production string `section` reaching
  `ProductionGraph.edge` where a section object is required; none of the
  changed files participate in that failure.
- Off-hand support intentionally produces centimetre-scale line error and a
  much lower delivery efficiency. Supported stock projects the intended and
  actual lines together.
- Kerf depth, removed volume, removed mass and applied work persist in the dt
  state table. Rollback restores both kerf state and stock mass.

## Lessons Learned

The existing Machine boundary accepts tool semantics cleanly when a working
edge is treated as a finite contact locus carried by a real part. Inventory
does not need another object type: the two hands and hotbar retain the same
Machine identities and only change custody.

## Follow-up: SDF material topology

`engine_toy/material_topology.py` now supplies the missing boundary. Saw
strokes subtract capsule SDFs from a mesh-derived occupied material field;
occupied volume updates the source part mass and a real chip-form Machine
during the cut. Depth is not a split condition. Removal of the final material
bridge retires the predecessor and divided part, then mints one new Machine
for each separate contiguous island with lineage and a closed mass balance;
there is no relationship between the resulting objects.

## Follow-up: woodworking clamp interface

`engine_toy/woodworking_joints.py` adds F-style bar and pipe clamps as real
multi-part Machines. A `WoodworkingJoint` is a plan; auto-deployment creates
the load path as two temporary jaw-contact edges per clamp (member A -> clamp
-> member B), never a direct member-to-member edge. WO5 supplies screw force,
pad pressure, friction capacity and frame deflection. Release removes the jaw
edges.

## Follow-up: first-person shader view

The initial pygame adapter's orthographic plan view was removed. The Python
frontend now resolves the live Machine-part geometry state into triangles and
submits it to Pluck's existing `BaseGLRenderer` / `base_material` shader. Its
captured-mouse camera uses the Living Data perspective constants and centre
ray for targeting, with WASD movement in the camera plane. This is only a
view/input adapter; no portal, room, map, JavaScript, or parallel simulation
system was added.

## Follow-up: declared hand poses and oriented drop placement

`MachinePose` gives ordinary Machines declarative `selected`, `used-1`, and
`used-2` hand-relative poses with a generic selected fallback. World Machine
nodes now retain their interaction pose and XYZ orientation. Saw activation
extends the tool and alternates its two authored stroke poses. Drop is a
two-stage interaction: Q creates a live placement, mouse drags its rendered
three-axis gimbal, and Enter/Q commits the same orientation to the world node.
Oriented part corners drive rendering, centre-ray picking, XY bounds, and
floor support height; a 90-degree X rotation therefore leaves the stud resting
on its real 1.5-inch edge. Position/mass remain in the dt identity registry;
orientation and interaction pose use normal StateTable columns rather than
changing that registry's schema.

## Follow-up: existing Newton honorary gravity and contact

The woodshop world-motion rule now consumes the existing Newton catalogue;
it does not add another mechanics family and does not route through the 2-D
classic-mechanics demo engines. N4.1 evaluates Earth gravity, N1.2/N1.1
advance each released Machine's momentum and translation, and N5.1--N5.7
supply unilateral contact, restitution, normal impulse and Coulomb friction.
The oriented real part boxes only provide contact geometry. Held Machines are
excluded, while released Machines publish momentum and resolved contact
records through the same StateTable. The world rule implements snapshot and
restore over object positions, authored base positions, momentum and contact
state.

## Follow-up: friction-fit sawhorse jig

The initial world now includes two one-part resin sawhorse brackets. Their
three existing PartPorts are typed compression sleeves accepting the dressed
2x4 cross-section independent of member length. A held stud can primary-click
a bracket in the other hand or under the crosshair to create a real
world-object edge. Penetration and pullout capacity remain live state; WO5.3
derives holding force from sleeve compression and friction. The edge declares
rigid condensation until load projected along the release vector exceeds that
capacity, at which point it is removed without changing either object's
identity. Each port also exposes generic finite fastener areas and WO3-ready
glueable surfaces rather than prescribing screws.

## Follow-up: starting stock and persistent world checkpoints

The starting world now has four resin sawhorse-bracket Machines and eight
true dressed 2x4x8 Machines. One stud remains loose at the established work
position and seven form two adjacent physical stacks. The nominal length is
still exactly 2.4384 m; neither the first-person camera nor the renderer was
changed to make the stock appear larger.

`WoodshopSimulation.snapshot` now coordinates the existing dt
`StateTable.snapshot` with the actual world-object graph and each participating
engine's existing snapshot contract. Restore rebinds `MachineSystem`, cutting,
and Newton world-rule engines to the restored Machine objects and identity
assembly. It retains object geometry and orientation, custody and both hands,
momentum and damage, SDF kerfs and chips, topology/lineage, physical edges,
woodworking and compression-sleeve joints, drop state, and engine clocks.
`save_world`/`load_world` persist that checkpoint with an atomic file replace;
the pygame client exposes them as F5/F9. The focused woodshop suite passes 26
tests, including in-memory and disk round trips.

## Follow-up: honorary laws use real LLVM pieces

Removed the honorary catalogue's local lambdified `Piece` lookalike. Selected
honorary SymPy equalities now pass through `compile_sympy_equations` to a
`SymbolicEquationCompilation`, then through `piece_from_law` to the compiler's
real batch-one `LLVMPiece`. The woodshop requests only the laws it owns: five
WO4 cutting laws, four WO5 joint laws, and six Newton laws. Generated native
artifacts are cached under the ignored `artifacts/llvm_pieces` tree. The
canonical compiler exposed invalid N5 symbol identifiers `v_n^-`/`v_n^+`;
these are now `v_n_minus`/`v_n_plus` without changing the equations. The
26-test woodshop suite passes while executing the native pieces. These pieces
are still called by leaf `DtCompatibleEngine` methods; this change does not
yet place the Newton engine or the other engines inside an LLVM dt system.

## Follow-up: batched Newton LLVM dt graph

Removed the scalar N4.1/N1.2/N1.1 calls from the per-Machine and per-axis
world loop. The Newton motion manifestation is now three real `LLVMPiece`
leaves at the current complete object batch: N4.1 gravity, N1.2 momentum, and
N1.1 position. They are children of the repository's actual sequential
`dt_graph.RoundNode` and run through `llvm_dt_system.dt_system_from_graph`.
The dependency is explicit and same-step: momentum consumes the gravity
column written before it, and position consumes the momentum columns written
before it. Held objects remain ordinary batch lanes with their active mask
off; a topology-size change rebuilds the fixed-shape piece set for the new
batch.

`equation_piece` is the multi-equation counterpart of `law_piece`; it only
passes an already-discretized SymPy equation set through
`compile_sympy_equations` and `piece_from_law`, with the same content-addressed
LLVM-piece cache. The Newton manifestation composes its right-hand sides by
substitution from the catalogue's N4.1, N1.2, and N1.1 expressions and applies
the existing symplectic-Euler order. It does not introduce another evaluator.

The managed graph was observed calling its three participants exactly once
for the 17-object starting batch in a one-step round. The complete focused
suite is 30 passing tests. Whole-program lowering was attempted through the
unchanged file-based `llvm_dt_system.lowered_system` entry. Both direct and
file-based attempts reached ProcessGraph source-closure construction, then
recursed while resolving `ExtractionContract.program_abi/path` in
`graph_express2._resolve_ast_parent_reference` until `RecursionError`, before
C emission or linking. This is the current whole-dt compiler frontier; it was
not replaced with another runner or a fallback artifact.

## Prompt History

> okay, let's copy the living data design inspiration, basically, ripping off minecraft with a hot bar for equipping and using, the user inputs, inventory - take the intentions in the live data web map system, and take the machine system and make sure our tools are machines that have held click actions.
>
> Lets try to make a kiln dried 2x4 on a default z plane, a piece of wood, a hand saw - projection of a cut line into a mesh for the lines that will be cut
>
> we're trying to make it in python not trying to compile anything yet, we want to use just machine sim objects as the presently exist and are parameterized, a 2x4x8 nominal kiln pine stud, and a handsaw which is made of just a metal shape, a wood shape, and a work edge, the generic object we'll use to define where there is a kerf generating cutter from A to B thickness C wave (hacksaw wavey blades, you know) D tooth pattern enum E)
>
> user can move around, pick up and put down the 2x4
>
> and we'll need two hand equipping of the hotbar so we can hold a 2x4 and cut with the other hand, at ENORMOUS accuracy penalty through the cutting process
>
> keep it simple, use the real dt system, real honorary laws, we have to make real machines, etc etc

> the living data map is javascript and we are not making javascript right now

> just put in the fact that it handles a geometry state handed to the existing
> shader from player point of view, no portals, nothing crazy that had nothing
> the fuck to do with what I mentioned, the orthogrpahic view. just make it use
> the same basic camera and shader please

> now we're going to need 4 of those I think and then a pile of 2x4s, you can
> make them 8' I don't know if they're defaulting for longer or i just feel
> small in the game. don't change anything about view if they're already 8',
> uh, we need to also work on save and restore of the state of the world,
> allowing our play here to experience persistence

> yes you see the necessary change please make it, and then we'll step back
> to the tree of our woodshop and we'll see if we can swamp out, say, the
> newtonian engine to be an llvm dt system

> please make the change to the better scheduling and see if you can lower the entire dt systems
