# Mesh field-domain substrate audit

## Scope

Compared the existing engine-toy machine/engine systems, the compiled vehicle
validator, the raincloud atmosphere/view split, and Spectral Analyzer's base
material shader to identify the shared contract for a general mesh-law engine.

## Findings

- `MachineSim` keeps the production graph authoritative and exposes the same
  damage/state shape as the engine simulation.
- `CycleEngine` checkpoints one exact contiguous engine state span for managed
  dt rollback.
- The vehicle validator keeps immutable topology/geometry feeds separate from
  persistent per-edge/per-vertex material state and compiles the complete dt
  window.
- The atmosphere owns voxel physics; rendering consumes a derived 3-D radiance
  field through the shared base-material shader.
- `cavities.py`, `chambers.py`, and `applicators.py` are useful analytic/reduced
  audit lanes. They are not the missing arbitrary-mesh field authority.
- `FARADAY_CHAMBER_CONCEPTION.md` already names the intended missing layer as
  `field_domains.py`, using DEC forms, real mesh occupancy, managed dt, and
  derived graphics.

## Repairs made in turing

- `HodgeStarBuilder.build_full_hodge_star` now fan-triangulates complete
  oriented polygon rings. Previously it used only the first three vertices, so
  a quadrilateral face had half its true area.
- Vertex and edge shares now divide by the actual polygon vertex count.
- The Hodge cache identity now includes embedded vertex coordinates. Previously
  equal connectivity and array shape could alias physically different meshes.
- `AbstractTensor.cross` now accepts bare `(3,)` vectors and preserves the
  public result shape; the private reshape workaround in `laplace_nd` was
  removed.

Focused verification: `tests/test_laplace_normals.py` and
`tests/test_dec_d_operators.py`: 44 passed.

## Next steps

Build the existing design's general mesh field-domain layer: immutable mesh
and DEC topology, real part occupancy/material maps, law-owned volume state,
exact state-span rollback, and dt publications. Keep analytic cavity models as
verification/reduction lanes and make graphics consume derived field buffers.

## Prompt History

> "exactly, we keep what we have that's strong, we form anew what will be so much stronger done one more time, we don't make bespoke products we make a system that prints game engines like they are gumballs in a dispenser"

> "look at machine sim and engine engine and the validator rigid body style, look at the spectral analyzer base material shader, the atmosphere example and shader"

> "it has to do with using real meshes and real laws correctly"

