# nD Force Assembly and Spherical Observation

## Scope

Replaced the active screen-space graph mechanics with a generalized
high-dimensional force assembly and made the WebGL radar a projection-only
observation surface.

## Implementation

- Added `NDForceAssembly`, which stores node position and velocity in typed
  `[node, dimension]` arrays.
- Allocated two explicitly shared world dimensions plus four private
  dimensions per network. Network-local pipes, shell constraints, and
  collisions act only in their owning subspace. Cross-network pipes act only
  through the shared dimensions and cannot drag nodes through foreign axes.
- Added sparse pipe-force assembly, pressure-dependent rest length, fixed
  anchor constraints, semi-implicit damped integration, and a local spatial
  hash for collision candidates.
- Replaced SVG-ring habitat contact with nD network-local habitat-shell
  proximity. The client publishes `client_nd` observations and a force
  relaxation proof; projected screen coordinates never feed biological
  uptake.
- Added deterministic arbitrary-dimensional projection onto `S²`.
  Hyperspherical angles can be converted to Cartesian vectors in `O(D)`.
  Each source dimension owns a stable hashed projection column, so adding a
  network does not rotate existing dimensions.
- Kept a reversible `window.fluxObservationFrame` containing source vectors,
  dimension labels, the projection matrix, confidence, shell coordinates,
  and screen coordinates. `window.fluxNDPhysicsFrame` exposes the underlying
  mechanical state and relaxation proof.
- Once nD mechanical coordinates exist, physiology affects visual materials
  but is not silently mixed into position.

## Validation

- Strict TypeScript type checking passed.
- Vite production build passed.
- The executable projection/force suite passed angle conversion,
  deterministic projection, projection-column stability, unit-shell,
  fixed-anchor, finite-force, network-dimension, and habitat-proximity checks.
- 220 directly callable graph/radar/persistence regressions passed; 23
  fixture-dependent tests were skipped.
- Python compilation, classic JavaScript parsing, guestbook validation,
  `git diff --check`, and `npm audit` passed.
- Headless Chrome confirmed WebGL2, a 10-dimensional force solve, normalized
  shell coordinates, finite projected positions, 6 draw calls, 388 triangles,
  and no page errors.

## Design Notes

The shell is a lens, not another physical world. Biological and pipe
mechanics must have one source of truth in nD. Projection confidence records
how strongly a direction is represented in the current 3D observation basis;
it does not become a force.

## Next Steps

- Move the typed-array assembly to WebGPU compute while preserving the same
  equations and proof schema.
- Delete the now-dormant ring/wedge helper functions after the remaining SVG
  guides are migrated or removed.
- Join the fluid-work acknowledgement and nD force proof into one client
  relaxation barrier so a live tick advances only after both domains finish.

## Prompt History

> "can we collapse an n-d spherical system into a 3d shell radar for visualization, and pack the other networks into different dimensions. that is possible right? the only trick would be an efficient algorithm for taking arbitrary dimension reduction of spherical coordinates"

> "if you could see us t hrough this that would be ideal"

> "Proceed after marking your progress in the repo and pushing"

> "I would prefer we not to keep 3d physics unless it were somehow necessary, and then I'd want us to manage physics local to networks' dimensions, the physicology, the pipe physics of force, a force assembly solving - we can do that better in the n-d with strict ordinary generalized rules and math and not try to maintain an older system"
