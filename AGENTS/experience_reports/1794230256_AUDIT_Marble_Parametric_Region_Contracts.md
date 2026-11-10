# Marble parametric-region integration boundary

Date: 2026-11-09

## Follow-up: exact spherical source integration completed

The user subsequently requested prominent triangle-budget and sphere-coverage
controls with a truly spherical optical surface. This was implemented using
the original `SphericalSurface.glsl_intercept_fn` in both current Studio GPU
directions, with authenticated saved region membership and ordinary interior
triangle traversal retained. This does not claim native triangle-group or
spline support. Full current contract and receipts are at the Spectral handoff
linked below; the original audit sections here record the earlier state.

CPU-only full-scene admission runs before any active job is stopped and has
an 8-second timeout (measured about 2.6 seconds). Triangle budgets reject
rather than decimate. New exact saves require an uncoated uniform nonemissive
spherical shell; painted bands, dice, and missing metadata fail clearly.
Existing mesh saves remain supported and mechanical mass is unchanged.

Actual forward/sensor GPU probes were bit-identical at radii 1 m and 0.01 m:
both roots, full silhouette, tangent/miss, analytic normals, nested bubble
interfaces, TIR, inside origins, and nearer ordinary triangles. Actual RGB
with Auto, monochrome Exact, and RGB mesh photos completed. The nine-sphere
fixture changes 53,010 triangles to 24,466 triangles plus 9 exact spheres.
Original bubble GPU interface/BVH suites passed again. No service restart,
deployment, build, or user exposure interruption was performed.

Additional user wording:
> Give us on the page prominently in the exposure controls a triangle budget and sphere coverage policy as well as using the original spectral analyzer region and parameterized kernels, they are fast enough to give us perfect spherical surface because this thing is like fractal golf ball right now

## Finding

Current Spectral source has three distinct mechanisms: triangle-group
parametric refinement, exact sequential compound-lens transport, and
scale-context region dispatch. Studio photos use the two independent GLSL
field/sensor programs; the saved photo NPZ has only mesh/material arrays.
They cannot consume the native triangle-group payload yet.

The original SDF_SPHERE name does not mean a continuous global sphere: it
displaces a triangle hit in local barycentric coordinates. The CPU also
averages a neighborhood that the GPU payload/refiner currently omits.
ScaleContext kinds are not present in the current photo buffer, despite
an older header comment claiming parity; native SPLINE_SURFACE merely
flags entry. Exact lens teleport cannot stand in for a marble boundary,
because it would skip colored geometry inside the sphere.

## Deliverable and verification

Detailed source map, ordered design, and acceptance criteria are in
`C:/dev/Powershell/spectral-analyzer/MARBLE_PARAMETRIC_REGIONS_HANDOFF_2026-11-09.md`.
The 2026-11-07 handoff links to this correction. Existing CPU/source-contract
tests passed: `tests/test_saved_studio_gpu_scene.py` and
`tests/test_exact_camera_transport_contract.py`, 8 passed in 0.55s.

No runtime code changed, no GPU test or build ran, and no exposure or
service was interrupted. General support remains incomplete due to the
missing authenticated region/primitive contract in the photo pipeline;
there is no access denial or pending permission.

## Next steps

Preserve authenticated region identity through photo serialization and
triangle filtering. Reuse the intended original evaluator under its actual
semantics. For a continuous marble, integrate an exact per-interface hit
into both shader directions/visibility, retaining the colored interior
triangles and existing material/film pipeline. Validate before exposing
the requested triangulated/continuous choice.

## Prompt History

User:
> add the second to fourth wall to the studio, inscrease studio scale (keeping marble small), and work on getting support in the ray tracer for the ray tracer's parametric regions, it's in the original material in spectral analyzer
