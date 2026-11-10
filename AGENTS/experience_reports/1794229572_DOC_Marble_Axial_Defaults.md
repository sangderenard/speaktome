# Marble Studio centered axial defaults

Implemented in the existing local Turing checkout at base f998b80f, 2026-11-09.
Existing unrelated changes were retained. Nothing was exported, deployed, or restarted.

New Studio designs use two identical co-located ribbons around one straight
longitudinal axis, separated by 90 degrees. Initial phase is 45 degrees and
longitudinal twist is 0.9 radians per normalized axial coordinate. The layout
supports 2..7 planes evenly spaced by pi/count, because a ribbon is two-sided.
Changing the seed changes their common flare, never their relative placement.

The clear shell uses nominal IOR 1.52; all three colored-glass catalog materials
use 1.54. Both have transmission 1 and opacity 0. These are plausible artistic
compositions, not measured material data or wavelength-dependent dispersion.
Shader opacity is a forced reflection mixture, not dye density; zero opacity
retains Fresnel reflection. Existing albedo/inner-color values are retained.
New default accent is green dyed glass rather than the artistic milk preset.
Optional milk glass retains its existing authored values (IOR 1.48,
transmission .42, opacity .79), not a claim of physical bulk scattering.

The saved recipe has an explicit ribbon-layout field. Missing fields select
legacy separated geometry; read/apply/load roundtrips preserve old triangles.
The low-level mesh's omitted layout intentionally stays legacy for other game
callers. Stored material tables are not migrated or rewritten. New overlapping
axial volumes are marked mechanicsResolved=false, so estimated mass cannot
silently enable exact-inertia rolling. Exact unions/nested-media transport are
not implemented by this defaults change; the current ray tracer still treats
interfaces as air/material and does not correctly resolve intersecting dyes.

Verification:
- New axial check passed for counts 2..7, common centerline, identical rotated
  cross sections, valid normals/winding, interior bounds, and honest mechanics.
- Pre-change legacy geometry digest remains
  b6d1b1ca49dce15364944415f134184f4ab95224de8103e570d5b820c20dad9a.
- New and legacy saved-object geometry/mass verification and UI load/read
  roundtrips passed in tests/meowdoku_marble_axial_defaults_check.mjs.
- Existing mesh, controls, rotation and preview-motion Node checks passed.
- Material catalog tests: 2 passed. Immutable store roundtrip: 1 passed.
- JS syntax, Python compile, scoped git diff --check passed.
- Existing meowdoku_marble_object_check.mjs fails at its distant-proxy count
  assertion both with these changes and isolated clean HEAD modules. The
  separate new saved-object roundtrip coverage above passes.
- The HTTP library route test's incidental api.ipify.org public-IP lookup was
  denied by network policy; route test remains unverified, no network bypass.
- No browser/GPU photo, full suite, export or restart was run.

The raster preview still uses opacity as display alpha; ray-correct zero opacity suppresses its surface reflection. This defaults change does not claim matching preview optics. A shared shader refactor was deliberately not guessed.

## Prompt History

> can you change the material defaults for our marble glass and inner dye, give the inner pieces a different refractive index reflecting they are different materials and set the right transmission and opacity, and then lastly, the default cats eye geometry should layer the bands in exactly the same place, turning them axially to spread evenly axially distributed, that X shape twisted in the marble is how you get the visual interest catching light
