# Studio fluid field and authored container audit

The exact Studio capture already feeds StudioMeshBake and the native local
feature contact environment. Fountain3D wheel contact uses equal/opposite
hinge reaction torque. These are the container/coupling starting points;
a second proxy container or visual liquid cannot replace them.

Found and repaired repeated-index accumulation in Turing Bath SPH sample_at.
Two supporting equal particles previously lost half the density and selected
one temperature. Nine focused tests now pass. Existing physical state and
integrator were not changed. Full Bowl/Studio coupling and humidity/dust
optical publication remain unimplemented; no production changes were made.
Details: turing/docs/hazards/20261111T152218Z_sph-field-publication.md.

## Prompt History

User: "can we work it out on humidity and dust and then just make it not matter if they aren't declared, but if they are it is read by the volumetric march"

User: "I keep thinking for liquids and gas we use an extremely sparse simulation doing that thing where you interpolate and use dirac or something, correct me please, but we should be able to enable the particle sim to project a continuous fluid field"

User: "the right place i think would be to overhaul the liquid appearance shader to be a liquid engine"

User: "putting a clear glass bowl on the product stage in which the marble sits, and the bowl is filled with liquid"

User: "on particle sim, look at the wheel, constrained rigid bodies, we bake the container from the mesh of the scene"

## Broader verification

Sixteen targeted tests pass; two Bath superstep-hook tests fail because legacy
M metrics lacks pub_exchange_time. Both reproduce in untouched c214b7fc at
C:/dev/sph-baseline. This managed-controller integration blocker is separate
from the repaired sampling publisher. No alternate timestep mechanism added.

## Native optical continuation

Implemented an explicit optional extinction grid in the existing spectral
forward tracer, sensor, local photo worker and related transport seams. The
grid contains actual nonnegative m^-1 values in world coordinates; omission
leaves the old shader source unchanged. SPH publication calls the existing
sample_at with an explicitly required mass extinction coefficient. It does
not add a second particle solver, inferred fog, emission or a free surface.

Cross-repository SPH publication matches its independent kernel sum and
preserves physical state. 114 targeted spectral CPU tests pass, one skips. A
real AMD Radeon(TM) Graphics GPU gate passed six shader variants, 544 analytic
segments, zero-field bit identity, actual camera/source attenuation and
directional integrated transmittance. The test waited until the tiled
exposure had completed and released the GPU before the field worker's tests.
Full bowl acceptance and live Studio field publication remain unfinished.
See spectral-analyzer/docs/20261111-volume-extinction-receipt.md and the API
contract in camera_designer/STUDIO_VOLUME_EXTINCTION.md in that repository.

Later user direction: "make the real one or make the demo the real one".
