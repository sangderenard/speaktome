# Marble Madness physical rotation stream

## Prompt History

> the particle sim needs rotation state for marble madness to work, it has friction qualities, it even has roughness I think which we can use to scuff marbles from forces, very cool, but, we need that frictional rolling that I think the 3d sim actually has we just need to link to the mesh representation of the marble so it's not just slip sliding, right?

Parent assignment: trace the existing 3-D fountain angular state and connect its packed snapshots to the marble mesh without duplicating physics or triggering a Wasm build.

## Findings

`turing/examples/meowdoku/fountain3d.py` already carries `wx`, `wy`, `wz` state, computes tangential Coulomb friction at the contact point, a separate rolling resistance (`roll`, labeled roughness), and spherical moment `I = 0.4 m r²`. The five-word display packet omits angular state. Phong surface roughness is independent of the mechanical `roll` coefficient. The law does not carry cumulative contact wear or scuff damage.

## Work

`web/compute.js` now copies the first particle's three angular-velocity columns into a 12-byte side region of each existing GPU readback batch. The Wasm product reads the same columns from its state arrays. The runtime integrates a normalized quaternion at each simulation snapshot and keys it alongside the recording, including replay and branch truncation. `compute.rotation()` returns `{frame, quaternion, omega}` for the displayed snapshot. The side stream leaves the 20-byte general particle packet, recording codec, and compiled physics law unchanged. `web/marble_rotation.js` holds the pure quaternion integration and is included in static export.

## Verification

`node tests/meowdoku_marble_rotation_check.mjs` passed; `node --check examples/meowdoku/web/compute.js` passed; `git diff --check` passed. Browser WebGPU playback and mesh appearance are for the parent integration check.

## Next Steps

If the product needs force-driven scuffs, expose contact impulse or accumulated wear as an intentional law state and material input. Do not map optical roughness directly to damage.
