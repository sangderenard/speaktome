# Canonical Studio secondary material field

Implemented conditional secondary shader in spectral-analyzer/csrc/shaders/base_material.frag.glsl.
It reuses the existing material SSBO equations and Turing GLSL project-IR export.
Passes 0/1 add linear body-scattering/reflectance with primary opaque depth EQUAL;
pass 2 renders sorted transparent layers with primary and field response together;
pass 3 resolves canonical ACES/sRGB once. Primary uLinearOutput=false preserves
legacy no-field display. Six directional atlas bins are explicitly an approximation;
material body scattering is not free-space ray marching and glass is not traced refraction.

Validation: six focused pytest cases passed, including 30 actual RTX3060 pixel checks,
legacy HEAD bitwise parity, material mask AND primary depth, directional reversal/rotation,
all six slabs, opacity-zero glass, sorted two-layer attenuation, thickness scattering,
and canonical resolve. Exported ES shaders linked with and without shadows through
installed official glslangValidator. No production deployment or restart. Integration
and browser work remains with the parent task; this receipt is not an end-to-end claim.

Contract: spectral-analyzer/docs/STUDIO_SECONDARY_FIELD_SHADER.md.

## Prompt History

User: “Make a secondary shader and ensure it can run after the primary in immediate succession, it will do three passes, volumetric, all reflectance, glass the material has it so the light field is used to derive the scene painted onto that material, and maybe a phong specular override”

User: “The light field can discriminate in recording all the details of material area from metadata in the delivered product so the where to do what work can be anded with visible triangles”

User: “Yeah make the light field a server side vector bundle with metadata”

## Category continuation

Added exact Subject/Stage/Room receiver opt-ins to the existing host. Shared
STUDIO_SCENE_PART_GROUP drives both native photo contract grouping and raster
lineage. Mixed table/gobo and stand/backdrop-holder meshes retain face categories;
transparent categories are contiguous runs inside the existing global index sort,
not independently sorted lists. Shader uFieldCategoryDisabled has a safe zero default;
disabled categories retain ordinary primary response and transparent layers remain.
Unknown particle/cage ownership falls back to ordinary shading. All-off bypasses
both scalar and secondary fields. Receiver category edits do not change scene revision.

Checks: 2 Turing pytest wrappers, scene material-engine Node gate, unchanged four
scene-builder fixture hashes, secondary orchestration, and 6 Spectral tests with
128 actual GPU assertions (AMD Radeon Graphics reported for the final run). No
fixture export refresh, production change, or deployment performed in this continuation.
The concurrent 32/64 comparison used a frozen pre-category source snapshot.

User: “Let's have a setting for division scale and make it opt in by category, subject, stage, room”

## 2026-11-11: native camera continuation observation

Implemented explicit borrowed-buffer observation in `camera_designer/studio_camera_intervals.py` and the existing SensorAccumulator opt-in source/binding seam. No renderer, GPU context, exposure or screenshot was invoked. Five CPU tests passed in `tests/test_studio_camera_intervals.py`. ABI and remaining limits are in `docs/NATIVE_CAMERA_INTERVAL_OBSERVER.md`.

The observer records original boundary intervals and full region state without extra RNG or optical-law changes. It preserves the native shared-band0 refraction limitation and rejects cooperative replay rather than duplicate records silently. This does not implement interior field screen evaluation. User requirements guiding this work: “also you need to alpha to the phong in sparsity of data. an empty region is not black, it's no information” and “include the rays calculated inside solids”. No confidence or completeness claim follows from nonzero energy or visit counts.

Additional source-only work: explicit per-band continuation correction is independently opt-in and is NOT enabled by response-copy information requests. Exact pre-integration RHS copies now extend the existing sensor evidence module with bounded per-dispatch storage; original integration and RNG calls remain. Scalar deposit-knownness and direct-NEE branch outcomes are distinct from expression-presence. Fifteen CPU/offline compiler tests pass, including native plain/region and opaque-PBR variants, plus a fake-GL lifecycle without a context. First-visible hit position/material/boundary metadata is retained at its original event. Final browser face/layer mapping remains unresolved for sphere compression, rod partitions and omitted bubble surfaces; no screen completion is claimed.


## 2026-11-11: scene-attached original response copies

Added `studio_completed_samples.py`, `studio_scene_responses.py`, and `studio_scene_application.py`. The original completed camera sample and mirrored native additions are copied before pixel integration. Component slots retain original film-filtered values; the installed evidence kernel and linear color profile are reused producer-side. Typed forward/reactive packets and native scalar deposit snapshots remain separate semantics, never summed as camera radiance. Scene retention has explicit bounds, physical-scene invalidation, eviction/no-information counts, and camera/population provenance. The completed mode uses one 512-byte row per original camera sample rather than bounce histories. Ordinary source and mode remain unchanged without opt-in.

48 CPU/offline compiler tests passed. No GPU context, exposure, rendering, screenshot, publication or deployment was performed. Runtime memory reservation, actual worker wiring and client query/composition review remain required. Application readiness remains false, and optional direct Phong specular override is not claimed supported by the new copied component product.
