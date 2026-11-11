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
