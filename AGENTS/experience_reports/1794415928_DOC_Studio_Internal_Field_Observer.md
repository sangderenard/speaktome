# Studio internal field observer

Added spectral-analyzer/camera_designer/studio_directional_media.py as an optional
composition helper. It observes entry packets on actual refracted segments and
integrates canonical Beer attenuation over each recording cell. Primary Beer
updates and scalar air-only guards remain unchanged. Region dye coefficient
selection is derived from the installed canonical helper. Invalid ownership is
excluded; simultaneous ExtinctionGrid is explicitly rejected pending joint
integration. No atmospheric rays, scattering, or gain were added.

Measured: ten focused CPU/GLSL tests pass. A bounded real canonical-tracer slab
probe passes on AMD Radeon(TM) Graphics: clear and strongly absorbing internal
paths populate the atlas, scalar output stays bit-identical, and the dominant
direction agrees with Snell within documented integer quantization. The test's
initial all-path-average comparison was invalid because actual internal
reflections contribute; it was corrected to inspect a directional bin.

The helper and proof did not edit demo_pluck_gl.py, studio_directional_field.py,
served assets, or the production service. Public activation belongs to the
coordinated job/schema integration. Full receipt and limits are in
spectral-analyzer/docs/STUDIO_INTERNAL_FIELD_OBSERVER.md.

## Prompt History

User: “Is that inner shadow caused by the engine used for glass transfer? We need the volume March to obtain complete internal volume field”
