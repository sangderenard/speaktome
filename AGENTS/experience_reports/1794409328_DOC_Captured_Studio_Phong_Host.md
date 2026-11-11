# Captured Studio Phong host

Date: 2026-11-11 UTC

Implemented in the canonical Turing repository: a bounded optional Python host
for the existing full-scene standalone Studio renderer. The explicit internal
product operation is `render-captured`, returns PNG, and rejects settings/path/
shader overrides. It reuses captured GLSL unchanged in software WebGL, without
adding a renderer or borrowing the ray worker's GPU. Actual two-camera validation
passed and the first PNG matches the earlier standalone runner's exact hash.

Details and limits: `turing/docs/STUDIO_PHONG_HOST.md`. No production deployment.
Arbitrary settings-to-pixels and active field/DOF replay remain unimplemented;
this captured-scene path must not be represented as those products.

## Prompt History

User: “Excellent let's work on allowing the site to record the phong output so users can take pictures and video of the phong version without ray tracing and figure out a way our internal server can run those same shaders itself somehow so the api shortcut is able to self host the phong render. Could be as simple as keeping glsl copies”

Parent instruction: “Proceed reusable full-scene host only if API shortcut can consume it meaningfully and explicitly reports limits. Do not change current operation='scene' meaning or substitute captured baseline for requested settings.”
