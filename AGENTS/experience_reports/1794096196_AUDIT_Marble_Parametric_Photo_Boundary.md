# Marble parametric photo boundary

**Date:** 2026-11-07

## Finding

The deployed Spectral photo worker uses an analytic thin lens but triangulated saved marble geometry. Both the forward light field and backward `SensorAccumulator` shaders intersect the same triangle BVH; neither accepts the native C++ tracer's `TRI_PARAM_SURFACE_SDF_SPHERE` group contract. A Studio switch today would not cause continuous glass intersections.

The precise GPU and saved-object work to offer a truthful optional continuous sphere is recorded in `C:\dev\Powershell\spectral-analyzer\MARBLE_PARAMETRIC_PHOTO_HANDOFF_2026-11-07.md`. It requires analytic shell hits in both GPU passes, authenticated recipe-derived radius and material mapping, and measured refraction comparisons. Keep the existing triangle mode as a deliberate visible option. No renderer or UI toggle was changed in this audit; no build was run.

## Next Steps

Implement and validate an analytic uniform-material spherical shell in both GPU photo passes while preserving the saved triangle interiors, then extend the continuous material field to bands and enamel before exposing a general Studio switch. Restrict early parametric mode to designs actually supported by that contract.

## Prompt History

> "the ray tracer should also support a parametric surface mode btw, we should be able to toggle, because sometimes I like the triangulated glass artifacts but we technically have a continuous wya to do it"

> "Audit Spectral marble photo GPU path for genuine continuous/parametric outer sphere support, per latest user request to toggle triangulated glass vs continuous. Work in C:\dev\Powershell\spectral-analyzer and Turing only if needed. Respect AGENTS and existing mechanisms. If a narrow true implementation is feasible, implement and test including UI toggle coordinated with root (root editing marble_photo.js, marble_studio.html/js for camera selector now; do not edit those files without messaging me). If not, write precise handoff and report, no fake toggle or tessellation masquerade. No Wasm builds, no long compiler builds. Inform me quickly."
