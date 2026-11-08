# Marble Madness contact and optical roughness audit

**Date:** 2026-11-07

## Scope

The existing 3D fountain law's contact coefficients, Marble Madness defaults, and the saved marble material and physical fields.

## Methodology

Read the law's contact and wall paths, the game controls, and the saved design mapping. Ran the uncompiled one-particle law for 20 steps with a ball touching the backplate, 0.3 lateral gravity, and initial lateral speed 0.02, at both μ=0.35 and μ=0.55.

## Detailed observations

- The law integrates angular velocity and a sphere moment of inertia `0.4 m r²`. Contact-point slip creates tangential force and torque. `mu0` controls sliding friction; `roll0` controls rolling resistance. The SDF walls and backplate use the same values. They do not read Phong material records.
- Marble Madness inherits species-0 `mu0=0.35` and `roll0=0`. The saved design has separate physical `sim-mu` and `sim-roll` fields which the page passes to these law uniforms. PBR roughness controls optical highlights only.
- After 20 steps, `|vx-r wy|/|vx|` was 0.271 at μ=0.35 and 0.269 at μ=0.55. Increasing the coefficient barely affected this early transient; the law's per-step friction-share limit and velocity regularization matter here. This narrow one-particle box test does not measure the full SDF course over a playthrough.

## Analysis

The fixed visual orientation was a strong source of apparent sliding, but the law also allows measurable transient slip. Mapping optical roughness silently to friction would conflate different quantities and override the user's explicit saved physical fields.

## Changes and next steps

Marble Madness now exposes its existing sliding-friction and rolling-resistance runtime controls in a `Marble contact` group. I kept their defaults because the short test did not justify changing them. Observe a full course run with mesh orientation connected before tuning defaults or changing the law's friction-share model. Study contact-energy or slip history before using force to scuff the visual marble.

## Prompt History

> the particle sim needs rotation state for marble madness to work, it has friction qualities, it even has roughness I think which we can use to scuff marbles from forces, very cool, but, we need that frictional rolling that I think the 3d sim actually has we just need to link to the mesh representation of the marble so it's not just slip sliding, right?

> also maybe we need materials to use their roughness maybe for friction and have sane baselines? maybe the marble is just very slippery in the test room you know? we have to figure it out
