# Meowdoku fountain auto Flip and oil toy floor

**Date:** 2026-11-07  
**Turing commit:** `e0f1bcd3` on isolated branch `codex/fountain-flip-oil-20261107` (base `9a9ec79e`)

## Findings and changes

- Fountain Flip already used `ComputeRuntime.startCaseFlip`, a bounded finite trajectory shared by the physical law and case renderer. The page's button now calls one `flipCase` function, also used by the Space shortcut and automatic controller. Repeated input cannot queue a second Flip while the first is unfinished.
- Automatic Flip is an opt-in control for the 2-D and 3-D fountain continuous runs. It reads the existing law diagnostics through `SettleDetector`, so the energy, speed, parked-id, relative-peak, and hold criteria remain the canonical settle rule. It waits until the previous motion and two simulated seconds have passed, and resets on compute-runtime token changes. It does not act during stop-at-rest replay or camera-down gravity.
- The oil toy's actual SDF chamber mesh already contained the full y=0 floor: at 4k, two long triangles span approximately x=±0.93 and z=±0.09. The page rendered these triangles with 30% opaque clear housing glass. The existing floor triangles now receive a visible Phong material. The collision SDF and compiled products were not changed.
- Only `examples/meowdoku/web/app.js` changed. Neither WebGPU nor Wasm needs recompilation; the page asset needs to be served from the integrated commit.

## Verification

`node --check examples/meowdoku/web/app.js` and `git diff --check` passed. A read-only `chamber_mesh('sdf_toy', 4096)` probe confirmed the floor coordinates in the decoded mesh. Interactive browser verification remains for the integrating agent.

## Prompt History

> "on fountain, the flip case needs a key binding and an auto on low energy option, there's no visible bottom piece to the oil toy but it's working perfectly. and we need a faster refresh rate or greater rotational accuracy on the wheel, it is in this very nice sim but it works kind herky jerky, is there some static friction braking or what is that"

> "Implement fountain case Flip key binding and automatic Flip on low energy, and fix missing visible bottom piece of oil toy while preserving working physics. Read AGENTS/experience reports. Use isolated Turing worktree/branch from latest main (9a9ec79e), preserve dirty WIP. Inspect existing case flip mechanism / dt/energy signal / UI controls and toy geometry; no alternate simulation. Design auto trigger using existing measured kinetic/settled signal, ensure sane cooldown/disable, faithful shared case motion. Fix visual bottom piece by mapping existing physical geometry/mesh, not decorative stand-in. Focused tests/browser if feasible, no Wasm. Send integration-ready commit and source/build digest impact. Coordinate files with wheel agent/root."
