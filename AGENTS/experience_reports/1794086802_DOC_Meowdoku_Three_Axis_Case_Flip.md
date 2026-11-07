# Meowdoku three-axis case flip handoff

**Date:** 2026-11-07
**Title:** X/Z physical turns through the 3-D law and loading Studio

## Scope

Isolated Turing worktree `C:\dev\Powershell\turing-studio-particles`, branch `codex/meowdoku-studio-particles`, commit `42374bca` after merging current `main` at `b26bb0d5`. Canonical Turing `main` and the live server were not edited. This continues the world-Y finite case motion in `33af3c2b`.

## Findings and work

- The 2-D fountain has a scalar case-normal turn. The 3-D law had the same scalar turn but lacked x/y angular vector inputs. Its law-to-world basis is `(law x, law y, law z) -> (world X, world -Z, world Y)`.
- Added runtime x/y angular velocity and acceleration inputs to the 3-D law. It uses vector centrifugal and Euler forces, exact Rodrigues Coriolis rotation, and angular acceleration on all three particle-spin components. The zero-x/y path retains the earlier z law. Vector rates obey the existing spring contact cap using the full slab reach.
- Loading Studio offers world Y with the 2-D product, and world X/Z with the actual 3-D SDF product. It enables a size only when the corresponding manifest contains that product and the 3-D axis inputs. The one finite simulation-step trajectory drives the law inputs, rotated gravity, case model, and recorded angle. Existing case-rendering paths draw the flat and 3-D cases through that model.
- The runtime refuses X/Z flips when the 3-D vector inputs are absent or a concurrent world-Y spin/tilt would make one-axis rendering false. The loading recorder clears spin and tilt first.

## Verification and limits

Focused WebGPU builds of the SDF 3-D `small`, `4k`, and `256k` products completed in ignored scratch output. The new small/256k generated products each have 51 slots; no Wasm was built. `python -m pytest tests/test_meowdoku_loading_library.py -q` passed 15 tests, `node tests/meowdoku_case_flip_check.mjs` passed including inverse scene/gravity alignment for all three axes, and changed JavaScript passed syntax checks. In a scratch server on port 8768, Chrome completed real world-X and world-Z small hourglass recordings: four turns each, respectively 196 and 183 frames, 256 grains, state-memory capture, with effective turn time 0.45 simulated seconds versus 0.40 requested. The 3-D case rendered in the recording preview. Scratch output and its library are isolated from the live site.

The 256k product compiled but has not yet completed an end-to-end browser recording or visual-only capture. Live serving requires integrating the commit and rebuilding/publishing 3-D WebGPU manifests and shaders. Keep Wasm generation lazy, on an actual fallback or server idle, per the user instruction. The 2-D product remains world-Y-only.

## Next Steps

Rebuild the deployed 3-D WebGPU products for the served site, verify a 256k X/Z Studio run and visual-only projection on the target GPU, then deploy when approved. Test the 2-D flat case visually as well.

## Later integration receipt

Latest Turing `main` at `83090fe5` merged into the isolated branch as `35e606c3`. The only conflict was `web_export.py`'s asset list; the merge includes both current marble modules and `case_motion.js`. Root subsequently merged this complete branch into canonical Turing `main`, preserving unrelated user work. On the merged branch, the focused loading-library suite passed 15 tests, the case and marble-preview Node checks passed, and fresh SDF 3-D WebGPU `small` and `256k` builds passed (46 and 191 dispatches, no Wasm). The 256k browser run remains open: a separate process was using scratch port 8768 and served 404s for the manifests; no other process was stopped. Do not copy the two-size scratch manifest over the canonical eight-size manifest. Rebuild all deployed 3-D sizes with `wasm=False` or explicitly gate per-size shader support before exposing X/Z controls.

## Prompt History

> "Continue physical Flip task from commit 33af3c2b. User explicitly wants a picked axis and actual containing-case motion shared with loading Studio. Review against latest main; run focused Python and browser/visual checks in isolated worktree. Determine whether world-Y-only can satisfy user; if X/Z require law extension, investigate and implement the real law path rather than a visual stand-in, but do not run unapproved Wasm. Provide concrete ready-to-land commit or exact blocker."

> "User approved WebGPU builds and explicitly requires Wasm lazy (on actual fallback or server idle). Please implement true X/Z 3D case-axis physics in source and manifest gating; rebuild only the focused WebGPU 3D product(s) needed for verification in your scratch worktree, not all sizes, and do not build Wasm. Leave 2D product Y-only. Coordinate branch hashes so root can integrate later."
