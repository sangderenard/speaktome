# Meowdoku physical case flip handoff

**Date:** 2026-11-07
**Title:** One finite case trajectory for law, scene, and loading recording

## Scope

Independent Turing checkout `C:\dev\Powershell\turing-studio-particles`, branch `codex/meowdoku-studio-particles`, commit `33af3c2b`. The branch was brought forward from canonical main before implementation. Main was not edited by this worker.

## Findings

Studio previously called `compute.setCaseTilt` instantly. That only rotated gravity in the case frame; the law's `case_omega` and `case_domega` remained zero. The saved MLA1 animation eased a view angle after the fact. The scene already had a `caseModel` that turns floor, plate, housing, ribbons, and grains, but Studio did not enable it, and the flat outline draw path did not use it.

The 2-D fountain and 3-D slab laws both represent case motion about the normal to their x-y law plane, which corresponds to world Y in the scene. Their case-frame force terms and per-grain spring-contact caps already exist. They do not support arbitrary X/Z rotation axes.

## Work

- Added a quintic finite turn in simulation steps with zero angular speed and acceleration at both ends. The same angle drives gravity and rendering; its derivatives feed the law's existing `case_omega` and `case_domega` inputs.
- Before a turn, the runtime reads the smallest live packed radius and current law uniforms to choose a duration below the spring-contact cap; it reports the effective duration. Studio exposes forward/reverse around the case normal.
- Studio uses the finite turn, records its actual per-frame angle in MLA1, and forces the case-visible view in its iframe. The flat outline uses the case model. The main page offers a Flip motion choice and a Flip case button.
- The export list includes the new pure `case_motion.js` module.

## Verification and limits

`node tests/meowdoku_case_flip_check.mjs` passed trajectory continuity, bounded derivatives, case-model orientation, and saved-angle checks. The existing visual tape Node check, browser JavaScript syntax checks, `web_export.py` syntax, and `git diff --check` passed. No Wasm compile, page export, browser GPU run, or deployment occurred. A real browser visual check remains necessary, especially for the 3-D slab, flat outline, and large Studio run. The Python suite was not rerun in this worker; the earlier shared environment failed loading Torch's `torch_python.dll`.

## Next Steps

Review/integrate Turing commit `33af3c2b`, run the focused Python suite in a working environment, then export to a test page and verify a physical 256k hourglass flip in a WebGPU browser before live deployment. X/Z axis motion would require new 3-D law inputs and force geometry, outside this change.

## Prompt History

> "User clarified Flip motion must be actual movement of the containing case, including flat cases in the 3D scene, not just gravity-vector rotation. Investigate existing case/body physics in canonical Turing and the loading Studio path."

> "Please implement your audited finite deterministic case flip mechanism in your isolated branch. First bring canonical main forward into branch (your prior studio commit is already in main). Add Flip motion option and axis selection where feasible, route loading Studio turnover through same ComputeRuntime trajectory, supply real case_omega/domega and rendering for both flat and 3D cases. Preserve existing laws and use existing case-frame force inputs. Test trajectory and visual alignment. Do not build Wasm or deploy."

> "Please prioritize a faithful physical trajectory and avoid huge unverified 3D-axis law edits. If X/Y/Z not supported by existing case frame, report exact limitation and leave branch at a coherent tested boundary."
