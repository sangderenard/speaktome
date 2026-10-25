# TODO Stub

**Date:** 1785859136
**Title:** Turing WebGPU backend / bundle-publishing follow-ups

See `AGENTS/experience_reports/1785859136_DOC_Turing_WebGPU_Backend_And_Bundle_Publishing.md`
for the full report. Short version:

- [ ] Backfill `origin.json` for the 35 pre-existing programs under the
      coordination root's `site/programs/` by tracing each to its real
      source file in `turing/` (many are named demo scripts, e.g.
      `columnar_multifluid_web_demo.py`). Skip and report any that can't be
      confidently traced rather than guessing -- probes/feed values that
      were only ever supplied at CLI build time (not baked into a literal
      `TURING_PAGE` dict in the source) are not recoverable and must not be
      silently defaulted to empty.
- [ ] Live-browser verification of the new WebGPU JS runtime in
      `turing/src/compiler/wasm_html_shell.py` (`navigator.gpu` device
      request, compute pipeline, bind groups, the hand-authored WGSL
      presentation shader) -- only syntax-checked so far (Node `--check`),
      never run against a real GPU/browser.
- [ ] Wire the fluidic/columnar module
      (`turing/src/common/dt_system/fluid_mechanics/columnar_multifluid_web_demo.py`,
      currently WASM-only because WebGL couldn't do compute) onto the new
      real WebGPU compute backend (`ssa_webgpu_backend.py`).
- [ ] Build the actual region-to-target dispatch planner. What exists now
      (`region_target_capabilities` in the bundle manifest) is only
      *capability* data -- which registered `machine_targets` entries could
      serve each region -- not an assignment. The user's stated end goal is
      distributed processing: programs dispensed to clients, with a planner
      deciding per-subgraph target language after sorting loops out.
