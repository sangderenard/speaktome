# Documentation Report

**Date:** 1785859136
**Title:** Turing WebGPU compute backend, bundle-publishing CLI, and program-level origin/versioning

## Overview

A long session of work in `turing/` (and the coordination root at
`C:\dev\Powershell`), not `speaktome/` itself, but recorded here per
`AGENTS.md`'s guidance that this guestbook is the closest thing the
workspace has to shared memory across subfolders. Four connected threads:

1. Deprecating WebGL in favor of a real WebGPU compute backend.
2. Building an "easy entry" CLI to compile and publish program bundles.
3. Making published bundles self-describing (program-level origin records).
4. A compiler-version-rooted sequential versioning scheme for forced
   regeneration, alongside the existing content-addressed default.

## Steps Taken

**WebGPU backend** (`turing/src/compiler/`):
- Renamed `strategize_glsl_deployment` -> `strategize_shell_deployment`
  (`glsl_deployment_strategy.py`) -- the name implied a GLSL-only stage; it
  is actually the compilation choke point every backend passes through.
- `machine_targets.py`: added `deprecated`/`deprecated_reason`/`redirect_to`
  fields to `TargetCapabilities`; `get_target("webgl")` now redirects to
  `"webgpu"` unless called with `allow_deprecated=True`.
- New `ssa_webgpu_backend.py`: a real SSA-to-WGSL compute emitter with its
  own op-expression tables (not borrowed from GLSL, mirroring how
  `ssa_fortran_backend.py` owns its tables), real dispatch/launch-plan
  sizing (`WGSLLaunchPlan`/`plan_wgsl_launch`, mirroring desktop
  `glsl_backend.py`'s `GLLaunchPlan`), and structured (non-goto) control-flow
  lowering for the two recognizable SSA idioms -- `loop { ... continuing {
  ... } }` for the `preheader/header/body/latch/exit` loop shape, `if/else`
  for `CondBr` diamonds -- with an honest shortfall for anything else,
  since WGSL has no `goto`.
- New `shader_stages.py`: a small `ComputeStage/FragmentStage/VertexStage/
  GeometryStage` taxonomy plus a backend-agnostic `BufferBinding`/
  `ShaderIOLayout` description, wired into both the WebGPU and WebGL
  emitters' output.
- `wasm_html_shell.py`: the published page's JS runtime now feature-detects
  a priority order (WebGPU compute+present -> WebGL2 fragment (byte-
  identical to before) -> plain 2D canvas, since a `<canvas>` commits to one
  context type for life the first time `getContext()` succeeds with a
  specific type, so selection has to happen before any `getContext()` call).
- Found and fixed a real, pre-existing bug: the always-compiled "default
  passthrough" shader (`return red, green, blue`) traced **zero operations**
  (AbstractTensor capture only records computed steps), so
  `project_public_numerical_program` found no output to name and every
  backend reported "no outputs" as a shortfall -- the "no
  presentation_entrypoint -> fall back to passthrough" code path had never
  actually fired, on any backend, ever. Fixed with `red + 0.0` instead of
  bare `red` (load-bearing, not decoration).

**Bundle publishing CLI** (`turing/publish_bundles.py`, new):
- `--source` accepts a file, a directory (shallow), or a directory with
  `--recursive`.
- `--targets fn1,fn2` restricts to named top-level functions per file
  (class methods were never candidates -- they live inside a `ClassDef`
  body, never in `module.body`, so nothing needed adding for that).
- `--backends glsl,webgpu` restricts published source tabs -- new
  `backend_targets` parameter on `build_program_bundle` itself.
- `--full` ignores `--source`/`--targets`/`--backends`/`--probes-json` and
  instead walks every program's origin record and recompiles it as one
  additional version.

**Program-level origin + versioning** (`site_bundle.py`):
- `site/programs/<slug>/origin.json` -- one per *program*, not per version
  (a program can carry many versions sharing one origin). Embeds the full
  source text directly, so regeneration is self-contained even if the
  external source file that originally produced it has moved or been
  deleted. Written/refreshed on every `build_program_bundle` call.
- Normal builds keep today's content-addressed idempotency
  (`v1-<hash>`, same source+config -> same directory, unchanged).
- New `force_new_version=True` bypasses that and calls
  `_next_sequential_version`, producing
  `v{compiler_number}.{iteration:03d}-{date}-{hash16}` (compiler_number from
  `BUILDER_VERSION`'s trailing digits, iteration scoped to that compiler
  number and read from what's already on disk, date, then the same content
  hash as before) -- always a new, never-reused version name.
  `publish_bundles.py --full` uses this, so regenerating a program adds a
  version, it never overwrites or reuses one.
- Fixed a latent bug found while wiring `backend_targets` in: it (and
  `include_backends`) weren't part of `_content_version`'s identity, so
  rebuilding the same source with a different backend selection would have
  silently reused the first build's directory and ignored the request.

**Docs**: `PUBLISHING_BUNDLES_TO_ROOT.md` (workspace root) documents the
two-repository layout (`turing/` vs. the coordination root that
`DEFAULT_PUBLISH_ROOT` resolves to) and the CLI publishing workflow.
Root `AGENTS.md` gained a pointer to this guestbook as shared cross-project
memory.

## Observed Behaviour

- Smoke-tested `publish_bundles.py` end-to-end against throwaway
  destinations: multi-file/multi-target/backend-restricted builds; `--full`
  regenerating a program correctly even after deleting its original source
  file entirely (proving the embedded-source design actually works, not
  just in theory); two successive `--full` runs producing
  `v18.001-20260804-<hash>` then `v18.002-20260804-<hash>` while leaving the
  original `v1-<hash>` untouched.
- Full test suites for every touched module pass (site_bundle, wasm_html_shell,
  machine_targets, the new webgpu SSA backend, computational-world
  compilation, fortran fidelity -- 70+ tests across the session, all green).
- The new WebGPU JS runtime (device/pipeline/bind-group setup, the authored
  WGSL presentation shader) is syntax-checked (Node `--check` on the
  extracted JS) but **not verified against a live GPU/browser** -- no
  WebGPU-capable environment was available in this session. Flagged
  explicitly to the user as needing a live-browser pass before trust.
- 35 pre-existing published programs under the coordination root's
  `site/programs/` predate the `origin.json`/`force_new_version` work and
  have no origin record yet; `--full` correctly reports them as skipped
  rather than guessing. Manually tracing each to its real source (to
  backfill origin.json without fabricating lost probe data) was the agreed
  next step, not yet started.

## Lessons Learned

- A canvas element commits to one WebGPU/WebGL/2D context type for its
  entire lifetime the first time `getContext()` succeeds with a specific
  type -- context selection has to happen *before* any `getContext()` call,
  not by trying one and falling back after.
- "Compiles successfully" and "has ever actually been exercised" are
  different claims -- the default passthrough shader had a shortfall on
  every single build for as long as it's existed, silently, because nothing
  in the existing test suite happened to depend on the fallback path firing.
- Content-addressed idempotency and "give me a deliberately new version"
  are two different, both-legitimate needs; they don't have to fight over
  the same version-naming scheme if the forced path is explicit
  (`force_new_version=True`) rather than a global behavior change.
- When a user's instructions arrive mid-turn (this session had several,
  including a scope-narrowing correction and later a scope-*widening* one),
  re-reading the running plan against the newest instruction before
  continuing to write code caught real mismatches early (e.g. origin
  belonging at the program level, not the version level, corrected before
  more code was built on the wrong assumption).

## Next Steps

- Backfill `origin.json` for the 35 pre-existing programs by tracing each
  to its real source file in the repo (many are named demo scripts, e.g.
  `columnar_multifluid_web_demo.py`); skip and report any that can't be
  confidently traced rather than guessing. Recorded as a `.stub.md` in
  `todo/`.
- A live-browser verification pass for the new WebGPU JS runtime (device
  request, pipeline creation, the hand-authored WGSL presentation shader)
  once a WebGPU-capable environment is available.
- The fluidic/columnar module (WebGL-couldn't-do-compute, stuck compiling a
  large parallel op into WebAssembly) is the next intended consumer of the
  real WebGPU compute path, plus the region/planner work to decide how to
  slice work across it -- not started this session; `region_target_capabilities`
  in the bundle manifest (which registered `machine_targets` entries could
  serve each region) is the raw capability data a future dispatch planner
  would consume, deliberately not a planner itself.

## Prompt History

> "could you rename the function that says it's for planning glsl to reflect the fact that it's a compilation choke point for all paths and then can you put some documentation in depreciating webgl and have the webgl path explicitly require an override flag or else it will redirect to webgpu, which you'll have to put the skeleton in for in the translation tables"

> "my understanding is webgpu can handle compute and webgl cannot so we need to change places like that in the shell that default to asking for webgl and redirect them to explicitly ask for webgpu so they can use tensors and deployment, and then start building out the language tables"

> "lets reduce scope and focus on step one because when we do this, we, and I've changed my feelings on this, we are not erasing webgl it just isn't going to be default. so really all changes _are_ additive, they're just another option, and in that option we have to be as prepared as in the glsl option - which means we're going to need the same dispatcher reasoning for compute, and nothing has a setup especially for fragment. that's an important overwight and I think if we focus only there we can get it done better than trying to do everything i asked for"

> "another agent may have done some work on this, there were some crossed tasks, please audit the state of the files involved and act on the plan where it seems right to do so"

> "we need to fix the html shell to accomodate a priority list of webgpu, then webgl, then canvas, using the io buffers to communicate with any of them, and we need to introduce the compute, fragement, geom, vec classes so we can specialize the way we translate and compile. it's going to be an extensive wing on the translation but we can wire it in now while we're in the wiring"

> "there will be distributed processing cases where these programs are dispensed to clients participating in computation, so we want to be able to automatically ingest as much as a repo and offer exactly what was necessary as a bespoke binary, so we need to be able to have things so anal retentive as to have a dictionary filled with every subgraph's intended target language by a planner sorting loops out"

> "now let's give it a --full cli that will find all bundles already existent and regenerate them from their origins, meaning I guess we have to make sure we retain their origin"

> "okay so ideally you would go in and manually make the origin for each program, bundles are per version, what we want is to maybe bump the version number (everything is v1 maybe now it's 1.001) and so when there are new versions, this script would go to the original source and process it all over again into an additional single new version per program"

> "the origin is program level not version level"

> "I want you to go to speaktome an follow agent guidance around the agent area leave reports then push turing, speaktome, and then push nogods"
