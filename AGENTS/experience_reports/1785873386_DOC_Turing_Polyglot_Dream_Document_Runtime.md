# Turing polyglot dream-document runtime

**Date:** 2026-08-04

## Scope

Work occurred in the adjacent `turing` repository. The simulator was requested
as one new multi-language document whose comment-block sentinels switch
computational arenas, declare parallel deployment, and place shaders directly
on the device. The recent card/tape graph was to be the division, caching, and
loading mechanism.

## Work completed

- Added the byte-scanned `turing.dream-document.v1` container and `.dream`
  source format.
- Added segment, shader, parallel, and closing sentinels. Headers are parsed as
  restricted ASCII before any language frontend; payloads use their declared
  encoding and may carry a SHA-256 assertion.
- Made shader blocks intrinsic in-place GPU deployments. They require only
  their language and compute/fragment stage, not an additional switch/launch
  directive.
- Projected every executable block into the existing `turing.card-graph.v1`
  ABI with content-hash cache keys, a lazy linear read path, resident boundary
  connections, parallel-deployment records, and language/stage metadata.
- Added a graph read-head runtime. Parallel members are submitted together in
  a thread pool and joined without manufacturing a shared-state lock; declared
  ports or an intentionally shared host arena are the communication contract.
- Added a GPU activity callback around every shader deployment.
- Added an explicit opt-in handler for trusted Python blocks sharing one arena.
- Authored `examples/reversible_chip_simulator.dream`, containing Python chip
  setup, a Python reversible head tick, GLSL register-light compute, GLSL chip
  presentation fragment, and a JavaScript GPU indicator. The CPU tick and
  compute shader form one parallel frame deployment.
- Added a CLI to inspect the document's card graph or reference-run it with
  visible GPU ACTIVE/IDLE transitions.

## Verification

The live reference command ran the document in card order and displayed both
shader activity transitions. Parser, graph, real parallel overlap, shared
Python arena, GPU indication, framing/hash failure, card-graph, component ABI,
WebGPU/WebGL, site bundle, HTML shell, and reversible-machine tests passed:

```text
96 passed in 33.85s
```

## Next steps

- Replace the reference shader callback with a live desktop OpenGL deployer
  that compiles the GLSL compute/fragment blocks in place and binds canonical
  component ABI ports/sentinels.
- Add a display surface around the fragment block so the register/cache chip
  and GPU lamp are visible rather than only reported in the CLI.
- Feed repeated frames through `BoundedMachineClock` and the reversible
  machine journal, preserving the per-frame parallel deployment.

## Prompt History

> so that's what i want us to build the simulator in, a new document using blocks just like that to manage the different computational arenas, we use languages we are comfortable with for each task, we make a dream document that is this simulator and we run it using the tape graph system for dividing programs and loading them that was just recently worked on, for now we could just have an indicator when the gpu is active - one big multi language program that doesn't bother with any of the difficulty of locking it just does what it wants to do and notates language changes, parallel deployment areas, shaders just deploy in place so they don't need any special notation for changing
