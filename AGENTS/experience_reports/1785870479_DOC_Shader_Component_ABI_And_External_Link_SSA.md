# Shader component ABI and external-link SSA

**Date:** 2026-08-04

## Scope

Work occurred in the adjacent `turing` repository. The goal was to establish
one Python-authored execution contract for desktop GLSL compute, GLSL fragment,
and WebGPU/WGSL compute; then use that contract to formalize local compiled
links, online cross-program links, hierarchical multi-shell planning, and SSA
boundaries.

## Work completed

- Added `turing.shader-component.v1`, a backend-neutral component ABI with
  contiguous logical port slots, backend-binding/transport decorations, and a
  fixed eight-u32 sentinel header.
- Attached the ABI to real WGSL compute and GLSL ES fragment artifacts and
  their published `CompiledProgramAPI` metadata.
- Added a Python-facing desktop GLSL 4.30 compute component emitter using the
  existing fused GLSL compute backend.
- Proved that desktop GLSL compute and WGSL compute publish identical logical
  port identities for the same program while retaining language-specific
  binding decorations.
- Defined fail-closed local and online external links. Online links require an
  endpoint and message transport; local links cannot silently use online
  transport.
- Lowered every external seam to an explicit repository SSA `Call` function
  retaining scope, transport, endpoint, aliasing, feedback, component/slot
  identities, and sentinel-generation rules.
- Added topological multi-shell waves. Cyclic feedback is accepted only as a
  declared non-aliasing versioned sentinel boundary.
- Added validation that consumes the existing `PlanClosure`/`PlanCall`
  hierarchy and proves each argument/result binding maps to exactly one
  canonical component port.
- Added a serializable `turing.shader-component-assembly.v1` manifest carrying
  components, external links, shell waves, and SSA boundary records for the
  online system.

## Verification

Focused ABI tests pass. The broader WGSL, WebGL SSA round-trip, site-bundle,
HTML-shell, reversible machine, and chip-layout suites also passed:

```text
88 passed in 23.66s
```

The final focused ABI suite passed 7 tests after assembly serialization was
added.

## Next steps

- Teach the live browser/local shell runtime to allocate and validate the
  eight-word header before dispatch and publish generation/ready/error after
  dispatch rather than only carrying the formal contract.
- Have site-bundle assembly accept `ComponentAssemblyPlan` directly and route
  online-message links through a concrete endpoint adapter.
- Run real desktop GLSL compute, WebGPU compute, and fragment components as one
  mixed assembly and compare outputs across each sentinel boundary.

## Prompt History

> let's do this thing in python, glsl for compute and fragment, and in doing so, we will formalize the decoration and sentinels necessary to prep a program to run flawlessly with the onlien system, including making the same abi for glsl as webgpu between components, and then we'll transition from that into verifying the systems thatare supposed to plan when there is a complex schema of multiple shells, and we'll formalize how those get worked in and lowered to ssa right, and in doing all this it would really help us out if we could figure out our IR and ssa etc for external links, external both in the sense of our online system linking across programs and local use of compiled material that's sysem local
