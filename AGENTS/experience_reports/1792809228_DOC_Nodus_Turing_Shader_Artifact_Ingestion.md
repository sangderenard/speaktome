# Nodus Turing shader artifact ingestion

**Date:** 2026-10-23

## Scope

Implemented the first Nodus-only unit that consumes retained shader source and
the existing Turing compiled-program/component ABI, compiles the source to
SPIR-V, validates it, and emits SPIRV-Cross translations. No Turing source and
no Nodus graph-runtime code was changed.

## Findings

- Nodus's `TranslationMatrix` is a `KernelIR` callback registry, not a shader
  source-language translation matrix, so this path is intentionally separate.
- Turing's existing `turing-compiled-program-api-v1` metadata already carries
  the required language, stage, entrypoint, and `turing.shader-component.v1`
  identity. No compiler API extension is required for ingestion.
- The installed Vulkan SDK provides `glslangValidator`, `dxc`, `spirv-val`, and
  `spirv-cross`. Its SPIRV-Cross build has GLSL/ESSL/HLSL/MSL outputs but no
  WGSL output, and no WGSL-to-SPIR-V compiler is installed.

## Work completed

- Added a strict `nodus-turing-shader-artifact-v1` JSON request manifest. It
  permits only relative, manifest-contained source/API paths and a unique list
  of known targets.
- Cross-validates the Turing API and component ABI language/stage/entrypoint.
- Copies source and API byte-for-byte into the output, hashes source, API,
  manifest, SPIR-V, and every converted source, and records origin paths.
- Invokes every external tool through an argv sequence with `shell=False`.
- Preserves argv, exit status, stdout, and stderr for every tool in a versioned
  success/failure receipt.
- Supports GLSL and HLSL input, SPIR-V validation, and GLSL/ESSL/HLSL/MSL
  output. WGSL source and target paths explicitly refuse with specific reasons.
- Wired the focused pytest into CTest.

## Verification

```text
python -m pytest -q tests/test_turing_shader_artifact.py
11 passed in 0.69s

ctest --test-dir build -C Release -R turing_shader_artifact_test --output-on-failure
1/1 passed
```

The suite performed real GLSL and HLSL compilation, SPIR-V validation, and
SPIRV-Cross output generation with the installed SDK tools.

## Next step

Turing packaging should place emitted shader source and its `.api.yaml`
together, or emit the small Nodus request manifest referencing them. This is a
delivery/packaging request, not a change to Turing's shader API schema.

## Prompt History

> something else worth asking ourselves is is can the turing shader deployment opportunities provide nodus with the shader source it would need for nodus to be able to use its spir-v matrix to convert between shader languages, so if you need turing changes for that request them and then set agents to work on the nodus side

> Implement the first complete Nodus-only shader artifact ingestion/conversion unit in C:\dev\Powershell\nodus, based on the completed audit. Do not touch Turing or graph runtime. Reuse existing systems and installed command-line tools; do not pretend TranslationMatrix is a source-language matrix. Scope: strict fail-closed versioned manifest referencing retained shader source and Turing .api.yaml/component ABI; parse enough existing Turing API schema to obtain language/stage/entrypoint; retain/copy original source with hashes/provenance; argv-safe GLSL/HLSL -> SPIR-V compilation using existing installed tools; spirv-val validation; SPIRV-Cross conversion to supported GLSL/ESSL/HLSL/MSL with explicit WGSL refusal and diagnostic receipts. Add focused tests including real tool roundtrip where available and malformed/refusal cases, wire CMake minimally. First inspect exact schemas/APIs and existing process helpers. Do not commit or push; report files/tests. Preserve current branch and existing architecture.
