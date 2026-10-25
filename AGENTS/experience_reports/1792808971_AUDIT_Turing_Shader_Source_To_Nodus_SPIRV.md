# Audit Report: Turing shader source to Nodus SPIR-V

**Date:** 2026-10-23

## Scope

Read-only audit of Turing's emitted shader artifacts and Nodus's current
KernelIR/SPIR-V facilities. No Turing or Nodus source was edited.

## Findings

- Turing already emits WGSL and desktop GLSL compute source and publishes the
  backend-neutral `turing.shader-component.v1` contract in
  `CompiledProgramAPI.metadata.component_abi`. The contract includes stage,
  entrypoint, logical ports, backend bindings, transport, and sentinels.
- `WGSLModule.write()` persists both `.wgsl` and `.api.yaml`. The desktop GLSL
  component owns equivalent `source` and `api` values but has no matching
  convenience `write()` method.
- Nodus's `TranslationMatrix` is not a shader-language conversion matrix. Its
  input is `KernelIR` and its callback returns no artifact. The registered
  `spirv` backend validates/assembles KernelIR directly.
- Nodus has a second path, `SpirvTranslator`, intended to emit GLSL from
  KernelIR and compile it with glslang/shaderc. Its `GlslEmitter` is still a
  no-op shader placeholder, so this path must not be used as evidence that
  arbitrary KernelIR semantics reached GLSL.
- The direct `KernelIR -> SPIR-V` assembler and Vulkan runner are substantive.
  The runner assumes compute stage, descriptor set 0, contiguous storage
  bindings, and caller-provided host buffers.
- This machine has `glslangValidator`, `spirv-val`, `spirv-dis`, `spirv-cross`,
  and `dxc`. SPIRV-Cross can emit GLSL/ESSL, HLSL, and MSL. Neither `naga` nor
  `tint` is installed, and SPIRV-Cross does not ingest or emit WGSL.
- Therefore Turing GLSL or HLSL can enter a source -> SPIR-V -> GLSL/HLSL/MSL
  conversion route now. Turing WGSL cannot enter that route without a real
  WGSL frontend, a Turing-emitted SPIR-V sibling, or a semantically identical
  GLSL sibling.
- Source conversion needs only language, stage, entrypoint, source, and target
  environment. Execution as a Nodus row additionally needs physical resource
  layout, dispatch dimensions/count, byte extents, access modes, and a mapping
  from logical component ports to reflected SPIR-V resources. Those details
  must be validated against SPIR-V reflection rather than inferred from names.

## Recommended boundary

Use the existing Turing `.api.yaml` as the semantic sidecar and retain the
original shader source verbatim. Add a small Nodus-owned shader artifact
manifest that references, rather than copies or reinterprets, those files and
records the compilation target plus produced SPIR-V and derived-language
artifacts. Do not route source shaders through `TranslationMatrix`, whose
contract is KernelIR-only.

## Next Steps

Captured in
`todo/1792808971_nodus_shader_artifact_import_and_conversion.stub.md`.

## Prompt History

> "something else worth asking ourselves is is can the turing shader deployment opportunities provide nodus with the shader source it would need for nodus to be able to use its spir-v matrix to convert between shader languages, so if you need turing changes for that request them and then set agents to work on the nodus side"

> "Audit cross-project documentation/research and Nodus artifact importer/row-tool contracts for a minimal shader-source artifact schema that can enter Nodus's existing SPIR-V matrix. Do not edit Turing. You may implement nothing yet: first produce an evidence-backed compatibility map and concrete Nodus subtasks/tests, noting any exact request to hand Turing workers. Avoid redesigning existing systems."
