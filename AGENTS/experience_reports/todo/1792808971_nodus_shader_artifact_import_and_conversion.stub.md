# Nodus shader artifact import and conversion

- [ ] Add a fail-closed Nodus parser for a shader artifact manifest referencing
      source plus Turing `CompiledProgramAPI` sidecar.
- [ ] Compile GLSL/HLSL source to SPIR-V using argv-only tool invocation; retain
      original source, SPIR-V, diagnostics, hashes, and provenance.
- [ ] Validate SPIR-V and reflect stage, entrypoint, descriptors, access, and
      workgroup size; compare reflection with `turing.shader-component.v1`.
- [ ] Emit supported derived sources through SPIRV-Cross (GLSL/ESSL, HLSL,
      MSL), preserving every artifact rather than replacing the origin.
- [ ] Refuse WGSL input by name until a tested WGSL frontend exists or Turing
      supplies an equivalent SPIR-V/GLSL sibling.
- [ ] Only after conversion is green, build a compute-row adapter over the
      existing Vulkan runner and AbstractTensor handle ABI.
- [ ] Test roundtrip conversion, malformed metadata, reflection mismatches,
      unsupported stages/languages, tool absence, and one real GPU dispatch.
