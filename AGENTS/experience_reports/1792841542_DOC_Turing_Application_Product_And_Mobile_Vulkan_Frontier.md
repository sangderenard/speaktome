# Turing application product and mobile Vulkan frontier

## Scope

Inspected Turing's compiler, Wasm/site publishing, compiled-program API,
ShellIO, and shader deployment paths to identify the shared application
boundary for web, Android, and iOS. Advanced that boundary without replacing
the existing segmented Wasm execution or Nodus shader-ingestion machinery.

## Receipt

- Added `turing/src/compiler/application_product.py` with the
  `turing.application-product.v1` reference-only product. It points to the
  compiler-owned program API, execution artifact/manifest, shader deployment,
  and runtime dependencies instead of copying their ports or resources.
- `site_bundle.py` now publishes `application/program.api.yaml`,
  `application/web-execution.json`, and `application/application.json`, and
  references the application product from `bundle.json`.
- Added the new publisher to the immutable bundle compiler fingerprint.
- Added pinned mobile Vulkan dependency receipts: Android's NDK-owned Vulkan
  loader and MoltenVK 1.4.2's iOS static XCFramework. The MoltenVK archive is
  pinned by SHA-256 and minimum iOS 15.
- Downloaded and hash-verified the official MoltenVK iOS 1.4.2 archive into
  Turing's ignored `build/mobile-dependencies/`. The required static
  `MoltenVK.xcframework` files extracted. Windows could not create the archive's
  dynamic-library symlink, which is immaterial to the selected static product.
- The workstation already has Vulkan SDK 1.4.335 tools including SPIRV-Cross,
  glslangValidator, spirv-val, and DXC. No Android SDK/NDK installation was
  discovered.

## Architecture decision

The application is the launch of one compiler-owned entrypoint plus a primary
presentation component. The application record is a composition manifest, not
a new ABI. Each platform target references the same logical program API and
shader component/deployment records while selecting a physical execution
artifact and runtime dependencies.

Android should use a thin Kotlin/Java lifecycle shell with a C/C++ NDK runtime,
native Vulkan, and SPIR-V shaders. iOS should use a thin Swift/Objective-C++
lifecycle shell with the same C ABI runtime. The first iOS renderer should use
MoltenVK so it can consume the same Vulkan calls and SPIR-V. A later direct
Metal lane can translate SPIR-V to MSL with SPIRV-Cross and compile the MSL to
metallib on an Apple build host. Vulkan is the graphics API; SPIR-V is the
portable shader input.

Turing/Nodus already owns a strict source-to-SPIR-V-to-MSL ingestion path, so
no duplicate shader translator was added to Turing.

## Validation

- `py -3.11 -m pytest -q tests/test_application_product.py tests/test_site_bundle.py::test_only_published_browser_shader_graduates_a_bundle_page`
  passed after the final entrypoint-negative test was added.
- `git diff --check` passed except existing line-ending warnings.
- The full real-bundle test currently fails before application-product
  publication in `glsl_deployment_strategy.py` with
  `planner_specialization: no source cell ... the binding came from the caller
  of the compile`. This existing compiler frontier was not masked.
- No iOS compilation was claimed: Apple/Xcode validation requires a macOS host.

## Files owned by this work

- `turing/src/compiler/application_product.py`
- `turing/src/compiler/site_bundle.py`
- `turing/tests/test_application_product.py`
- this report and its todo stub

Pre-existing edits in LLVM backend/tests, the baseline receipt, and unrelated
Speaktome files were not touched.

## Prompt history

> I want you to look at the backends on the Turing compiler and i want us to work out what backend we should have for Android and ios app development, what language backend should they use and how do we standardize an app as the launch of that shell plus a main output shader

> Okay so, we gave web shaders web assembly Vulcan, we have most of this, but right now our web assembly backend is mildly divorced from the other backend paths by a ton of bespoke code for certain websites

> After you do that i want you to work on advancing this frontier, getting the necessary native libraries for android and ios, there's a way to compile metal from Vulcan or spirv no?
