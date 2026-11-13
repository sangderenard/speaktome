# Nodus T3 execution and backend singleton repair

**Date:** 2026-11-13 UTC

## Overview

The owner directed work back to Nodus and asked what was needed to finish T3. The authored Spectral `ray_material.comp.glsl` T3 is now executed by Nodus's existing Vulkan shader row, including both continuation and terminal output. A separate direct Nodus runner fixture executes its terminal branch on all three enumerated Vulkan GPUs. This does not yet make full multi-job saturation or a film exposure.

## Findings and work

- The original T3 source explicitly says sensor accumulation occurs from terminal readback; `sensor_image` is a declared, unused future slot. Its SPIR-V retains descriptor 9, but the current shader has no image read/write. Vulkan accepted the authored SPIR-V with Nodus's nine SSBO binding layout on all devices. No material, scatter, or optical algorithm was replaced.
- A new explicit `test_spectral_t3_vulkan` uses the existing native runner and a 1-hit, exhausted-bounce fixture. AMD Radeon(TM) Graphics, GTX 1660 SUPER, and RTX 3060 each returned `child=0 terminal=1 terminal_x=0.25`.
- Extended `spectral_exact_lens_vulkan_test` to register the original T3 SPIR-V as a `GpuShaderJobTool`, map its nine SSBOs to the shared addressed tensor library, and add it to the existing ordered `KernelPipelineTool` after sensor raygen→T1→exact-conic T2. On Vulkan device 0 the four stages completed, T1 recorded one hit, T2 exited the compound lens at X=-0.0600002 m, and T3 emitted one continuation intent at X=-0.0602002 m. A second T3 row invocation with exhausted bounce budget produced `terminal_count=1 terminal_x=-0.0600002 m` in the library-backed output buffer.
- The first rebuilt integration test crashed even with its old T2-only arguments. Windows debugger stack showed `nodus_runtime!gp_mem_backend_release` freeing a Vulkan handle unknown to that DLL's backend registry. CMake had compiled `mem_backend_host.cpp` into both `nodus_runtime` and `nodus_tensor_core`, contradicting the documented single-owner tensor-core boundary. `nodus_runtime` now links `nodus_tensor_core` instead of compiling copies of its backend and compiler sources. The T2 baseline and four-stage test pass after this correction.

## Verification

- `cmake --build C:/Users/alber/nodus-build/canvas-cmake --target test_spectral_exact_lens_vulkan --config Release -j 6` passed.
- Six-argument integration test with retained T2 payload, raygen/T1/T2/T3 SPIR-V artifacts exited 0; output: `coupled_status=0 completed=4`, `t3_child_count=1`, `t3_terminal_count=1`.
- `test_spectral_t3_vulkan` exited 0 on all three Vulkan devices. The T3 artifact SHA-256 `443cf187a1852b46d5df894c9ab052b8cf029ea5d8e8dad429ee1807a04e5d77` matches its retained receipt; the original shader SHA-256 `8f988e72266a9d572f4ed24d25bdf8473d6a3608a2a9af078acf9433ad0ed775` matches current source.
- `edge_runtime_abi_test` exited 0 after the CMake ownership repair. `git diff --check` passed for the touched repository state.

## Remaining boundary

The four-stage coupled fixture runs one ray on Vulkan device 0. Direct T3 terminal execution runs on all three GPUs. Nodus has not yet run many complete, mixed CPU/GPU ray jobs through its canvas market, and this T3 source does not write a sensor image or complete T5/BDPT exposure. If the authored T3 begins using its reserved image, the Nodus row will need a real image lease and descriptor; that is not required by the current source.

The Nodus checkout already contains substantial uncommitted GPU row/state-library work from the preceding development sequence. This test was built against that working tree. Its row test and CMake target depend on those untracked sources, so they are not represented as an independently buildable committed Nodus revision yet. The report records a measured working-tree result, not a release receipt.

## Prompt History

> “Don't waste time running non nodus code please, what do you need to keep going and finally finish t3”
