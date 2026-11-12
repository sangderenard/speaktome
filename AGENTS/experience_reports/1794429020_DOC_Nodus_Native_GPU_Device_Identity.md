# Native GPU device identity and diagnostic boundary

Date: 2026-11-11 UTC.

Extended the existing Nodus Vulkan backend in
`nodus/include/kernel_vulkan_runner.h`, `nodus/include/mem_backend.h`, and
`nodus/src/kernels/kernel_vulkan.cpp` with raw device enumeration, UUID/LUID
identity, exact indexed context selection diagnostics, and optional 64-bit
memory-budget snapshots. No parallel scheduler, GPU row, renderer, GL adapter
selector, automatic fallback, or workload replay was created by this change.
The existing native ITool/ToolRegistry/ThreadManager/EdgeTensorFifo route is
the owner of admission and execution. A sibling change owns the native GL
harness and GPU-row integration.

Standalone native CPU regressions passed with MSVC 19.44: the mocked Vulkan
inventory/selection/status test and the build-without-Vulkan refusal test.
Mocks preserve raw ordinals, reject noncompute and invalid selections, retain
driver error detail, test partial-context cleanup and exceptions, and abort
if shaders, buffers, pipelines, or submissions are accidentally reached.
No real Vulkan enumeration, logical device creation, shader link, dispatch,
draw, or rendering was run by this work. Full integration remains unverified.

The prior protected Studio GL link log identifies RTX 3060 / NVIDIA 610.88.
Read-only Windows inventory identifies RTX 3060 (10DE:2504), GTX 1660 SUPER
(10DE:21C4), and AMD Radeon Graphics (1002:164E). Windows list position is not
a Vulkan or GL ordinal. Win32_VideoController AdapterRAM is only 32-bit and
must not be used as reliable capacity. Vulkan memory budgets are not GPU
utilization or process/system RAM measurements. Vulkan inventory cannot
advertise adapters as live GL contexts. C++ try/catch does not contain a
driver crash or system RAM exhaustion; protected shader-link limits remain
necessary.

See `nodus/docs/NATIVE_GPU_DEVICE_IDENTITY.md` for APIs and verification limits.
Tests live in `nodus/tests/vulkan_device_query_test.cpp` and
`nodus/tests/vulkan_unavailable_test.cpp`; initial standalone build receipts
are in `nodus/build-device-query/`.

The normal CMake build of only these two tests triggered unexpected vcpkg
dependency regeneration in `nodus/build-service/vcpkg_installed`. Its verified
owned process tree was stopped. Ten packages are now absent, with removals
recorded in the vcpkg updates journal; vcpkg-cmake was restored. No live
Release DLL was replaced. The developer document gives the exact same-root
vcpkg resume command for the next deliberate full native build. Do not trust
the stale base status file or delete journal/package directories. Final
standalone CPU builds/tests still passed and `git diff --check` passed.

## Authorized physical-query follow-up

At 2026-11-11 20:46:09 UTC one bounded native test of the actual query APIs
completed in 1,126 ms, exit 0. Vulkan raw ordinals are 0=GTX 1660 SUPER,
1=AMD Radeon Graphics, 2=RTX 3060, differing from Windows enumeration order.
All three expose compute queue family 0, valid UUID/LUID and optional memory
budgets. Four Vulkan instances were created and destroyed. Executable-local
guards prohibit logical devices, buffers, shaders, pipelines and submissions.
No GL context or affinity was established. The complete identities and heap
snapshot are retained in `nodus/build-device-query/inventory-20261111T204609Z.stdout.log`
and the native integration source is `nodus/tests/vulkan_inventory_integration_test.cpp`.
No dependency installation or build-tree repair was attempted for this query.

## Prompt History

User: “we want this development work in native source, bespoke, because it's going to be a vital kernel in nodus”

User: “we are going to nodus now we are making native source that's going to use the tool row api, but it doesn't necessarily need to care about inputs or outputs except that it's getting a shader program, place it into the first best open gpu. gpu's will have their harness built into the gpu row. nodus routes the programs and they enter fifo and get cranked out by gpus that will have ports in that can recieve automation and deliver telemetry, the pieces are all in the canvas shell it's actually quite advanced”
