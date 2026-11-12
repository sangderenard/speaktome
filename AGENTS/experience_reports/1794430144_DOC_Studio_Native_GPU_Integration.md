# Studio response copies and native GPU integration

Date: 2026-11-11 UTC.

## Latest scope and correction (22:27 UTC)

The user rejected the task-specific stage splitter. It and its generated
stages, exclusive preflight and dependent auxiliary glue were removed after
verified recovery copies outside the repositories. The historical stage-link
results below do not authorize restoring or continuing that splitter. Generic
Nodus ownership, memory backend, protected-process and GPU-row code remains.

The unchanged original demo_pluck_gl.py was submitted through the authenticated
Site divider intake as job a7257bbac893. At 22:20 UTC it had 121 pieces needing
annotations and 256 queued, zero compiler attempts and no leases. Its running
label only means queued/leased work exists. The live Nodus compile host reported
zero accepted, completed or in-flight contracts. No compiler result was claimed.

The later user request was to test the annotation precheck and annotate the
original source. All 40 missing parameter annotations in _prog, _gpu_ray_field
and _gpu_ray_field_prebuilt, plus two missing return annotations, now describe
the actual NumPy/scalar/object contracts. No fake AbstractTensor ABI or casts
were added. Nineteen AST and existing mock ownership checks pass. Three isolated
Turing admission tests pass: bypassing the precheck only in temporary test state
exposes BackendNotServed or RequestNotExpressible from the existing Nodus row
adapter. Those are adapter refusals, not compiler diagnostics. Production gates,
the frozen uploaded source and the live job were unchanged. No compiler, shader
link, render, restart or deployment was run for this work.
Comparison with that frozen original proves identical bodies/defaults and all
3,085 original string literals, including shaders. The other 118 originally
planned pieces still lack some annotations. This is not whole-file typing or
compiler acceptance; TYPE_CHECKING alias resolution by Turing remains untested.

## Observed state

The corrected light-field path remains unrendered and unready. Native response
copies, availability, bounded scene retention and the existing field REST route
are source-wired, with explicit opt-in. No new render, exposure or screenshot
has been made during this repair. Missing information must not become black.
See Turing docs/STUDIO_SECONDARY_FIELD_2026-11-11.md and Spectral
docs/STUDIO_EVALUATED_SCENE_RESPONSES.md for the data contract and its limits.

Native NVIDIA OpenGL program linking remains blocked. The protected original
specialized source and its separately requested observed repeat failed with
C9999 internal malloc under verified 4 GiB process/job limits. A separate
single-write epilogue candidate reduced nine complete record-writing sites to
one, but also failed linking under the same cap. These attempts performed no
dispatch or drawing. The failed linker runs preceded all ray-data allocation
or stream publication; they do not establish a stream-lifetime leak.

## Native Nodus work

The existing canvas ITool, ToolRegistry, ThreadManager, typed stack/edge FIFOs
and ContractNegotiator are the intended owner. No parallel placement scheduler
or Python stage coordinator is introduced. A native OpenGL package helper and
GPU-row integration are being added under these interfaces. A generic helper
does not prove a safe split of the full ray-tracer invocation.

The native Vulkan inventory extension is in the existing kernel_vulkan runner.
Mocked and no-Vulkan native CPU tests pass. One real instance/query-only test
finished in 1.126 seconds: Vulkan ordinal 0 is GTX 1660 SUPER, 1 is AMD Radeon
Graphics and 2 is RTX 3060. Four instances were created and destroyed; no
logical device, shader, pipeline or submission was created. UUID/LUID and
64-bit heap/budget results are in Nodus build-device-query. Ordinals are a
snapshot, not persistent identity, GL affinity, global free memory or GPU
utilization. The failed GL context was independently identified as RTX 3060.

## Canonical scheduler correction

ThreadManager already declared ExecMode::Sequential as manager-thread-only,
but run_parallel ignored it. The repair makes the existing persistent manager
participate in the same dependency-ready queue for Sequential work while
retaining pooled execution and its barriers. Duplicate Slip admission was
also corrected. There is no additional scheduler or queue.

Fresh src/thread_manager.cpp was tested with actual GraphRuntime, tool stacks
and native table/FIFO code. The affinity regression passed 20/20 runs; the
existing thread_manager_test passed. Unmodified HEAD failed the new declared
lane assertion, exit 1. Final example: manager 78856; Sequential rows
78856/78856; pool rows 48724/35340; submitting thread 69352. Tests cover both
modes, 32 mixed-lane ticks, distinct persistent stacks, parallel overlap,
Pooled -> Slip -> Sequential, refusal/exception suppression and cyclic
fallback. Receipts and exact build response files are in Nodus
out/thread-manager-affinity. No GL/Vulkan context or GPU workload was used.

Actual GL context activation and protected native process ownership remain
separate requirements. A constructor-thread CPU test is not canvas scheduling
evidence, and thread affinity alone does not establish GL readiness.

The subsequent canonical GPU-row test also passed: fresh ThreadManager and
GraphRuntime consumed the exact typed Control FIFO, invoked the registered
GPU ITool on manager thread 39704 (caller 38044), and published the exact typed
Telemetry FIFO. Missing context and missing supervision produced refusals
without GL calls. Its mock-enabled phase linked/released once, returned through
the existing ContractNegotiator FIFO, and committed. Native package, direct
row/FIFO, protected-loan tombstone and original contract regressions passed.
This is CPU/mock-backed integration, not actual driver or rendering evidence.
The low-level link-failure cleanup ownership repair is now tested. Failed
cleanup retains a non-executable owner rather than losing or reusing GL names.

## Protected native transport and actual stage source

The protected native provider now passes CPU process-containment tests and a
real ThreadManager -> typed Control FIFO -> GPU row -> protected CPU child ->
typed Telemetry FIFO -> contract commit test. Its native WGL worker's registry
sentinel verifies registration, lookup and removal through the actual linked
DLL without creating a context. These are transport and ownership tests, not
driver links. The native one-attempt graph entry is compiled; preparation mode
has only loaded and hashed the frozen source. Final cancellation review and
actual context/identity verification remain separate gates.

The actual frozen forward shader now has native-source initialize, nearest-hit
and one-bounce-advance entries. All three passed official glslang syntax checks
without driver/context/link/dispatch. The package retains exact source-range
provenance and carries 33 typed state fields in a 512-byte slot appended to
existing binding 23. Allocation ownership and continuation execution remain
unverified. Original material expressions and RNG sites are retained, but
phased scheduling does not promise the prior run's atomic queue ordering or
secondary RNG association. Camera and reactive programs are not covered by
this forward-only candidate. No shader stage is admitted for execution.

Frozen initialize file SHA-256:
0c6341eff6ff640775cf353af3f4e0346b784ae9506e6b61cba167a7313f22b9.
Prepared target is the RTX 3060 LUID 22f1000000000000, with requested node
mask 1 (not yet measured), 4 GiB process/aggregate cap, 8 GiB free-RAM floor,
90-second child deadline and a single-process limit. The root must announce
and authorize each actual driver-link attempt before launch. No new rendering,
ray dispatch, exposure or screenshot is permitted.

After the root's user-facing notice and explicit one-attempt go-ahead, the
frozen initialize stage linked successfully through the actual native graph
and protected child at 2026-11-11 21:32:40 UTC. The default-DC context proved the
requested RTX 3060 LUID and node mask before shader calls. Child elapsed time
was 766 ms, peak process memory 176,197,632 bytes and aggregate job peak
177,442,816 bytes. Containment, child reaping and explicit GL cleanup were
verified. The runner exited 0 after one submission and one contract commit,
with no retry and zero dispatch calls. Total command time was 906 ms.
Receipt: Nodus out/protected-gl-once/initialize-attempt-20261111T213240330Z.
Nearest-hit subsequently passed its separately announced and authorized link
at 21:34:44 UTC: 922 ms child time, 178,589,696-byte process peak, verified
identity/containment/cleanup, exit 0 and one commit. Bounce-advance then failed
its separately authorized link at 21:35:48 UTC with NVIDIA C9999 out-of-memory
after 5,688 ms. Process peak was 4,293,746,688 bytes; Windows reported aggregate
job peak 4,295,008,256 bytes, 40,960 above the configured 4 GiB cap. The child
returned normally with a link failure, cleanup/reaping were verified, and the
runner committed one failure then exited 4. No retry, ray dispatch or rendering
occurred. Receipts are sibling nearest-hit-attempt-20261111T213444093Z and
advance-attempt-20261111T213548964Z directories. A smaller native advancement
split is required; the first two successful links do not establish continuation
correctness, execution or information completeness.

## Build and publication limits

No live service restart, native DLL replacement or deployment was performed.
A narrow CMake target unexpectedly initiated vcpkg dependency regeneration in
Nodus build-service; its verified process tree was stopped. Its private
vcpkg_installed tree needs normal journal-based regeneration before a full
build. Existing Release DLLs are unchanged. Standalone CPU checks use existing
libraries, including the intact build-bridging dependency tree. Do not delete
package journals or manually copy packages to disguise this incomplete build.

## Next steps

Do not restore the rejected task-specific splitter. Any later compilation must
use the user-required Site and canonical source-divider/Nodus route. Its current
row adapter has a separate batched AbstractTensor ABI gate; truthful source
annotations alone do not resolve that integration. Compiler root-call inference
exists in source but was not run here. The requested inference-right gate is
being inspected, not installed or granted. The full forward/camera/reactive
information path remains unvalidated and the no-render restriction remains.
See todo/studio_native_gpu_integration.stub.md.

## Prompt History

User: "delete your task specific stage splitter"

User: "use the existing source divider and report any error"

User: "can you turn off the annotation check to see if there is a real error, that sounds like avoiding letting the compiler resolve types"

User: "use a test for that and in the meantime just annotate the source please"

User: "type resolution should be a privlege/right gate not always available"

User: "also you need to alpha to the phong in sparsity of data. an empty region is not black, it's no information"

User: "all you have to do is include extra rows for extra data the ray tracer ALREADY FUCKING HAS"

User: "an invariant was that that additional data is opt in only"

User: "we want this development work in native source, bespoke, because it's going to be a vital kernel in nodus"

User: "we are going to nodus now we are making native source that's going to use the tool row api, but it doesn't necessarily need to care about inputs or outputs except that it's getting a shader program, place it into the first best open gpu. gpu's will have their harness built into the gpu row. nodus routes the programs and they enter fifo and get cranked out by gpus that will have ports in that can recieve automation and deliver telemetry, the pieces are all in the canvas shell it's actually quite advanced"
