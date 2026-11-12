# Native Studio forward stage candidate

2026-11-11 UTC. Added isolated Nodus native C++ source authoring and CPU tests:
`include/studio_forward_stage_package.h`, `src/kernels/studio_forward_stage_package.cpp`,
`tests/studio_forward_stage_package_test.cpp`, and `docs/STUDIO_FORWARD_STAGE_PACKAGE.md`.
No existing producer, runner, scheduler, row, allocator or production wiring was changed.

The exact 116387-byte forward source, SHA-256
`8718336e6144a90a1da9d40c87b29a9a8457558908fe2f133eadcaf41832048c`,
is decomposed into initialize, nearest-hit and one-bounce advancement entries.
All copied native ranges carry byte provenance. The original complete post-hit
body and its break/continue targets are retained. The typed carried state has
33 fields, 84 used uint32 words in a 128-word slot, appended only to an explicitly
bounded owner view of existing binding 23. No seventeenth SSBO is introduced.
All stages remain GP_GL_UNVERIFIED_CONTINUATION and are refused by the native runner.

MSVC 19.44 CPU tests passed. Authorized offline syntax-only validation with the
already installed official glslangValidator 16.0.0 passed all three entries with
exit 0 and no diagnostics, under separate 10-second process timeouts. No real
shader driver, GL/Vulkan context, dispatch, image, exposure or screenshot was used.
Final sources and receipts are in `nodus/build-studio-stages/package`.

The source reachability audit separates the partition/divider-root path from
segment/material advancement. Boundary advancement still performs its original
classification/BVH/sphere work. No actual linker-memory improvement is measured.
Forward phase ordering can refine native-allowed unordered invocation schedules;
it does not reproduce a prior atomic queue order. Saturation retention and later
queue-index-derived reactive RNG association can therefore differ from a prior
run. This is not automatically a new forbidden outcome. The exact assumptions
and remaining numerical/execution admission limits are in the Nodus document.

## Prompt History

The owner's native-source direction is recorded verbatim in the related native
GPU identity report `1794429020_DOC_Nodus_Native_GPU_Device_Identity.md`:

> we want this development work in native source, bespoke, because it's going to be a vital kernel in nodus

The current task explicitly retained the no-rendering boundary and required
source/CPU work first. Later task authorization added only bounded offline syntax
validation of the emitted entries. Neither permission authorized real driver
shader links or dispatch in this work.
## V2 advancement split after measured protected-link failure

The subsequently measured v1 advancement driver link exhausted its protected
4 GiB host-memory cap; initialize and nearest-hit linked successfully. Those
were separate protected operations, not shader execution. A separate native
v2 source package now divides advancement into segment/deposition, material,
and response-flush entries. V1 files and generated artifacts remain unchanged.

V2 files are `nodus/include/studio_forward_stage_package_v2.h`,
`nodus/src/kernels/studio_forward_stage_package_v2.cpp`,
`nodus/tests/studio_forward_stage_package_v2_test.cpp`, and
`nodus/docs/STUDIO_FORWARD_STAGE_PACKAGE_V2.md`. Frozen sources and receipts
are in `nodus/build-studio-stages-v2/package`.

The 57-field ABI uses 256 uint32 words (1024 bytes) per batch slot. It carries
all response working snapshots with valid-bit masks, the original computed hp,
and separate call-time terminal arguments. The old epilogue candidate supplies
only its exact argument-latching helper. All nine original terminal argument
expressions and native material/control bodies remain unchanged. Only flush
reaches the original response writer; material reaches neither it nor DDA.

Native CPU tests, all five official offline syntax checks and actual official
preprocessor-output callgraph checks passed. Effective reachable definitions
are 27/25/36/72/9. Independent read-only liveness review found no concrete defect
under fixed-input, single-owner, serial-barrier assumptions. No v2 hardware
context, query, link, dispatch or rendering was performed by this source work.
All outputs remain explicitly unverified continuations, with dynamic state/row
parity and end-to-end path completeness still unproved.
## V3 segment isolation after measured protected-link failure

A separate protected owner subsequently measured v2 material link success and
v2 segment link failure with NVIDIA C9999 host allocation failure. This source
task did not run those driver operations. V3 isolates exact native directional
record, Beer attenuation, scalar multiscale splat and wave/capture statements.
It adds no optical formulas and preserves the v2 57-field, 1024-byte slot with
version3 and remapped eight-stage phases. Positive directional completion never
reads uninitialized response scratch. Native segment source spans partition
exactly; all original material/control/RNG and terminal argument strings remain.

All eight MSVC native CPU/provenance tests and official glslang offline syntax
and preprocessing checks passed. Reachability is27/25/22/15/22/14/72/9.
Independent read-only review is green under serial owner/barrier/fixed-input
assumptions. V1/v2 frozen source hashes remain unchanged. V3 native files and
receipts are documented at nodus/docs/STUDIO_FORWARD_STAGE_PACKAGE_V3.md and
nodus/build-studio-stages-v3/package. No graphics, source scheduler, production
wiring, rendering or whole-family completeness was admitted by this work.