# Nodus camera kernel pipeline adoption

Date: 2026-11-13 UTC.

The owner requested native, GLSL, and Vulkan attempts for Studio Phong,
thin/compound lenses, forward/backward rays, and BDPT, with ordered
multi-language components represented by a tensor buffer. The authoritative
thick-lens model is Spectral Analyzer's exact conic `CompoundLens`, not Turing
Studio's derived analytical thin-lens contract.

Nodus now has `KernelPipelineTool`, an `ITool` composition row consuming
`AbstractTensor` U64[N,7] work strings and returning U64[N+1,9] receipts. It
calls existing registered component rows, shares the addressed state library,
checks API/program identity before any stage, and stops on the first failed
stage. A native test ran Vulkan fabric then Turing LLVM CPU on the same state;
another ran two Vulkan exact-lens components on mapped tensor addresses.
`GpuShaderJobContract.binding_offsets` permits adjacent stages with different
GLSL binding numbers to refer to the same logical state addresses. OpenGL is
explicitly refused by this pooled row because its context and memory are
process/lane owned.

Spectral's exact-conic T2 shader was compiled with only a uniform-storage ABI
change. `glslangValidator`, `spirv-val`, and reflection passed. Using bytes
from the original `CompoundLens.build_gpu_payload()` simple doublet, an
off-axis ray passed all three Vulkan devices with identical exit position
(-0.0499998, 0.00125763) m, Y direction 0.049325, and teleport flag 4 with no
absorption. Its source Python scalar trace produced Y=0.00125423 m and Y
direction 0.04851; the active GPU spectral band differs from the scalar
reference, so this is contextual, not an exact numerical parity claim. T5
compiled and validated with its original native BVH/material preamble, but
was not executed. T1 had already executed a 16,384-ray hit trial on all GPUs.
Spectral's original sensor mip raygen also compiles and executes through the
state library. On all three Vulkan devices, its one-sample aperture fixture
produced +X launch in target mode 0 and -X reciprocal launch in mode 1,
with one intent recorded each time. Aperture radius was zero for this ABI
trial; it is not a lens image or full sensor response.

A further three-stage mapped-address pipeline ran sensor raygen→T1 BVH→T2
exact-conic on Vulkan device 0. The raygen wrote the intent that T1 consumed;
T1 recorded one lens-proxy hit into the same HitRecord that T2 consumed. T2
traversed the original doublet backward from the sensor side and exited at
X=-0.0600002 m with flags 5 (camera path and teleport, no absorption). All
three stage receipts succeeded. This is coupled transport, still only one ray
with zero aperture radius and no T3/film exposure.

T3 now also compiles to a validated Vulkan artifact by moving its OpenGL
default uniforms to SSBO 8 and its sensor image to descriptor 9 (OpenGL had
used numeric binding 7 for both image and SSBO). The current Nodus shader row
cannot dispatch it because it only leases storage buffers. Phong still needs
descriptor/resource adoption. Phong is a raster
shader pair whose frame can become a tensor product once a graphics runner
owns and synchronizes its render target; it is not a compute dispatch.

Follow-up inspection found that no new image container is needed for a frame.
Nodus already has dense HWC `AbstractTensor` image helpers, tensor-handle FIFO
transport, a state-library `Put` operation, and tensor PNG export. The missing
piece is the graphics runner that draws into a target and transfers its
completed color attachment into an ordinary tensor. Its receipt should carry
frame generation and color/origin interpretation. A CPU readback can serve as
the first boundary; GPU-resident publication needs image-to-buffer transfer
and synchronization in the runner.

The owner clarified that captured frames should be compressed through the
existing exec row and stored in the user's SQL-owned job. Inspection confirms
that `exec_row` transforms tensor bytes into a U8 compressed tensor but never
writes a file; it also drops the input shape/dtype unless those are serialized
with the payload. `PhotoJobs.job_directory` resolves an admitted digest through
the `job_owners` SQL index to the account-scoped asset directory. The existing
worker writes files there and updates `status.json`; `service_jobs` tracks
in-flight work rather than holding image bytes. The intended path is therefore
frame tensor → descriptor/payload span → exec compression row → trusted job
host atomic file commit → status update. No graphics runner or job-host
attachment was claimed complete by this inspection.

The owner then recalled file FIFOs. Inspection confirmed `EdgeTensorFifo`
supports a filesystem backend that writes its storage to a file, and
`gp_meta_edge_snapshot_capture_to_edge` publishes filesystem buffer handles
through a FIFO. The prior statement that a separate writer was necessarily
needed was too strong. The file FIFO's ring slots are reusable, and the
backend's generated `fb_*` file name does not itself become a committed job
artifact. The trusted job host still must bind it to the admitted owner/job
directory and finalize the status/asset reference after compression.

Native lower-path implementation followed. `FrameTensorPacketTool` retains
dtype, shape, color encoding, origin and exact pixels before the existing exec
row transforms the byte span. `FrameArtifactFileTool` writes the compressed
output through Nodus's filesystem backend into a host-selected job directory,
closes the file handle before rename on Windows, publishes a generation
receipt, and refuses duplicate generations. A registry helper exposes both as
job-specific canvas rows. The real exec row with gzip and a test F32 image
passed compress → file → decompress → packet decode with exact pixel parity.
The host sets separate raw-packet and compressed-byte caps; the sink test
refuses an oversized compressed token without creating a file.
An isolated `job_library.claim`/`PhotoJobs.job_directory` check proved the
file lands under `account:7`'s directory and account:8 is denied by the owner
index. These tests did not draw a Phong frame or attach a live server job.

Next, an actual Phong draw was obtained from Spectral Analyzer's original
`BaseGLRenderer` and original `base_material` vertex/fragment sources on a
local GL context. The 256×256 RGBA frame was passed through the Nodus packet,
exec-row gzip, filesystem sink, exec-row unzip and packet decoder with exact
byte parity. An isolated SQL-owned `account:7` PhotoJobs directory held the
3,352-byte compressed result; account:8 failed owner lookup. This proves the
source-rendered lower path, while the protected Nodus graphics process is
still link-only and the live service did not receive a new job or status.

A further native transfer seam, `gp_gl_frame_read_rgba8`, was added to the
existing Nodus OpenGL kernel ABI. It takes the caller's current read framebuffer
and returns a quota-bounded U8 HWC AbstractTensor in bottom-left GL order.
It verifies the registered native context and restores GL pixel pack state.
`tests/opengl_frame_readback_test.py` passed on a real Windows GL context:
every byte matched direct `glReadPixels` from a two-region draw, nondefault
pack row/skip settings were restored, and an undersized quota was refused.
This transfer is not yet invoked by the protected link-only child or the live
PhotoJobs worker; there is still no native raster draw host in that process.
The real-context test was extended to have Spectral Analyzer's original
`BaseGLRenderer` draw its `base_material` Phong shader pair before the new
Nodus readback. Its 64×64 lit blue surface had more than two distinct colors,
and the Nodus tensor matched the GL readback byte for byte. The image is at
`C:/Users/alber/nodus-build/frame-phong-proof/native-transfer.png`.
Those native readback bytes then completed packet → actual exec-row gzip →
filesystem file → exec-row unpack → packet decode with exact byte parity.
The compressed 64×64 frame was 820 bytes, with bottom-left origin retained.
An isolated SQL admission of those same bytes assigned the digest to
`account:7`; `PhotoJobs.job_directory` selected that account's asset folder,
the native file row committed there, and `account:8` failed the owner check.
No live service database or worker was modified.

The owner's automation request led to `glsl_uniform_block.py`, a shared strict
transform used by sensor raygen, T1, T2, T3, and T5. It rejects unsupported
declarations and does not silently drop initializers or opaque samplers. All five adapters
regenerated validated SPIR-V after the refactor, and the T2 off-axis execution
test passed again. The full matrix, limits, and continuation are in Nodus
`docs/KERNEL_PIPELINE_ADOPTION_2026-11-13.md`.

## Prompt History

User: “We have a web phong shader that i want in glsl and Vulcan, we have a thin lens anda thick lens sensor and optics system, runnable with a thin lens standing in right now and runnable with... It may always use a thin lens somewhere in the process but i don't know. We can work in backward and forward or we can work with bdpt. I want all of those kernels attempted to be made in native, glsl, and Vulcan. They would enter this state library as being kernel pipelines, organized strings of work with discrete multi language components expressible as a tensor buffer”

User: “There is thick lens conic intercept zones it's not a studio thing unless it was fully integrated with the real original lenses not just it's thin lens contracts”

User: “See if you can automate uniform removal for translations”

User: “Oh, so, graphics runner you're right, i wanted to harvest frames from it as if it were a compute ofa frame but that might not make sense.”

User: “Images are just tensors, usually, what is the image handling you need for nodus? I'm sure you can reason out row tools you might want, the data can be byte span fifo, reference, or tensor”

User: “The render target should write to file using the exec compression row probably, i think, inn the user's job in the sql”

User: “File fifos are used for writing i think, but i might be wrong”

User: “Can you continue to finish the work”

## Next Steps

Extend the coupled raygen→T1→T2 path with T3 camera/sensor buffers from an
original scene, then adopt a graphics render-target tensor into the state library
for the actual Studio Phong raster pass. Continue only with verified source
and host resource contracts.
