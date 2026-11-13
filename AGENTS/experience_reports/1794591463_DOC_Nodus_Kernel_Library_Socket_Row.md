# Nodus kernel library owner and TCP byte stream row

**Date:** 2026-11-13 UTC

## Overview

The Nodus ray-kernel branch gained a process owner for its existing tensor
state library and GPU/pipeline registrations, plus a reusable TCP byte-stream
tool row. This responds to the requested launch-vector/contract ingress and
early output off ramps, but does not yet implement the optical job interpreter
or off-ramp taps.

## Steps Taken

- Checked the canonical Spectral optical-graph and Nodus handoff documents.
  The earlier private-FIFO optical tools were removed deliberately; real
  graph edges and the Nodus scheduler must carry work.
- Confirmed `register_gpu_shader_job` and `register_kernel_pipeline` are
  currently called only by tests, while Nodus already has a headless graph,
  tool registry and contract-socket control plane.
- Added `KernelLibrarySession` to keep the real state-library row alive while
  GPU and pipeline factories reference it. The joined wave trial passed on
  three Vulkan devices after adopting this owner.
- Added `TcpByteStreamTool`: a generic Argument/Return byte-tensor row using
  existing loopback Hello admission, bounded length-prefixed TCP frames and
  nonblocking backpressure. Exposed headless registration by tool ID.

## Observed Behaviour

The direct socket test sent and received exact byte tensors, including
restoring an unsent input before a peer connected. The headless runtime test
bound the registered row, stepped the graph, destroyed it and registered the
same ID again, all successfully. The wave schedule still verified 262,144
complex cells on each of three Vulkan devices through the new owner.

The socket row is a data-plane building block. It does not yet carry a
versioned optical job envelope, publish T1/T2 crossing records, drain full
T3/T4 path records, or expose pre/post integration products. The existing
`KernelPipelineTool` emits its summary only after its whole string; early
off ramps require individually scheduled stage rows and real graph taps.
See `C:/dev/nodus-thick-wave/docs/RAY_KERNEL_JOB_STREAM_BOUNDARY_2026-11-13.md`.

## Next Steps

Feed authored `RayIntent` launch vectors and an explicit job contract into
a graph-owned interpreter. Select stage rows from that contract and publish
typed taps as each stage completes. Give every TCP payload job/generation,
product kind, layout, record count and completion provenance. Preserve the
Spectral source's separate path, endpoint, linear and film semantics.

## Prompt History

User: “Okay i hope some work you did on getting output is of some use in your
actual job of making all ray tracing configurations runnable by the kernel
library”

User: “Jobs should feed launch vectors and contract and something needs to
interpret it into work flow and deliver back any of: crossing vectors, total
path records, pre integration, post integration, all offered off ramps, sent
out as soon as possible, offering tcp connections to stream in and out”

User: “A socket row would be an enduringly useful tool”
