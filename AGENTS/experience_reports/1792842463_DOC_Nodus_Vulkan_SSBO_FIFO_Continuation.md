# Nodus Vulkan SSBO FIFO continuation

**Date:** 1792842463 (2026-10-24)
**Title:** Persistent Vulkan edge storage and communicable graph properties

## Overview

This continuation followed the corrected boundary: Turing lowers computation
through its sanctioned SSA compiler and publishes deployable row products;
Nodus owns and communicates its graph. It did not send Turing ProcessGraph to
Nodus or invent a parallel GPU graph model.

The work established a concrete persistent Vulkan memory backend in Nodus,
made table `EdgeTensorFifo` storage truly rebindable to it, exposed the storage
choice as an additive `nodus.graph-definition.v1` FIFO property, and proved a
device-local packet crossing between two shader dispatches. Stack transport
remains available and independent of whether an edge has FIFO storage.

## Steps Taken

- Added the Vulkan implementation of the existing memory-backend vtable:
  persistent instance/device/queue/command-pool ownership, buffer allocation,
  map/unmap, resize, native handles, and native device-to-device copies.
- Corrected generic cross-backend transfer so a CPU vtable is never invoked
  with a Vulkan destination handle; unlike-backend transfer uses established
  map/copy staging hooks.
- Added Vulkan dispatch against existing backend buffer bindings and retained
  the compatibility host-data dispatch entry.
- Added explicit compute/transfer/host visibility barriers and synchronous
  fence completion at the current row boundary.
- Changed `EdgeTensorFifo::rebind_backend` and the Table ABI so changing an
  edge backend reallocates/migrates the actual ring storage.
- Added Table readback of the effective storage backend.
- Extended `FifoConfig`, mapping validation, Python/native ABI adoption, and
  native definition readback with `storage: host|vulkan`. Older mappings omit
  it and therefore remain host-backed.
- Made each adopted headless graph share one Vulkan backend context for its
  Vulkan FIFO edges.
- Added `vulkan_ssbo_fifo_test.cpp`: shader A writes a 256-byte packet, the
  packet crosses an `EdgeTensorFifo` through Vulkan native D2D, shader B reads
  it, and the numerical result is checked. The same test asserts that a public
  Table edge's storage handle becomes Vulkan-backed after rebinding.
- Added a Python native-readback regression for `storage="vulkan"`.
- Added `tools/materialize_turing_spirv_fifo_demo.py`, which requests a real
  `turing.spirv-product.v1`, adopts the generated row twice, installs an edge
  with Vulkan FIFO storage, requires exact definition/native-readback
  concordance, and saves the native canvas and definition.
- Served that canvas through `demo_canvas_tables.py --mode server`, captured
  `/frame`, and visually inspected the PNG. It contains both complete row
  tables and the cable; it is not the earlier empty-shell proof.

## Observed Behaviour

- `python -m pytest tests/nodus_python_graph_test.py -q`: **15 passed**.
- Focused CTest set (`edge_backend_coordination_test`,
  `edge_fifo_drop_contract_test`, `vulkan_dispatch_test`,
  `vulkan_ssbo_fifo_test`): **4/4 passed**.
- `python tests/turing_nodus_spirv_crossing_test.py`: **OK** against canonical
  Turing, including real plugin generation and Vulkan execution.
- `git diff --check`: passed; only existing Windows line-ending notices.
- Retained visual proof:
  `nodus/out/turing-spirv-ssbo-proof/turing-spirv-ssbo-fifo-web.png`.
- Retained definition/canvas:
  `nodus/out/turing-spirv-ssbo-proof/graph-definition.json` and
  `turing-spirv-ssbo-fifo.canvas`.

## Lessons Learned

The old declarations were meaningful evidence: `GP_MEM_BACKEND_VULKAN`,
native-D2D capability, `EdgeTensorFifo`, and the Table backend attachment were
the intended axis. The missing part was implementation and wiring, not a new
GPU-edge abstraction.

The direct two-shader test and the visual ordinary-row graph prove different
layers. The former proves a real 256-byte payload remains device-local through
the existing FIFO. The latter proves that Turing can fully materialize the
Nodus rows, edge, and `storage="vulkan"` property. The ordinary generated row
wrapper's `AbstractTensorHandle`/`RawStackFrame` route is an intentional,
independent execution option, not evidence that the FIFO needs another reader
or reservation protocol.

## Next Steps

1. Preserve `EdgeTensorFifo` as the sole sequencing authority and use its
   existing `push_from_frame` / `pop_to_frame` operations for SSBO traffic.
2. Preserve explicit handle-token and `RawStackFrame` paths as alternatives;
   backend selection is not permission to make every edge a FIFO.
3. Keep Vulkan-to-Vulkan FIFO transfer fail-closed: native D2D failure must not
   silently fall through to host staging.
4. Retain the persistent-backend lifecycle, barriers, storage readback,
   existing SPIR-V crossing, C/LLVM dynamic-shape, and validation-77 gates.

The earlier direct-slot continuation stub has been corrected to record this
boundary.

## Prompt History

> "did you cut out problems by allowing more things to be fifo edges? that's not wrong persay, but the stack should be an option"

> "can you show me during this an image from the web gui nodus instance of a turing created nodus graph materializes as specified in nodus"

> "i don't know exactly how to explain this but if you don't look at the image i can't see it"

> "if you'll notice with me, that has no rows or edges"

> "just like there's an eigan way to handle abstract tensor there's supposed to be like sso or something, some indication we're going to let rows and edges be shaders and on gpu buffers"

> "past evidence would be in the fifo types available if anywhere"

> "i'm piling on try to keep it all as queued tasks, the goal in this additional shader work is to be able to make vulcan shaders that communicate and are arranged in a graph that use ssbo for fifo"

> "Aim your ambition for about 40 more minutes max at which time you'd need to start thorough reporting anda continuation of necessary"
