# Nodus SSBO FIFO boundary correction

Persistent Vulkan memory, backend-resident `EdgeTensorFifo` storage, native D2D
transfer, graph-definition storage communication, and direct two-shader packet
execution are green. `EdgeTensorFifo` plus `GP_MEM_BACKEND_VULKAN` is the SSBO
FIFO. Do not invent a new GPU queue, reader, reservation API, or execution
contract.

## Required continuation

1. Use `push_from_frame` / `pop_to_frame`; they own publish, consume, sequence,
   bounded-full behavior, and backend-to-backend movement.
2. Require Vulkan-to-Vulkan movement to use the native D2D callback and fail
   intact if it fails. Never stage an SSBO FIFO payload through host memory.
3. Keep `AbstractTensorHandle` token transfer and `RawStackFrame` execution as
   explicit alternatives. Vulkan storage does not make every graph edge FIFO.
4. Retain the two-shader numeric proof, FIFO graph-property readback,
   persistent-backend lifecycle/barrier tests, and canonical Turing SPIR-V
   crossing.

Read first:

- `nodus/docs/SESSION_CONTINUATION_2026-10-23.md`
- `nodus/docs/TURING_NODUS_GRAPH_DEFINITION.md`
- `nodus/tests/vulkan_ssbo_fifo_test.cpp`
- `nodus/tools/materialize_turing_spirv_fifo_demo.py`
- `nodus/out/turing-spirv-ssbo-proof/graph-definition.json`
- `nodus/out/turing-spirv-ssbo-proof/turing-spirv-ssbo-fifo-web.png`
