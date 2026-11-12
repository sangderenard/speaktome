# Complete the canonical GPU inventory production binding

Source and CPU tests exist in Nodus `gpu_inventory_rows.h/.cpp` and
`gpu_inventory_rows_test.cpp`. Continue through the existing registry,
GraphRuntime, ContractNegotiator and native typed/named ports; do not add a
parallel scheduler or GPU-specific persistence format.

- Bootstrap UUID factories in the same native image before canvas load.
- Persist and restore canonical typed-stack/Sequential execution metadata and
  verify transport type identities, rather than treating saved numeric IDs as
  portable.
- Bind shared automation, monitoring and immutable source/commit adapters.
- Keep harness occupancy separate from unknown hardware utilization.
- Review persistent process-local buffer/context lifetime before admitting full
  shader execution. Link-only receipts do not carry live GL handles or prove
  producer-to-screen correctness.

Production installation and hardware execution require their own approved
validation; this stub is not authorization to deploy, link or render.

Source: `AGENTS/experience_reports/1794434260_DOC_Nodus_GPU_Inventory_Registration.md`.
