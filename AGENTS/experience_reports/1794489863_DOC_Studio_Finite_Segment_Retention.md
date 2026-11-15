# Studio finite-segment response retention

2026-11-12. Work is isolated in Spectral worktree
`C:/Users/alber/spectral-response-retention-20261112`, branch
`codex/studio-response-retention-20261112`, based on
`917183f4c5e20ba2a2ffbce7add504061949c0cc`.
Retained commit: `4bf417d3483ea89bf7a3013b771540e9630ac908`. The owned
worktree is clean; this change was not merged, pushed or deployed.

The existing bounded response store now retains actual forward/reactive finite
segments crossing the requested ROI even when their endpoints are outside.
The existing payload and index ABI, native coordinates, evaluated zero versus
unknown, category scope, surface/native partitions and bounded eviction survive
unchanged. Exact IEEE-coordinate rational comparisons avoid losing narrow real
overlaps. No optical transport, core renderer source, Nodus, compiler, Phong or
default opt-in changed. No GPU link, dispatch or render was performed.

Verification: 84 focused CPU tests passed using the existing native CPU module
on process-local PYTHONPATH. Initial isolated run lacked that binary and gave
83 passed / 1 dependency failure; no code or package was rebuilt to resolve it.
Real store snapshots round-tripped through canonical Turing's existing decoder
with all 256 native words unchanged. The isolated finite-record index also
returned 64 cell references to those same two records, not 64 deposits.
Application readiness remains false. Full receipt and reproducible commands
are in `docs/STUDIO_FINITE_SEGMENT_RETENTION_2026-11-12.md` in that worktree.

Remaining limits: missing camera-pair scene codec (actual two 128-word buffers,
reserved but undefined 384-word artifact row), prebuilt reactive work not
dispatched, hardware execution held, and no completed Phong/energy application.
Do not pad or invent the missing camera ABI, replace native optics, or interpret
the retained packets as calibrated regional energy deposition.

## Prompt History

The owner: "you are then taking on the job of updating a new version of it".

Earlier user requirement: "We're looking to classify a grid and make statements
about energy deposited ina region under conditions we will use to texture,
animate, or otherwise state change based on a March the includes all non opaque
volumes and we want to retain just enough directional information to paint
speculars as texture over geometry".
