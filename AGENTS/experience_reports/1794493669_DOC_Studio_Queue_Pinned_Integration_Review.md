# Studio queue pinned integration review

Merged only committed canonical changes into owned branches, never canonical
files or the live service. Spectral canonical 009178d merged as 9a7039e; receipt
f6c9c5ad4c456d752868c7b11f2f6f6a8663c64f. Turing canonical d030e18d merged as
a4102c19; receipt 50f5b2c4dc8f8de38c701dec21e15a7b1136ec8d. Both owned worktrees
are clean. Canonical dirty Stage Programmer/UI/lighting/export/test work was
reviewed read-only and remains outside this integration.

Verification: 229 Spectral CPU tests passed (including 16 offline shader variants
and trigger/preview regressions) plus the original software trigger-grid check.
Turing 54 backend tests, 18 index cases, receiver/lifecycle/gem checks and actual
store roundtrips passed. Its committed retouch mock lacks classList.toggle;
the pending canonical dirty test adapts that interface, but was not copied.
The optical renderer was not run. CPU tests emitted only solver/grid diagnostics.

Remaining concrete coverage issues: corrected opaque PBR continues/breaks before
native edge-diffraction enqueues, so those hits produce no diffraction children
(existing behavior, not queue regression). Fluorescence remains outside that
branch. ExtinctionGrid plus material/region Beer observation is explicitly
unsupported until joint partition integration exists. Forward/reactive escape
rows lack finite spans; camera miss parameter word 11 is NaN, not zero. Invalid
medium and absent/evicted/untraced data remain unknown. Queue completion means
generated native children were consumed, not all path types were generated.

Whole source assembled through existing CPU/native composers: 118053 bytes,
SHA256 32f1f492ff61aefd18fbbc8ebdd078cdc43c83374799575df68a375b8796734b,
16 declared storage blocks and 24 binding slots required. This is not the old
bespoke-splitter artifact. Device support/link success remains unverified.
The native prebuilt constructor dispatches 128 warmup rays even when initial_rays
is zero, so it must not be used as a supposed link-only probe. Later bounded
compute testing must include an actually nonempty secondary queue and separate
camera-pair readback; a zero-child test proves too little.

No GPU link/compute, optical render, live request, push or deployment. Hardware
and rendering holds remain. Full receipt is
docs/STUDIO_QUEUE_INTEGRATION_REVIEW_2026-11-12.md in the owned Spectral branch.

## Prompt History

The owner directed: "you are then taking on the job of updating a new version of it".
The owner requested: "whenever you get to where you have a render, show me so we can debug wild accumulation or rendering".

## Next Steps

Resolve the corrected opaque-surface diffraction optical/order/accounting choice
without reconstructing transport. Keep other missing data explicitly unknown.
Announce any hardware activity before execution and retain the complete-data
constraint before producing/showing an actual optical render.
