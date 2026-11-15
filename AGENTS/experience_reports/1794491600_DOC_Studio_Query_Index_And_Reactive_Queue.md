# Studio query indexing and reactive queue ownership

Continued the existing light-field response pipeline in owned worktrees only.
The bounded index now accepts explicit camera query-parametric support, not
deposition, surface evaluation or coverage. Native origin/direction/distance are
retained; the derived endpoint is not confused with the first-hit point. The
actual producer and decoder preserve all native bits and unknown versus zero.

Turing branch `codex/camera-pair-receiver-20261112`, worktree
`C:/Users/alber/turing-camera-pair-receiver-20261112`, is clean at
`2d605524046d340f17c372ea4550d237f1541315`, after the camera extension
`72f914195a58cf57d75d1e08310fa0ef84387b28` and original bounded index
reused as `b99d7f9d80a88e4f5e0474c1eaed378da68e002b`. The extension defaults off,
requires v2 and a positive memory budget, and uses support bit 4 separately from
existing copied endpoint/segment support. The receiver is not application-ready.

Four producer/decoder CPU fixtures passed from the owned Spectral worktree:
two exact camera pairs with and without query indexing (512 words), forward/
reactive copied segments through the updated receiver, and the original index.
Each indexed fixture gives 64 bounded references. The 18 index cases and
three other Node checks passed in the receiver worktree. Review caught a real
extreme-coordinate case where entry/exit clipped parameters collapse to equal
float64 values; the final correction classifies this conservatively unavailable,
without tolerance or guessed cells, with both camera and copied-segment tests.
No GPU compilation,
link, dispatch, render, live job, push or deployment ran.

The current prebuilt reactive queue cannot simply be enabled. Its existing
optical pass is real and reused by the ordinary renderer, but the prebuilt
specialization removes that entry; the queue index is not batch-offset aware;
the queue can exceed one response batch; and there is no drain lifecycle across
scene/source changes. The 64-byte queued ray retains no source root/population/
replay ancestry. It receives both edge diffraction and fluorescent/re-emission
rays, which cannot be distinguished from the shared flag or guessed packet
values. The spawning triangle is not a source-root identity.

These are missing execution/provenance contracts, not permission denials and
not justification for a separate scheduler or fabricated normalization. No
reactive production code was changed in this pass. Detailed source evidence
and the supported next contract work are retained in the owned Spectral branch,
`docs/STUDIO_REACTIVE_QUEUE_OWNER_GAPS_2026-11-12.md`. Canonical concurrent dirty
work was left untouched. Hardware remains held until complete data and the
required advance notice; any eventual actual render is to be shown to the owner.

## Prompt History

The owner directed: "you are then taking on the job of updating a new version of it".
The owner later requested: "whenever you get to where you have a render, show me so we can debug wild accumulation or rendering".

## Next Steps

Resolve the genuine existing-queue ancestry, bounded dispatch and lifecycle
contract before integrating its native reactive execution. Keep query geometry
separate from energy or evaluated coverage. See the companion todo stub.
