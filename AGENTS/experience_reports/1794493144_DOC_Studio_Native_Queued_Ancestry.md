# Studio native queued ancestry and bounded consumption

The existing prebuilt renderer now has an explicitly opted-in extension of its
actual PendingRay and owner lifecycle. Original first64 bytes and optical/RNG
expressions are preserved. Two appended uvec4s copy authentic dispatch, parent
ray, selected baked root, bounce, actual replay weight and enqueue cause.
The ordinary Pluck application renderer and canonical shader literals are
unchanged; no compiler, Nodus or stage splitter was changed.

Retained isolated commits:

- Spectral `2d07278861e40e59174461e635fdeb0803b278a8`, branch
  `codex/studio-response-retention-20261112`, worktree
  `C:/Users/alber/spectral-response-retention-20261112`.
- Turing `2103aa04077615d37b5ca584ff8d007d9b453dd5`, after
  `0c41c33a5e763b896b13b36dd2d60f139422185d`, branch
  `codex/camera-pair-receiver-20261112`, worktree
  `C:/Users/alber/turing-camera-pair-receiver-20261112`.

Both worktrees were clean after committing. Canonical concurrent code remains
untouched. This guestbook receipt and the updated next-step stub are separate
coordination files, not a merge or deployment.

The same prebuilt owner drains bounded chunks after each forward chunk, before
source/scene mutation, root completion and scalar publication. Native queue
offset and logical seed index are retained. Children inherit the parent's
recording weight and are not counted as new roots. Any uncertain submission,
readback, consumer, reset or restore poisons reuse; no additive retry.

Versioned forward-v2/scene-v3 delivery uses existing storage/publication/lease/
decoder mechanisms and immutable parent context. Default paths stay unchanged.
No v1 records are silently relabeled. Camera pairs retain their existing ABI.
Queue capacity uses the source-proven three-edge-plus-one-fluorescence bound;
measured aggregate memory admission precedes the larger queue allocation.
Validation scratch is bounded/reserved and zero-child jobs remain valid.

Verification:208 Spectral CPU tests passed, including16 offline shader-validator
variants;27 Turing request/lease tests passed, plus existing decoder/lifecycle
and18 index cases. Actual-store fixture roundtrips preserve256 ancestry words
and512 camera words exactly. Review found and closed allocation-admission,
zero-capacity and scratch-budget gaps. An unrelated existing mock readback bug
was reproduced in unchanged canonical source and corrected only in the owned
test fixture to honor offset/size; no production accumulation law changed.

No GPU link, execution, render, live job, push or deployment occurred. The
worker's evaluated-response execution hold remains. CPU syntax/ownership and
serialization are not hardware validation or proof of complete path/volume
coverage, deposited energy, image quality or Phong readiness.

Detailed receipts are in the Spectral docs:
`STUDIO_NATIVE_QUEUE_ANCESTRY_OWNER_2026-11-12.md` and
`STUDIO_FORWARD_ANCESTRY_TRANSPORT_V3.md`.

## Prompt History

The owner directed: "you are then taking on the job of updating a new version of it".
The owner requested: "whenever you get to where you have a render, show me so we can debug wild accumulation or rendering".

## Next Steps

Integrate/review against concurrent canonical changes before considering any
hardware validation. Notify the owner before GPU linking. Keep rendering held
until complete data is established, and show any eventual actual render.
