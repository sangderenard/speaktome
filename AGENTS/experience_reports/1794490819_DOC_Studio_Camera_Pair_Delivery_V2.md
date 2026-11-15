# Studio camera-pair delivery v2

Continued the existing response pipeline without creating a separate transport.
The actual observer has two aligned 128-word rows. The old unused family 3
reserved 384 words with no defined third row; it is not a valid reason to pad
or invent data. A distinct v2 envelope/codec now carries exactly 256 words.

Retained, isolated commits:

- Spectral `c68d27deaf29a6b06058ec97ffce455f852a0a6b`, branch
  `codex/studio-response-retention-20261112`, worktree
  `C:/Users/alber/spectral-response-retention-20261112`.
- Turing `c71d892b59483e6ca7efb85069e6b6ec9bdcfe0c`, branch
  `codex/camera-pair-receiver-20261112`, worktree
  `C:/Users/alber/turing-camera-pair-receiver-20261112`.

The explicit additional camera-record mode selects the existing raw interval
capture instead of completed-sample capture. Default completed samples and v1
remain unchanged. Raw pairs are native-only/unlocated, not guessed raster
recipients or Phong color. Actual geometry, direction, medium, region and RHS
payload bits remain intact. Store/publication bounds, ownership checks, API
opt-in, leases and decoder validation all apply. Hardware execution is still
held before GL creation. No Nodus, compiler, optical law or shader change.

Verification: 116 focused Spectral CPU tests plus the existing fake-GL lifecycle
test; 18 new Turing Python tests plus 21 existing response regressions and three
Node checks. Real store pairs round-tripped as all 512 words unchanged for two
records. Tests covered the actual callback, store, publisher, authenticated
leased-file reader and decoder. Existing v1 and separate record-index fixtures
also passed. Existing native CPU dependency/path overrides were reused; no
package was installed. Turing test startup built its existing CPU CFFI helper.
No shader/GPU compilation, GPU link, dispatch, render or live request ran.

Full receipts: `docs/STUDIO_CAMERA_PAIR_DELIVERY_V2_2026-11-12.md` in Spectral
and `docs/STUDIO_CAMERA_PAIR_RECEIVER_V2.md` in Turing. Both worktrees were clean
after their commits. No canonical edits, merge, push or deployment; concurrent
canonical work was left untouched. This guestbook entry is a separate untracked
coordination receipt.

Remaining limits: raw interval spatial indexing/application is not implemented,
the prebuilt reactive queue is not dispatched, all-volume/all-path energy
coverage is not established, and the hardware hold remains. Presence is not
coverage; absent/evicted/invalid/untraced data are not black.

## Prompt History

The owner directed: "you are then taking on the job of updating a new version of it".
The continuing goal is regional classification and energy statements from
actual paths through non-opaque media, retaining direction for later specular
use without reconstructing transport in Phong.
