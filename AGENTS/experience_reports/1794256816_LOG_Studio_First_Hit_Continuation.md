# Studio first-hit continuation

**Date:** 2026-11-09

Implemented a bounded actual capture/resume seam in Spectral's composed sensor,
not a separate camera sampler. Exact launched rays are retained until a first
triangle query is accepted. Capture advances no sensor/history/pass state.
Original GL transport resumed with authoritative reference hits is bitwise
identical to uninterrupted rendering for the 256-ray fixture. Duplicate and
expired completions reject; reissue keeps the exact launch; explicit cancellation
permits ordinary server fallback. No live worker has been enabled.

## Commands

`python tests/studio_first_hit_loan_gpu_check.py` in spectral-analyzer passed.
`python -m py_compile camera_designer/studio_first_hit_loan.py demo_pluck_gl.py`
passed. The source files and scope are documented in
`spectral-analyzer/FIRST_HIT_CLIENT_CONTINUATION.md`.

## Prompt History

Parent assignment: "Implement substantive real launch/continuation boundary and ownership using existing ray code, work with triangle worker, no fabricated credit service; if access blocked recover environment via environment_management and report precise. Keep user's work safe."

User statement relayed by parent: "if I'm not mistaken my pipeline can already handle retaining old rays, we would just be telling them, hey it's retained because it's on loan, don't skip it".
