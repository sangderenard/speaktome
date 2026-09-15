# Finish the Python viewer / compiled coupled simulation

Continue from `turing/docs/CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md`.
The eager simulation extraction passes; the first eight-lane native build
failed on cyclic atomic regions before SSA emission. Repair the responsible
compiler mechanism, as explicitly requested by the user, rather than editing
intermediate graph ordering. The parallel coordinator-path fusion regression
is repaired and reproduced the exact failure in `balloon_tire_vector_step`;
the unchanged function graph passes atomic ordering with the repair.
The next fresh build exposed typed tuple-call projections incorrectly
dispatched outside their loop. Classification and serialized-cache versioning
are repaired; the exact saved specialized function passes control replanning.
The fresh v3 build includes both repairs and retains late-stage checkpoints.
Complete a fresh native build, run the consecutive-window physical/controller
comparison, and verify the existing Python viewer launch before calling the
mostly native validator ready. Preserve the user's patience for long builds.
