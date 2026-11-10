# Studio Auto bounce pilot

2026-11-09 AUTO BOUNCE PILOT (SOURCE ONLY; GPU/PUBLICATION PENDING)

Auto is the new UI default; the existing numeric manual count and quadratic
1..200 slider remain separate and survive Auto/manual switches. New requests
carry bounce_mode. Legacy requests with an explicit max_bounces stay manual.
The probe horizon is 1000 interactions, independently of the render cap 200.

The isolated Spectral module instruments the canonical forward shader; it does
not implement replacement intersection/material physics. It emits four
independent isotropic unit-energy samples per 4x4x4 spatial cell (256 paths),
records every observed hit's transport energy through depth1000, and records
paths still carrying energy after1000 separately. Diagnostic sources/buffers
are owned; canonical scene SSBOs are read-only; splats are disabled and source
bindings restored. The render field is cleared after its initial warmup before
using the chosen limit. No render path gains a new termination rule.

Selection uses the entire observed remaining energy tail, with at least64
further depths available. It requires all four batches and their t-based batch
spread to be below1%, at least16 first hits per batch, and no survivor after1000.
The minimum automatic choice is8; a required choice above200 is reported and
rendering stays200. Sparse coverage, missing/failed pilot and survivors at1000
are explicitly inconclusive. Persistent paths trigger a mirror-room warning.
This is forward transport-energy guidance, not a bound on camera-image error.
Small objects/rare caustics can be missed; existing forward cutoff.002 and the
different sensor roulette/throughput cutoffs remain and are disclosed.

Measured CPU/static checks:9 Spectral probe tests pass, including late returns,
closed-tail samples, one rare survivor, empty coverage, spatial stratification,
canonical instrumentation and host binding restoration. glslangValidator
accepts the generated compute shader.13 Turing Python tests pass for Auto
admission, photo API, room packing and gobos. Auto/manual JS checks pass;32
photo lifecycle/retouch cases pass; gobo geometry/scene, camera framing, room
and material-ID checks pass. Python/JS syntax and scoped whitespace pass.
The live lifecycle currently awaits the concurrent2GiB history worker's test
updates; its old128MiB assertion is obsolete, not an Auto failure.

GPU QA is not yet executed: the user's production live then180-second photo
jobs remained active throughout implementation. No job was interrupted and no
server restarted or static export published. The opt-in tiny12-triangle GPU
check is tests/marble_bounce_probe_gpu_check.py in Spectral. It validates diffuse
selection, slow opaque mirrors beyond200, glass total-internal-reflection
survivors at1000, unchanged field bytes and unchanged production depth uniform.
Run only after verifying all production GPU photo/live work is idle.

## Prompt History

User: “also let's by default have bounce on auto and we'll do a sparse scatter through the room on a limit of like 1000 and find like, when do they become insignificant in a huge window over only a few samples, that's where you probably want it, up to 200”

User: “decouple the max, i specitied 1000 for the probe because it's a fact it can present the user to alarm them of mirror room effects”

## Next Steps

Complete the opt-in isolated GPU verification only when the production GPU is idle, then coordinate publication and final combined tests with the active integration owner.
