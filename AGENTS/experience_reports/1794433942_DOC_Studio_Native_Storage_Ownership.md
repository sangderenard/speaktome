# Studio native storage ownership

2026-11-11 UTC. Added native OpenGL buffer ownership to Nodus's existing memory
backend registry/vtable, not another scheduler or renderer. The actual source
and verification record is `nodus/docs/NATIVE_STUDIO_STORAGE_2026-11-11.md`.

`mem_backend_opengl.h/.cpp` provides registered-current-context/manager-lane
allocation, upload/readback, native copying, precise view clears and checked
release. Legacy OpenGL parameter blobs previously produced an unusable generic
backend without allocation/free hooks; that route now refuses. Existing CPU
callers need no GL hooks and remain regression-tested. Failed deletion retains
the canonical registry handle; failed allocation retains an undeleted name with
bounded cleanup ownership. A failed bind cannot mutate the previous binding.

`studio_auxiliary_storage.h/.cpp` reserves the actual binding-23 visit, scalar,
terminal-copy and optional v1/v2 continuation views. Absent explicit opt-in
allocates nothing. The actual package layout functions validate proposed and
then allocated extents. The five-stage v2 tail needs 1024 bytes per original
batch slot; no extra SSBO or optical computation is introduced.

The standalone native test builds the actual edited registry code and passed
CPU-injected allocation, failure, ownership, context/lane, layout and adjacent
view regressions. Fifty-six focused Spectral tests and three Node response
receiver/application/field suites passed. The broader scene-builder test failed
its captured scene hash; its cause/baseline status was not established here.

The running service, frozen protected linker binaries and live DLLs were not
changed. No real graphics context, hardware query, shader link, dispatch,
rendering or service reload was performed. The protected child still exits
after linking and cannot retain these resources. Persistent-child integration,
actual scene upload, all-family scheduling and numerical admission remain open.
The evaluated-response worker hold and unverified-continuation refusal remain.

## Prompt History

The user requested:

> we want this development work in native source, bespoke, because it's going to be a vital kernel in nodus

> an invariant was that that additional data is opt in only

> also you need to alpha to the phong in sparsity of data. an empty region is not black, it's no information

These requirements motivated native ownership, explicit opt-in, unchanged
response meanings and fail-closed availability rather than renderer substitution.

## Next Steps

Continue the existing protected-child owner with a process-local persistent
context/resource lifecycle and opaque cross-process descriptors; never treat a
link receipt or expired GLuint as executable ownership. The canonical typed
Control/Telemetry rows and ThreadManager remain the only scheduler. See the
corresponding `todo/1794433942_studio_native_storage.stub.md`.
