# Studio T5 REST dispatch

Continue `1794459924_DOC_Studio_Atlas_REST_Baker_And_T5_Scene_Boundary.md`.
The exact `MaterialDatabase`/geometry handoff is in
`spectral-analyzer/camera_designer/studio_t5_scene.py`. Connect the native
`CppExposureBackend` T5 latch, a physically located sensor group, bounded
per-sweep work controls, and the existing Studio HDR/status/ownership route.
An `integrator=t5` label without a real T5 latch must fail closed. Preserve
the Studio model shutter independently from renderer work duration; expose
sample counts only as sampling controls pending calibration. Verify first with
mocked stage and server contract tests; respect `turing/AGENT_OBSTINANCE.md`
before any GPU image run.
