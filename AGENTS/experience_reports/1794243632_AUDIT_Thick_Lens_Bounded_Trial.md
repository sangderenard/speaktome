# Bounded thick-lens trial, 2026-11-09

Used existing canonical exposure_render_demo.py with no production source changes. Evidence lives in spectral-analyzer/exposures/dot_thick_lens_trial_20261109/trial_summary.json and neighboring logs.

The 16x16 stock native GPU BDPT/VCM invocation exited 1 after 28.67 seconds. Its recursive scheduler defaults refinement_steps to 8 (exposure_render_demo.py:6563), while run_recursive_sensor_epoch now requires exactly 1 (:2295). Supported SPECTRAL_SENSOR_STEPS_PER_LAYER=1 avoids this configuration error. Related focused tests pass because their calls explicitly request 1.

Two bounded retries with that runtime setting reached compound camera setup but the execution host then blocked network access to dc.services.visualstudio.com:443. These are execution-policy blocks, not measured BDPT timeouts. No new image or actual lens traversal/T5 completion counters were obtained. Do not treat old acceptance PNGs as current proof. No additional retry or alternate execution route was attempted after the repeated block. The known retry PID was confirmed absent; RTX 3060 was idle afterward.

39 tests passed in 19.81 seconds across test_native_bdpt_work_plan.py, test_compound_optics_transfer.py, and test_camera_software_optical_design.py. Full physical correctness, color, sensor energy, and wave coupling remain unvalidated by this run. Exact camera setup selected 9 surfaces, 8 spectral bands, native parametric conic intersection/Sellmeier dispersion, and VCM. Wave arenas were off. The existing Studio path was not replaced.

## Next Steps

Align default recursive refinement steps with its existing one-step lifecycle, then repeat a permitted bounded native trial and inspect actual sensor/lens/T5 evidence. Do not claim engine readiness from setup or focused tests alone. Evaluate color and wave-boundary energy independently.

## Prompt History

> If you could do a trial thick lens full pipeline and see if it works and would be a possible engine

> If you need to finish the energy transfer in and out for complex wave sim coupling i would really appreciate if you did that but it did work without the aperture, it has great lens distortion but the same trouble with color

## Resumed camera-focused trial

The user's clarified camera priority led to a permitted CPU path, avoiding the GPU route's access block without changing policy or network configuration. The installed legacy CPU CLI silently returned empty because bidirectional_packed is absent. The current ForwardCppLensBench async route traced but its auto T5 completion latch was lost. Calling the documented native run_bdpt_connection once after idle completed the real estimator.

Current 32-band, 9-entry compound lens, 108,898-triangle scene: 16x16 / 8 pupil samples totaled 9.7233 seconds; 64x64 / 8 pupil samples totaled 34.1568 seconds including 6.1121 seconds setup and 27.5479 seconds T5. Finite 64x64 output with 2,074 lit pixels. No upstream record loss. Connection diagnostic queue truncated 194,200,893 records; independent sensor accumulation still completed. Wave transport off. Magenta raw sensor remains uncalibrated.

Existing camera_software light-table projection also succeeded with the actual production lens:128x128, 30,000 rays, 3.3274 seconds. It is RGB geometric transport, not spectral physics; its existing development green-tints neutral patches through rough relative QE weights. Camera timing/substep/flash seams are genuine, while moving physics/BVH integration remains separate. 33 camera-output/chain/timing tests pass. Updated full evidence, images, raw arrays, scripts and summaries are in the original output directory. Production source unchanged.

Additional user prompts: "My primary concern is can we use a camera simulator not if the wave transitions are good programming we'll compile and optimize it later"; "That thick lens sim carries the equipment for substepping if you look around in it and flash timing which is really really useful"; "Look in camera software for clues"; "Do a report on the path forward for thick lens".

## Automatic lens contracts

Added only camera_software/lens_contracts.py, tests/test_lens_contracts.py and LENS_CONTRACTS_2026-11-09.md in spectral-analyzer. Cold export automatically discovers six YAML profiles, two registered surface prescriptions, and optionally solves eleven existing focus-lab presets. The 19-contract collection preserves original fields and source/payload hashes. It distinguishes authored approximations from actual cardinal/pupil data, checks nine near-axis exact rays, and tests the existing rear-group actuator without mutating the live scene.

The macro YAML authors 0.23 m minimum focus, not 2 cm. Studio's independent 20 mm UI/API and 10 mm CPU/GLSL guards were found but left unchanged. Focus uses metres and an explicit datum; zoom is focal length with unknown continuous domain retained; aperture is f-number with only supported source ranges. The exporter does not install or replace any shader. The current four-group solver meets requested EFL and sensor-focus tolerance for 80/1.4, midpoint 55-110 and 120Macro only. None of the eleven configured setups passed sampled 5-20 mm actuator requests. This is not a universal macro limitation. Existing registry conversion yields negative power for the doublet and eye, and eye near-axis comparison fails; preserve these facts rather than correcting the source without investigation.

New source and evidence: spectral-analyzer/exposures/dot_lens_contracts_20261109/lens_contracts.json and tests.log. Next: resolve profile distortion/band conventions, use the current shader's component adapter, validate continuous focus/zoom/aperture domains and source coordinate conventions before enabling a physical preset. No live restart, new GPU job, production camera mutation or current source overwrite occurred.

Additional prompt history:

> i was thinking having thin lens contracts that we write up to match what exists in that directory so closely that we actually automatically process that collection into contracs

> this is going to mean we need focus and zoom sliders, which means we really need aperture, I don't know of other settings that have a true analogue 1:1

## Studio controls staged, not published

Implemented source-derived analytical profile controls in Turing, keeping the current composed shader. Six authored profiles are selectable approximations and thirteen compound states remain reference-only. Units, source hashes, source minima, renderer pupil/FOV bounds and uncalibrated effects remain explicit. Custom focus now has a consistent 1 mm numerical domain. Corrected the analytical shader's former spherical focus construction to axial planar focus; preserved tilt/horizon/pupil/transport behavior and versioned the job cache. Separate manifold bake remains unchanged and is not enabled.

Measured 97 Turing backend tests, nine Node suites, 231 Spectral focused tests and 540 actual RTX 3060 shader-camera samples passing. Isolated mobile Chrome QA and real 72,960-triangle captured payload/backend decoding passed without a full-scene GPU photo. Independent review found no remaining source blocker. Partial invalid live edits retain the accepted worker rather than cancel it; lens settings freeze before snapshot hashing; active live camera provenance is published per snapshot revision.

Five static assets are staged under turing/build/dot-lens-controls-20261109/release, from the currently published bootstrap, not unrelated source-only changes. Parent paused publication and restart pending confirmation that the user can connect. Recovery restart loaded initial backend changes; final validation corrections need an idle final refresh. See turing/examples/meowdoku/STUDIO_LENS_CONTROLS_2026-11-09.md for exact scope, tests, source paths and release steps. No user GPU job was interrupted by this work.


## 2026-11-09T21:42Z: controls live and macro mismatch repaired

The combined lens/color release and backend validation are now live. The user's
actual blurry e8806838 exposure has230 mm selected focus and23.126 mm target
distance, using macro_50mm. The physical source bound remains intact. Published
an explicit custom target-focus action, mismatch warning, selected-lens frame
and focus, and removed independent21 mm framing and scale-based dolly floors.
Five static HTTPS assets verified,224 other assets unchanged, no restart or
photograph interruption. Eleven focused Node suites and exact published handlers
pass. New browser QA blocked before completion; no new full-scene GPU quality
signoff. See Turing's current lens-controls handoff and macro-focus hazard.

## 2026-11-09: original native BDPT without camera software

Investigated original modern RayTracer submit/sweep/T5 entries rather than the absent historical bidirectional_packed wrapper. A virtual film, native materials and native T5 render without compound lens, sensor-camera group, CameraSoftware or T4. The GPU-resident six-triangle 64x64 trial completed in 1.344 s, including 68.8 ms T5, with all pixels lit and zero overflow. This is not full-Studio performance or physical calibration.

Found and minimally repaired a double seq increment in native emitter sampling, which drops odd bands for even band counts. Actual current-binary regression failed 4/8/32 bands; isolated rebuilt existing CMake target passes those plus 1/3 and the work-plan tests (19 total). Live binary unchanged. The current sensor sweep also under-covers wavelengths if pupils < bands; no seed rotation fills the omitted lanes. The source fix was independently reviewed.

Important open mismatch: CPU post-scatter beta vs GPU pre-scatter beta, plus host first-camera SENSOR flags vs resident PDF scatter flags. Host-packed all-GPU failed receiver-reflectance scaling; resident behavior is different. Do not treat CPU agreement alone as physical proof or swap pipelines silently. Native piecewise display color mapping is separate from Studio CIE/D65.

Detailed report: spectral-analyzer/NATIVE_BDPT_COMPONENT_TRIAL_2026-11-09.md. Reproduction, raw arrays, logs and patch: spectral-analyzer/exposures/dot_bdpt_transport_20261109. Studio material composition is owned by the material worker; both legacy and region paths already use MaterialDatabase, not independent databases.


## 2026-11-10 native emissive surface integration

The assigned request was to remove Studio centre-point lighting and reuse the original engine surface sampler while preserving the composed fast shader. The Studio adapter now invokes native emissive-triangle roots using the shared MatBuf and exact authored geometry. Per-triangle submissions avoid the original multi-triangle UV-domain area bug; complete spectral cycles and a runtime integrated-energy guard protect the contract. The known one-line native spectral correction was promoted with verified rollback copy and no server restart. 54 focused CPU tests passed against the production module; real GPU area integration, sample-budget invariance, distance falloff and one-sided emission passed. Independent review caught and resolved emitting-boundary ordering and skinny-triangle precision. The unsupported voxel sum is excluded from the new area estimator to avoid double counting; full caustic merging remains unvalidated. See spectral-analyzer/NATIVE_EMISSIVE_SOURCES_2026-11-10.md and build/dot-native-area-20261110/release.json.
