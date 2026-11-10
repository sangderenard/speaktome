# Marble fixed-three-wavelength sensor separation

Date: 2026-11-09

## Scope

The active Marble Studio photo worker in the independent spectral-analyzer repository now opts into a fixed 650/550/450 nm transport basis. Material operations use that basis, and custom film response is integrated once after camera-path accumulation. This is a bounded correction to the existing GPU renderer, not a new renderer or a claim of calibrated colorimetry.

Changes belong to demo_pluck_gl.py, camera_designer/spectral_sensor_basis.py, the optional spectral_emission_rgb argument in camera_designer/scene_builder.py, and tests/test_studio_spectral_sensor_basis.py. Coordinated film-control changes in camera_designer/film_response.py and marble_preview_worker.py supply the opt-in tuple and mesh keywords. Existing unrelated working-tree changes were preserved. No commit, deployment, server restart, source-budget change, or active exposure cancellation was performed for this correction.

## Findings and correction

The fixed3 FilmLayer preset has white light_rgb for every layer. Camera material reflectance and transmission previously projected authored RGB against those display colors, giving the blue material [0.10, 0.43, 0.91] the same 0.48 weight in every channel. Camera throughput also lived in film-layer space, which is not a valid wavelength basis after overlapping filters mix the bands.

The opted-in forward field now stores fixed-line energy. Camera material response reuses the existing forward spectral RGB approximation. Emission uses the existing additive q=12 Gaussian line convention from RayOrder, with authored emission amplitudes preserved before display conversion. Film centers and sigmas affect final measurement, not the material. Non-opted-in behavior is retained. Aggregate source weighting is intentionally unchanged pending the user's desired aggregate-light contract.

## Validation

- 56 focused tests passed: spectral sensor basis, saved Studio GPU scene, camera sensor output, marble film response, and worker status.
- Both complete forward and camera GLSL programs compiled and linked on the NVIDIA GeForce RTX 3060.
- Actual worker renders of the same saved scene completed at 128x128, 64 epochs, four bounces, and 524,288 forward jobs each. Custom centers [630,540,470] nm and sigmas [.24,.21,.18] were compared with defaults [650,550,450] nm and [.06,.06,.06]. The custom raw image matched M_custom × inverse(M_default) × default_raw with relative L1 error 1.7402547e-7.
- A GPU evaluation of the shared blue-material spectral function returned [.138815,.415072,.885181], rather than identical channels. Pure emitter basis responses matched their bands with the existing Gaussian tails.
- Independent read-only code review found no blocker in the active Marble path. The historical PASS_SENSOR route is outside this result and must not be presented as validated.

## Known boundaries

This does not correct nested-media tracking, the opacity-to-reflection mixture, Beer attenuation's segment ownership, the forward transmission-weight estimator, or physical color calibration. The older saved side-light exposure retained milk opacity .79 and blue opacity .90; its dark interior regions cannot be uniquely diagnosed from its image. Running exposures keep code already loaded; later exposures load the corrected modules.

## Follow-up: reference output color profile

Added camera_designer/marble_color_profile.py and its focused tests. It uses verified CIE 1931 samples for the three transport wavelengths, conditioned film-response inversion, a fixed equal-power three-line white adapted to D65, and the existing ColorScienceProfile interface. Tone mapping and one sRGB transfer follow color conversion. New HDR files preserve raw sensor arrays and separately store float64 profiled linear sRGB plus profile metadata. Old HDR files retain legacy retouch behavior. Live/saved retouch recipe parity and serialization are tested; the shared direct-sensor test-pattern CLI is CPU-only.

Final checks passed: 67 focused Spectral tests and 23 Turing photo/film tests. Reprocessing the actual default/custom-film captures from the earlier GPU test produced profiled colors matching to relative L1 9.72e-7. Independent math review cleared the bounded result after float64 persistence and fallback wording corrections. The profile is a three-line colorimetric baseline, not measured-camera calibration; display exposure remains adaptive. No further GPU job, restart, deployment or exposure cancellation was performed.

## Prompt History

> did you get any color profile material done? some kind of baseline that's perceptual on the linear sensor site to output path

> feed intended values like the color bar was the sensor, like direct transfer, and then output the color panel

> I believe that's intentional to balance many lights to be one agregate

> our color calibration is right for single light source color but obviously it's not resolving to perceptual


## Follow-on: display controls, 2026-11-09

User instruction: "I think that post processing window should have the color conversion profile for the sensor and or any nonlinearity or normalization".

Implemented bounded auto99/manual normalization and exponential/linear luminance mapping in the existing profiled output. Manual white is a relative luminance reference, not physical calibration. The immutable sensor profile, raw HDR, and one sRGB transfer remain separate. UI-facing display metadata describes the selected recipe. Legacy HDR retains its prior transform and explicitly reports unsupported new controls.

Focused verification: Spectral74 tests and Turing49 tests passed. Live/saved recipe equality covers all mode combinations; persisted HDR bytes remain unchanged after retouch. No GPU run or restart. The coordinated UI worker owns browser validation and publication.

## Follow-on: artistic display controls, 2026-11-09

Parent assignment: "Please coordinate with controls worker on user's new postprocessing vibrance, saturation, black point request. Implement scientifically clear shared output transforms, defaults identity, raw HDR intact, tests for live/saved/panel consistency; coordinate ownership."

Saturation [0,2], vibrance [-1,1], and normalized black point [0,.5] now run in the existing profiled display path. Defaults are identity and byte-identical to pre-change128x128 seed804 images. The controls are expressly artistic linear-sRGB operations, not measured calibration or uniform perceptual color science. The same formulas and metadata apply to live PNG, direct test panel, and saved-HDR retouch. Legacy HDR stays unchanged and reports unsupported controls.84 Spectral and69 Turing focused tests pass; raw HDR byte preservation and exact output parity verified. No GPU work or server restart in this increment.

## Follow-on: full bounce budget and emitter-facing audit, 2026-11-09

User instruction: "I would like the bounce slider to go out to 200 with a nonlinear control giving more articulation the lower the value".

Spectral worker now validates integer max_bounces from 1 through 200 before creating a GL context. Default remains4. Both existing GPU paths already use dynamic uMaxBounces, with no hidden12 clamp: forward uses bounce<uMaxBounces, camera retains its existing bounce<=uMaxBounces convention. No shader, sampling or physics change was required. Mocked real-worker tests prove budgets1,4,12,13,50,200 reach both forward and camera constructors unchanged, and invalid/fractional/>200 values are rejected before renderer setup.90 targeted tests passed. The active user worker PID44056 was detected, so no high-depth GPU timing was run or user exposure interrupted.200 is50 times the forward loop budget versus4, not a total-runtime bound; early exits, secondary work and scene complexity vary. Fixed field/pending buffers do not scale with this setting. Watchdog behavior at maximum depth is not verified.

Separate read-only facing audit used actual current Studio circleFace source, Spectral _derive_lights_from_fixtures, and RayOrder.bake_rays. Default background intended direction is [0,.4819187498,-.8762159087], while triangle-derived source direction is its exact opposite (dot=-1). All4096 generated Lambertian rays were in the opposite hemisphere; none were aimed into the intended hemisphere. The photo export is positive scale plus translation and does not compensate handedness. Camera emissive hits remain two-sided and direct-light estimation ignores source direction; visible glow therefore cannot prove correct emission orientation. No winding was changed. Parent and fixture owner received the result.

Current-state correction after coordinated fixture update: the fixture owner subsequently changed buildFixtureFaces to swap only the emissive faces for all ten roles. I independently ran the actual current sceneForPhoto/MRP1 fixture harness through Spectral's source reader. All ten derived directions now align with their authored axes (dot approximately1); all4096 sampled background rays now fall in the intended hemisphere. The integrated fixture test passes. The earlier opposite-direction probe remains evidence of the old construction, not the current one. Camera two-sided emissive visibility and directionless direct-light estimation remain unchanged approximations.

## Follow-on: opt-in firefly suppression, 2026-11-09

User request: "can we do something like a noise filter toggle on sensitivity to stray pin lights".

The existing display path now has a disabled-by-default, adjustable eight-neighbor isolated-luminance filter. A threshold detects outliers; strength blends them toward the brightest neighbor while preserving RGB ratios. Connected detail is favored, borders remain unchanged, and the original HDR still determines auto-white. Raw/profile data is never overwritten. Genuine isolated glints can be indistinguishable from noise and may be reduced; that tradeoff is explicit in metadata and UI handoff.102 Spectral and76 Turing tests passed, including small real features, live/saved parity and toggle recovery without HDR changes. No GPU, transport, restart or publication work in this increment.

## Follow-on: adjustable percentile and specular-shot clipping, 2026-11-09

User request: "give me a slider on the percentile or something, we need a mode that is like, okay, this is a specular shot. forget what the light levels are, everything has to be clipped".

Added percentile and specular normalization modes. The percentile is adjustable from 1 to 100. The omitted default is 99, or 95 for specular mode. Existing auto99 remains exactly fixed at 99, and manual white is unchanged. Specular uses linear luminance clipping above the selected white; lower values remain graded. Raw HDR and the sensor profile remain intact. The shared recipe covers live output, saved retouch and the direct sensor panel, with actual effective-white metadata. Tests validate known HDR, scaling invariance, black, legacy defaults, range rejection and saved-HDR byte preservation. A small old Turing red-luma coefficient discrepancy was corrected to the exact Spectral coefficient rather than weakening parity checks.

Final targeted checks: 119 Spectral tests and 96 Turing tests passed; one existing cffi warning remains. Both baked and direct-live snapshot metadata paths are explicitly tested. Default and manual output pixels match the pre-change reference arrays. No GPU job, active-exposure interruption, production API, publication or restart was performed. UI/browser work is coordinated separately.
