# OpenRGB Full-Frame Shell

**Date:** 1791057588
**Title:** Minimal OpenRGB SDK interface for ASRock full-controller frames

## Overview

Added a thin Python interface and command shell for an already-running OpenRGB SDK server. The wrapper preserves the ASRock Polychrome USB controller's real ownership model: Direct-mode per-position updates must be sent as complete controller frames rather than as nominal zone or single-LED updates.

## Steps Taken

- Reviewed the pinned OpenRGB 1.0 CLI, SDK protocol, ASRock Polychrome USB controller, and server-host binding paths.
- Reviewed the maintained `openrgb-python` client and pinned its current stable release within the `0.3` line.
- Added a dry-run-by-default shell, injectable API, bounded dependency file, user documentation, and mock-only tests under `tools/openrgb_sdk_shell/`.
- Did not launch OpenRGB, connect to its SDK server, detect devices, or issue any hardware command.

## Observed Behaviour

- Confirmed whole-device Static red (`400000`) and white (`202020`) had worked for the user on the cooler and motherboard.
- Confirmed that apparent pale red under the white command was the motherboard's colored backing plate, not a channel-order failure.
- Direct full-frame addressing is verified on `Addressable Header 3/Audio`.
- The driver's `UpdateSingleLED()` and `UpdateZoneLEDs()` paths transmit only the zone's first color. Only the full-device `UpdateLEDs()` path reaches `WriteAllZones()` and its per-position payload.
- `--server-host 127.0.0.1` is supported by the pinned CLI and is documented explicitly for loopback-only use.

## Lessons Learned

- API surface does not guarantee hardware granularity; controller-specific method implementations determine what reaches the wire.
- Logical header capacity is not a physical LED census.
- Passive parallel ARGB splitters necessarily mirror; serial chains can expose ordered positions.

## Next Steps

- With all competing RGB software closed, the user may run one explicit Direct-mode, one-position mapping test and record which physical sites respond.
- Record verified physical wiring and logical-to-physical LED ranges before defining software segments or animation loops.

## Managed-dt LLVM walk update

The shell now maps controller-provided addresses and regions onto a dense law
domain while preserving each address's region identity and full-frame slot.
Noncontiguous addresses and region order no longer have to equal frame order.

`tools/openrgb_sdk_shell/openrgb_dt_walk.py` authors one simultaneous SymPy
law for a white point walking across that dense domain. The repository's
`piece_from_law` route materializes its batch-shaped AbstractTensor source and
compiles it to LLVM. A persistent `instantiate_system` state owns the columns,
and each display frame is one `advance_round` window. The law publishes
`max_vel` and `dt_limit`; no separate animation integrator or state loop was
introduced.

The authored wrap inside the white-point condition made SymPy produce a
Boolean `ITE`. Turing already represented the same identity as `Select` and
materialized it as elementwise `AbstractTensor.where`, but its SymPy ingestion
table omitted `sympy.ITE`. The table now maps `ITE` directly to the existing
three-input `Select` operation, with a focused ingestion regression.

Offline results:

- Nine shell/mapping/managed-dt tests passed, including the real native route.
- The real nested-`ITE` law compiled through AbstractTensor to LLVM with no
  lowering shortfalls.
- Four real managed-dt windows over a deliberately reordered address domain
  produced the expected `0 -> 1 -> 2 -> 0` white walk.
- At that stage, no OpenRGB process had been launched and no hardware write had
  been issued.

## Live controller validation (2026-10-03)

- Started exactly one reviewed portable OpenRGB process, bound its SDK server
  to `127.0.0.1:6742`, and inventoried one `ASRock B650M Pro RS WiFi` device
  with 245 logical addresses across the expected five regions.
- A one-position Direct-mode write isolated Header 3 position 0 on the
  cooler/fan chain. Subsequent slow calibration resolved its installed,
  upside-down-view location as bottom-left.
- Logical red rendered physically green; logical green rendered physically
  red. This establishes an R/G permutation for colored effects; equal-channel
  white is unchanged.
- A managed-dt/LLVM white walk resolved the installed radial map as
  `0=bottom-left`, `1=left`, `2=top-left`, `3=top`, `4=top-right`, `5=right`,
  `6=bottom-right`, and `7=bottom`. Increasing positions advance clockwise.
- The controller advertises 80 positions on Header 3, while the measured chain
  occupies positions 0 through 7. The walk can now project that measured
  subset onto dense indices 0 through 7 and map every result back into the
  original 245-slot full-controller frame.
- A finite nine-frame run completed without controller or dt errors to exercise
  positions 0 through 7 followed by the symbolic wrap to position 0.
- Thirteen offline tests pass, including native SymPy-to-AbstractTensor-to-LLVM
  compilation, managed-dt advancement, selected-zone projection, full-frame
  back-mapping, calibrated position bounds, and explicit start phase.
- The four independently addressable motherboard `PCB` positions were walked
  with the cooler black. Position 0 appears above the board; positions 1, 2,
  and 3 appear along the board's right side. The emitters are mounted on the
  backside of the motherboard.
- The current OpenRGB SDK inventory contains no GPU device. All 245 enumerated
  addresses belong to the motherboard controller, so GPU lighting is outside
  the presently discovered domain.
- The session ended with one full black Direct frame; no mapped LED was left
  intentionally illuminated.

## Continuous locator and 60 Hz result

- Added an analytic radial float generator rather than a nine-site spatial
  discretization. Each arbitrary point produces one eight-float contribution
  row. Squared positive angular alignment gives `1.0` to the aligned perimeter
  light and `0.5` to its direct 45-degree neighbors; radius blends toward an
  equal dim contribution across all lights at the center.
- Added a normalized hypotrochoid path compiled through the same SymPy,
  AbstractTensor, LLVM, and managed-dt route. For the default `R=5, r=3`
  parameters, repeat detection correctly finds the full `6*pi` cycle.
- Added an optional adaptive LUT bake. The analytic generator remains the
  truth; the LUT samples until adjacent contribution rows fit the selected
  color-code epsilon. Dynamic site and time budgets return a complete closed
  LUT with explicit achieved epsilon, saturation status, and cutoff reason.
- The 8-bit default bake used 2,916 sites, reached maximum adjacent delta
  `0.00391839 <= 1/255`, and closed with repeat error `1.665e-16`.
- A two-second dry 60 Hz run achieved 59.985 Hz with zero deadline misses and
  2.683 ms p95 solve time.
- A five-second live 60 Hz run achieved 59.996 Hz with zero deadline misses,
  3.184 ms p95 total work, 10.387 ms maximum work, and 0.467 ms p95 SDK send.
  Python consumed 14.69% of one core, OpenRGB 0.14% of one core, total system
  CPU averaged 7.92% and peaked at 11.79%, and OpenRGB working-set growth was
  0 MiB.
- Nineteen focused offline tests pass across the shell, linear managed walk,
  continuous generator, color-depth LUT, repeat closure, and dynamic cutoff.

## Chaotic CMY three-body field

- Added HSV-to-sRGB conversion and fixed cyan, magenta, and yellow source
  colors. The calibrated Header 3 `GRB` wire correction is applied only at the
  final full-frame boundary.
- Added a genuine free three-body law: softened Newtonian mutual gravity,
  symplectic velocity-first state updates, and Burrau's chaotic Pythagorean
  masses `3, 4, 5` starting at rest on a scaled `3:4:5` triangle. The managed
  dt system owns all position and velocity columns; no path or orbit is
  prescribed.
- Each integrated body generates one continuous radial float row. The three
  CMY-weighted rows scatter-add into every one of the eight cooler addresses.
  A shared instantaneous radius preserves relative placement and maps the
  outermost body to unit radius; direction vectors are normalized separately.
- Twenty-three focused tests pass after the continuous-radius correction.
- A one-second dry run achieved 59.980 Hz with zero deadline misses and 2.839
  ms p95 work.
- A ten-second live run achieved 59.999 Hz with zero deadline misses, 3.028 ms
  p95 total work, 5.685 ms maximum work, 0.395 ms p95 SDK send time, and 16.72%
  of one CPU core in the Python process.

## Ray-tracer spectral field unification

- Replaced RGB/HSV field transport with the spectral-analyzer ray tracer's
  established eight-band display basis: 700 through 400 nm, followed by its
  exact band-to-linear-RGB matrix only at the output boundary.
- Added one canonical radial kernel shared by spirograph and three-body.
  Its original `center_contribution=0.25` form conserved total driver weight
  at exactly `2.0` over all radii and angles on the eight-site ring. This was
  later superseded by the sharper distance-localized kernel documented below.
- Authored field composition as SymPy matrices. Per LED lane, the compiled law
  evaluates `(1xS source weights) @ (Sx8 spectra) @ (8x3 display response)`.
  SymPy expands the fixed matrix products into the repository's scalar law
  expressions, and `piece_from_law` executes all eight LED lanes as the
  `AbstractTensor` batch in LLVM.
- Cyan and yellow use single wavelength bands. Magenta is a two-line
  red-plus-violet composite; the spirograph white is a nonnegative three-line
  spectrum that projects to equal linear RGB. Thus pure and composite sources
  remain spectra through spatial accumulation.
- Removed per-channel pre-display clipping. A smooth exponential exposure is
  applied only after spectral accumulation and linear display projection,
  retaining distinctions between overlapping sources instead of immediately
  producing clipped white plateaus.
- Added `openrgb_dt_field.py` as a selector for `spirograph` and `three-body`;
  the effects retain their own managed-dt engines.
- Twenty-six focused tests pass. Both updated laws compiled through SymPy,
  repository SSA, AbstractTensor, and LLVM with zero lowering shortfalls. In a
  120-frame offline native measurement, spirograph solved in 1.56 ms/frame
  (~640 Hz) and three-body in 1.62 ms/frame (~617 Hz), excluding one-time
  compilation.
- A bounded 300-frame hardware run of the spectral three-body mode achieved
  59.995 Hz with zero deadline misses, 2.632 ms median work, 3.099 ms p95,
  4.871 ms maximum, 0.416 ms p95 SDK send time, and 16.56% of one CPU core.
  The fresh-process symbolic/native build took roughly two minutes before the
  timed frame loop; runtime cadence and compilation latency are reported
  separately.

## Intrinsic torus and observable eager lane

- Replaced the planar periodic interpretation with two angular coordinates on
  a standard embedded torus. Pair interaction distance is the torus metric
  line element integrated by eight-site Gaussian quadrature along the direct
  shortest wrapped parameter lift; its analytic gradient drives the force.
- The repository `RK4Integrator` now manifests four causal derivative pieces
  and one endpoint piece. The endpoint publishes relative mechanical-energy
  and cyclic-`u` momentum defects through declared dt error channels.
- Added an eager piece lane using CSE'd SymPy equations, the same truth used by
  the compiler validator, behind the ordinary PieceState interface. This lets
  motion and controller behavior be inspected before the lengthy LLVM lower.
- Added an optional Pygame view of the face-on torus, three CMY bodies and
  trails, eight logical LEDs, coordinates, dt continuation, attempt counts,
  rejection counts, and conservation errors. `--frames 0 --window` leaves
  stopping to Escape or closing the window; `--print-every 1` streams every
  completed coordinate state.
- A two-frame eager dry run changed all six angular coordinates, accepted both
  `1/60 s` attempts, and published maximum energy and momentum defects of
  `4.052e-15` and `4.235e-12`. Nine focused geometry/color tests pass.
- Removed the unjustified `frame_seconds * 1e-8` lower timestep floor. The
  controller now has `dt_min=None`, so it may continue subdividing until its
  declared error targets are met rather than being stopped by an animation-
  specific numerical cutoff.
- Added `--trace-rejections`. It reports each candidate from inside the real
  `advance_pieces` attempt before rollback: dt, exact failing channel and
  limit, all `(u,v,du,dv)` values, torus-metric speeds, and three intrinsic
  pair distances. It also reports the accepted dt after a rejection streak
  and retains the latest rejection summary in the Pygame status display.

## Laplace--Beltrami Green gravity correction

- The wrapped direct-path metric integral was diagnosed as the frame-90
  failure source, not the dt system: one pair crossed the principal-angle cut,
  its selected path length jumped, the pair potential changed discontinuously,
  and the energy channel correctly rejected every crossing attempt.
- Replaced that force with the zero-mean Green operator of the embedded torus
  Laplace--Beltrami operator. `ParametricDomain` supplies the exact symbolic
  metric and operator. A finite real periodic Fourier basis excludes the
  constant mode, and `AbstractTensor.linalg.solve` inverts the weak
  `-Delta_g` stiffness matrix once. The inverse returns to the SymPy particle
  laws as the common potential and analytic coordinate gradients.
- Particle angles are deliberately unbounded; every field basis function is
  periodic, so no wrap counter, principal lift, or wrap-ratio convention is
  present in the force or energy.
- Twelve focused tests pass. They cover spectral color accumulation, exact
  derived torus metric, the repository Laplace--Beltrami operator, Green
  stiffness symmetry/invertibility, Green reciprocity and periodicity,
  continuity across the old principal wrap cut, and local force direction.
- A 120-frame eager dry run crossed the former frame-90 failure point with 120
  accepted full-frame attempts and zero rejections. Measured output was
  59.998 Hz, zero deadline misses, 4.206 ms median work, 4.627 ms p95,
  5.790 ms maximum, maximum relative energy error `1.30107841e-13`, and
  maximum cyclic-momentum error `2.94431146e-13`.

## Distance-localized light projection

- Replaced the uniform radial interpolation background with the compact
  squared-distance lobe `max(0, 1 - ||p-u_i||^2/2)^2`. The established
  calibration points remain exact at the default center setting: center
  `0.25`, aligned perimeter light `1.0`, direct neighbors `0.5`, and
  perpendicular/opposite lights `0.0`. Near the ring, the opposite light is
  now absent instead of receiving the former center-derived baseline.
- The Pygame diagram now labels every physical ring site with its logical
  index and final calculated integer `R/G/B` drive tuple. Normal visible runs
  leave terminal coordinate and rejection tracing disabled; both remain
  explicitly available through their existing flags.
- Twenty-two focused spirograph/three-body tests pass, and a two-frame eager
  managed-dt dry run accepted both full frame steps with finite output.

## Processor-following system temperature

- OpenRGB 1.0 and the installed `openrgb-python` SDK expose lighting
  controllers but no processor-sensor API. Windows ACPI and thermal performance
  counter probes returned no temperature samples on this machine.
- Reused the installed NiceHash Miner `LibreHardwareMonitorLib.dll`, which
  identifies the processor as `AMD Ryzen 9 7900X` and its package sensor as
  `/amdcpu/0/temperature/2`, `Core (Tctl/Tdie)`. Non-elevated access exposes
  the sensor but returns zero; the program therefore requires an elevated
  parent when `--cpu-temperature` is selected.
- Added a persistent Windows PowerShell sensor worker. The application requests
  and validates one real Celsius sample before each frame, then updates the
  registered managed-state `system_temperature_c` column across all lanes.
  Missing, zero, nonfinite, and implausible readings stop the run rather than
  substituting an invented temperature. Pygame displays the live value.
- Added `run_temperature_three_body.ps1` and documented a single `RunAs`
  command for the complete sensor-following, eager, visible hardware run.
- Python and PowerShell sources compile/parse successfully. A non-elevated
  sensor run fails immediately with the intended rights diagnostic, while a
  two-frame temperature-disabled managed regression accepted both full frame
  steps with unchanged conservation behavior.
- The first live elevated attempt exposed that an initial-table-only
  temperature column is not retained by `PieceState`. The endpoint law now
  declares the identity parameter/output
  `system_temperature_c -> system_temperature_c_next`, placing it on the
  ordinary piece contract and dt save/restore path. An instantiated eager
  managed system retained an injected `61.25 C` value across one complete
  frame on all eight lanes.
- Added the actual kinetic thermostat after clarifying that merely publishing
  the temperature was insufficient. On each new reading, all `du` and `dv`
  columns scale by `sqrt(T_new_K/T_old_K)`, making their torus-metric kinetic
  energy scale exactly by `T_new_K/T_old_K`. Thus a falling processor reading
  removes energy and a rising reading adds it without treating Celsius zero as
  physical zero or introducing a fitted speed constant. Pygame reports the
  multiplier and signed kinetic-energy transfer.
- A managed-state validation with nonzero velocities moved from `40 C` to
  `80 C`. The measured energy ratio and Kelvin ratio both equaled
  `1.1277343126297301`; the reported and directly measured energy increments
  agreed to floating-point roundoff, and the `80 C` state survived the next
  complete dt frame. Thirteen focused three-body tests pass.

## Interactive Pygame field studio

- Diagnosed the all-black Pygame LED spots: the renderer zipped the eight ring
  directions with the first eight entries of the full-controller frame, while
  the selected Header 3 domain occupies its own mapped frame indices. Both the
  original viewer and the new studio now extract colors through each
  `DomainAddress.frame_index`. A focused non-prefix-layout test covers this.
- Generalized the shared radial kernel with a runtime `falloff_power` column.
  The distance base is raised to that positive power while the center
  correction is recomputed against `0.5**falloff_power`, preserving the chosen
  center value and exact aligned perimeter peak. Both three-body and
  spirograph laws consume the same column.
- Added `openrgb_field_studio.py`, a direct Pygame control surface over the
  existing managed eager engines. It switches between torus physics and
  spirograph positioning, edits falloff and center values, fits arbitrary HSV
  hues into the nonnegative eight-band ray-tracer spectrum, advances per-body
  hue angles, tunes all three masses, and activates one to three compiled body
  slots without rebuilding laws.
- Added physical/viewer calibration: forward or reverse order, integer offset,
  and mouse dragging which rotates the displayed ring continuously before
  snapping at eighth-turn boundaries. The snapped transform is applied to the
  selected full-frame hardware slots while preserving unrelated controller
  addresses.
- Temperature is opt-in. `run_field_studio.ps1` runs normally without
  elevation; only `-CpuTemperature` selects the privileged sensor worker and
  thermostat.
- Inspected Turing AbstractUI. Its present common object owns neutral model and
  HTML/CSS/JavaScript/JSON packets, with Canvas/WebGL consumers elsewhere, but
  there is no Pygame packet, renderer, or event adapter. The studio therefore
  did not invent a nominal AbstractUI backend that could not round-trip through
  the existing C-shell/SDK format.
- Sphere mode is shown disabled rather than faked. The current torus state and
  Green operator cannot become sphere physics by changing projection; a real
  mode needs a spherical state chart, spherical Laplace--Beltrami Green law,
  and constraint-preserving integration.
- Twenty-nine focused studio/spirograph/three-body tests and all forty-two
  OpenRGB-shell tests pass. A complete
  two-frame headless eager studio run instantiated and advanced both managed
  engines successfully with the runtime-power law.

## Empirical spirograph phase closure

- Diagnosed a split between the locator's LUT and its compiled motion law: the
  LUT found and validated the default hypotrochoid's full `6*pi` repeat, while
  the managed SymPy phase equation still wrapped at a hard-coded `2*pi`.
- Repeat discovery now tests the actual normalized point signal at several
  phase offsets after each whole base turn. This rejects isolated
  self-intersections and observes genuine shorter closures in degenerate
  configurations, such as `pen_offset=0` yielding a `2*pi` circle.
- The discovered period is a registered `repeat_period` column in the ordinary
  managed piece contract. The compiled law consumes that column for both
  positive and negative angular velocity rather than carrying an independent
  period convention.
- Focused tests verify that the default phase crosses `2*pi` without wrapping,
  wraps at `6*pi`, and wraps backward to the end of the same period. Seventeen
  spirograph/studio tests pass, and a two-frame headless studio run constructed
  and advanced both established engines successfully. The studio status panel
  displays the active empirical repeat.

## Dynamic eager spirograph field controls

- Kept the compiled/eager positioning law single-source and reusable rather
  than baking a fixed source count into it. The studio owns a dynamic list of
  independent managed eager instances; adding or removing one does not rebuild
  or compile the law.
- Each spirograph source has independently selectable spectral hue, hue rate,
  and integer `(R-r):r` controls. Ratio changes update that source's registered
  `fixed_radius`, `rolling_radius`, and empirically discovered `repeat_period`
  columns together.
- Source contribution rows are scatter-added against all source spectra with
  the established numeric spectral-field matrices. The accumulated spectrum
  is converted to display RGB and exposed only after addition, preserving the
  same composite-colour contract as the physics field.
- The spirograph speed slider is no longer fixed to `[-5,5]`. Its range follows
  the launch refresh rate and the fastest active ratio harmonic under the
  eight-site ring's fourth-spatial-mode temporal Nyquist bound. The PowerShell
  launcher now accepts `-Fps` so the user can raise both cadence and that bound.
- Hardware-enabled shutdown now sends one full-controller black frame from a
  `finally` path on normal completion, Escape, window close, or an error after
  Direct mode selection.
- Corrected calibration ownership after the first implementation permuted
  solved colors both in the viewer and again at the hardware boundary. Offset,
  ordering, and live drag rotation now define the real-world direction of each
  controller address before field evaluation. The GUI places that address's
  resulting color at the same direction, while the full frame is sent in its
  original controller-address order. Point position, displayed color, and the
  physical emitter therefore remain coupled throughout dragging and after the
  eighth-turn snap.
- A remaining calibration failure was a concrete orientation-sign error. The
  calibrated ring address indices increase clockwise, while mouse `atan2`
  angles increase counterclockwise. Drag release had added the angular step to
  the address offset, causing the continuously correct drag preview to snap in
  the opposite direction. Release now subtracts the rounded angular step.
  Ring labels now show the controller address identity at its calibrated site,
  rather than a fixed site number, so the object being aligned is explicit.
- Single-source observation isolated a second issue to color-field deposition,
  not point placement. The general eager spirograph path still evaluated a
  canonical ring after counter-rotating the point and then reordered the
  resulting row. That parallel angular convention was removed. The canonical
  numeric radial kernel now accepts an explicit direction matrix, and the
  spirograph evaluates every source directly against the same calibrated
  per-address directions used by the GUI and physics law.
- Because the reported hue mismatch also occurs in physics mode, exposed the
  remaining common boundary explicitly: a live `RGB` versus `GRB (swap R/G)`
  control now selects final full-frame channel order for both engines. An R/G
  exchange reflects and offsets the HSV wheel, so it can make correctly placed
  sources appear to carry another source's color or traverse hue in the wrong
  direction. Spectral accumulation and GUI colors remain RGB; the selectable
  exchange exists only at the controller wire boundary.
- Split evolving source state from receiver sampling. The existing eight-lane
  managed engine remains the sole physics/spirograph state owner, while its
  current source positions and spectra are sampled every frame at both the
  eight-point cooler ring and a four-point vertical motherboard `PCB` strip.
  Those domain frames merge by preserved full-controller frame indices before
  the common wire-order boundary, so both physical surfaces continue updating
  regardless of which top tab is visible.
- Added Ring and Motherboard strip tabs. The ring's default viewer calibration
  places controller identity 0 at the directly leftmost receiver; its drag,
  snap, and direction controls remain persistent. The strip displays identity
  0 at the top and identities 1 through 3 downward while using the same
  continuous distance/spectral field law as the ring.

## Persistent independent array systems

- Added `openrgb_field_studio_config.json` and a strict versioned loader with
  atomic replacement saves. UI tab changes, control releases, identity
  calibration, source membership, and per-array wire order autosave.
- Separated configuration into controller-owned `arrays`, independently
  evolving `fields`, and explicit array-to-field assignment by `field_id`.
  Array records own zone/address geometry, channel order, and a relative
  `(x,y,rotation,scale)` placement. Field records own engine selection,
  physics bodies, dynamic spirograph sources, hues/rates, and light-field
  response controls.
- The default ring and motherboard bindings now construct distinct managed
  three-body states and distinct eager spirograph collections. Every field is
  advanced once per display frame whether or not an assigned array's tab is
  visible. Each array samples only its assigned field; array results receive
  their own RGB/GRB boundary and then merge by preserved controller frame
  indices.
- Multiple arrays may reference one field in JSON, and receiver placement is
  already applied before sampling. This establishes the data boundary for the
  later emitter-scene editor without implementing that UI prematurely.
- Added `ensure_openrgb_server.ps1` to both hardware launchers. It reuses a
  reachable loopback SDK server, or starts the already-reviewed portable
  OpenRGB build hidden with the established isolated configuration and
  loopback-only server arguments. It waits for socket readiness and refuses to
  launch a competing copy when an OpenRGB process already exists without a
  reachable SDK endpoint.

## Live speaker vectorscope and torus projection ratios

- Added `vectorscope` as a third per-field engine beside `physics` and
  `spirograph`. It uses the spectral-analyzer's established mid/side screen
  law, `x=(L+R)/sqrt(2)` and `y=(L-R)/sqrt(2)`, and treats a decimated recent
  stereo trace as simultaneous spectral emitters in the same receiver
  contribution matrix used by the other modes.
- Added a native shared-mode WASAPI loopback reader for the default Windows
  speaker render endpoint. It requires neither processor-sensor access nor
  elevation. The physical capture is shared, while vectorscope hue, hue rate,
  gain, field level, trace population, tail duration, pause state, and field
  sampling remain independent.
- Added autosaved physics projection radii for the torus inner edge, torus
  outer edge, and LED ring. Their ratio may put motion inside, across, or
  outside the receiver ring without changing the intrinsic `R=1.5, r=0.5`
  geometry used by the established Laplace--Beltrami Green dynamics.
- Per the user's standing direction, no automated tests or live hardware run
  were performed for this UI iteration.
- The first user run showed no vectorscope output. Direct inspection of the
  real capture object found `RPC_E_CHANGED_MODE`: the worker initialized COM
  as multithreaded immediately before `comtypes` initialized it as an
  apartment thread. Capture now imports `comtypes` on the caller and performs
  one matching worker-thread COM initialization. The same speaker-loopback
  path then remained live at 48 kHz stereo and returned changing nonzero
  samples.
- A second suppression was measured in the light-field boundary. Audio
  amplitude already determines vectorscope coordinate radius, but the first
  implementation multiplied every emitter's spectrum by that amplitude
  again. With the saved zero-center, 6.38-power field this yielded a peak LED
  contribution of only 0.0016. Emitters now have uniform trace energy behind
  one global RMS activity gate, avoiding squared attenuation; the UI reports
  live RMS and XY peak for direct observability.

## Native fftfree logarithmic-band vectorscope

- Replaced the Python COM capture path with a native C++ ingestion library.
  Each independent field owns a WASAPI loopback context which captures the
  default stereo render endpoint and dynamically loads the existing
  `fftfree/build/Release/fft_cffi.dll` C ABI. No substitute FFT implementation
  was added beside the subrepository.
- The native analyzer owns independent left/right real FFT plans. FFT window
  size is a selectable power of two and `oversample` is a selectable
  power-of-two temporal overlap (`hop = fft_size / oversample`). A Hann window
  is applied before each transform.
- The external node count is intentionally not restricted to a power of two.
  The analyzer aggregates FFT coefficients and power into arbitrary-count,
  logarithmically spaced bands over the selected frequency interval. Each band
  supplies a band-limited mid/side vectorscope position, a log-frequency hue,
  and a power-derived contribution strength to the existing spectral light
  field.
- Added persisted per-field controls for log-band node count, FFT power,
  overlap power, and low/high frequency. The launcher automatically builds the
  native scope DLL when absent or older than its source.
- Band power controls both each emitter's spectral-field contribution strength
  and the value/brightness of its matching Pygame point. The status line shows
  live input RMS and peak normalized band power, so native ingestion and the
  brightness path remain directly observable before considering the hardware.
- Added a common per-field source rotation offset/rate and a per-array receiver
  rotation offset/rate. These sit outside the vectorscope engine: the source
  transform applies to physics bodies, spirograph sources, and band emitters,
  while the receiver transform applies in every mode. Viewer and hardware
  sampling consume the same transformed points.
- Made the vectorscope's `sqrt(N)` band-count normalization selectable. Keeping
  it enabled stabilizes aggregate level as band count changes; disabling it
  sends the unnormalized sum of measured band powers into spectral
  accumulation for genuine input-dependent dynamic range.
- Added source-local strobe modulation beside the hue controls. Every physics
  body and dynamic spirograph source owns an independent waveform, frequency,
  and phase. The vectorscope colour source owns one envelope applied after the
  independent FFT band powers, preserving their relative spectral dynamics.
  The selectable shapes are off/constant, sine, square, triangle, and saw;
  rate sliders stop at the current refresh-rate Nyquist bound. Pygame source
  brightness and the sampled spectral field consume the same envelope.
- Added onset matching as a sixth strobe waveform without adding a tempo or
  phase predictor. The existing logarithmic FFT powers are returned to linear
  amplitude, positive spectral flux is measured against a 1.5-second adaptive
  floor and 0.75-second decaying peak, and an 80-millisecond release produces
  a visible unit beat envelope. Each source has a signed response: `+1` maps a
  strong onset to light, `-1` maps it to a dropout, and `0` leaves the source
  unmodulated. Physics and spirograph fields lazily start their independent
  native capture context only when an active source selects `beat`.

## Entropic temperature coupling

- Superseded the earlier temperature-change velocity rescaling. Setting a
  temperature now only updates the registered managed temperature column; it
  never multiplies all body velocities by a common factor.
- Before each conservative managed-dt frame, an exact Ornstein--Uhlenbeck bath
  update acts in the orthonormal tangent frame of each particle on the torus.
  Independent Gaussian impulses affect both tangent directions, while the
  exponential retention term supplies matching dissipation. This makes fixed
  temperature an ongoing entropic influence rather than a one-time speed
  change. The managed integrator then advances from that externally forced
  state with its existing conservation error channels unchanged.
- The explicit simulation convention is dimensionless thermal energy
  `temperature_C / 100`; the per-second bath coupling is a separate persisted
  control. Pygame reports the realized tangent impulse RMS and signed kinetic
  energy exchange for each frame.
- Without the privileged processor sensor, every independent field now has a
  persisted manual 0--120 C slider. Selecting `-CpuTemperature` replaces that
  slider with the validated hardware reading while leaving bath coupling
  adjustable.

## Restored multiband vectorscope tails

- Restored temporal tails after the native multiband conversion had retained
  only the latest FFT-band coordinates. Each logarithmic band now owns a path
  through its historical mid/side positions; the viewer connects the same band
  across time rather than connecting adjacent frequency bands as though they
  were one trace.
- Added a persisted `tail decay ms` control. Each receiver array applies the
  same exponential decay to its already-solved RGB field using peak hold
  (`max(current, retained * decay)`), so the physical LEDs show the temporal
  evolution rather than only the Pygame diagram. The operation does not sum or
  normalize frames and therefore preserves source dynamic range.
- History has a value-derived cutoff at 6.25 decay constants, where a retained
  full-scale 8-bit channel falls below half a code value. A zero setting makes
  both viewer and receiver output instantaneous.
- The native DLL compiled successfully against the installed toolchain. A
  direct non-hardware capture check reported a running 48 kHz context, 32
  populated bands, finite nonzero stereo coordinates, and finite nonzero band
  power. The full OpenRGB studio and hardware output were deliberately left to
  the user's launcher command.

## Prompt History

> "if you can hook me up with a c and or python shell that just gives me the api/abi interface to the controller with as little in the way as possible I can work out the rest with programming"

> "Build a thin usable Python SDK wrapper/shell in an appropriate local root-repo location (inspect guidance and avoid collisions), favor existing maintained openrgb-python rather than handwritten protocol. Keep minimal: enumerate devices/zones/global offsets, explicit mode selection, send full-controller RGB frame, mapping/example one dim pixel and solid colors, public underlying client/device escape hatch."

> "Offline/mock tests only; do not run SDK writes or detection against hardware, preserving prior denied-action restriction."

> "I'd like to be able to map the arbitrary system provided domains of addresses/regions/w/e into a domain in a sympy to abstract tensor to llvm style rule, using dt system to solve the led states each frame, doing for now just a linear walk across addresses with a white light while others are off"

> "well hold on what do we need for ITE"

> "can you continue then with the end to end compiled control of system led lights"

> "so do we have the spiro as a mode and physics bodies as a mode? both using our principle of treating leds as the drivers of a light field, we may need to be more careful to check our math of light field accumulation and expression and representation"

> "yes but make sure we work in tensors or matrices for sympy tensors maybe i don't know, an efficient evaluation and that we allow spectral accumulation, allowing pure or composite colors to accumulate correctly in the field"

> "oh look at the spectral analyzer subrepo spectral ray tracer"

> "intrinsic distance it should be toroid surface distance integral"

> "why don't you try to get it working eagerly before compiling"

> "can you let me run the command or something it flashed a few times then it's constant. can you make a fucking pygame god damn window or something of the objects or put their coordinates in live output and let me run the program"

> "is gravity the real surface integral or is it something else"

> "abstract tensor has an laplace beltrami on continuous toroidal geometry, you might have to build something that's easier to express in terms of abstract tensor plus some sympy laws or much more elegant and complete sympy, we can maintain a normalized wrap ratio if a wrap count seems too likely to have a problem? I am not sure we have green functions worked out but we can if you want, I don't want to take all night on this but I don't want to pretend it's all that hard to accomplish the particle sim we're making"

> "the eager mode is fast enough we should play with developing a ui and integrating the other modes into the pygame system. you can check abstractui but i don't think it has been made robust yet to supply a python pygame version that can be absorbed by the c shell and sdk. that's a huge intention you can report on it but not realize it if it woyld take too long to get it to match our format. we should make temp optional in a way that will not require admin if it's not being used. you should then be able to iterate on control and see the impact in pygame to realize the goals. each light spot should be filled by its color from whatever the last signal was, presently they all stay static 0/0/0 with no fill color. we want to be able to tune the distance fall off, tune the hue of each mass and if there is any hue angle velocity (rainbow shifting) by some rate or function. we want, add or subtract masses, switch the torus to a sphere, switch to the spirograph positioning engine instead of the physics, rotate the led layout by some integer to correct start offset to some location corresponding to real life, such as dragging a ring and the viewer rotates smoothly with snap at the eighth angles, and switch the ordering direction, again for the viewer, to be made to match with observed reality"

> "looks to me like you probably set a min dt that cannot achieve the stability metrics because the min dt is just some random crap"

> "when 2 reaches the center ring of the torous, then things freeze. makes me think you don't have the math right maybe on how the 2d gravity works on the surface of a torus? not sure, can you put some messages inside the dt sbudivision on failure tracking where it's at and what's failing"

> "continue adding features, the spirograph shouldn't just loop at 2pi or whatever it needs to understand it's empirical periodicity"

> "give us sliders for the ratio and have it exit to all off"

> "also let the speed go way higher on angular velocity on the spirograph, and let it add bodies, give them all their own ratio and color, they'll all fly around overtop each other belnding light like the physics, we want to be able to also push the refresh rate see what the nyquist is of circling the ring and limt to that or something"

> "don't compile them just make it a general case for sliders"

> "no tests don't waste my time I don't care if you see it, just make the changes and give me the command"

> "the only problem at the moment is that when i rotate the led identities the point and the color application become decoupled, the bottom of synchronizing the gui to the real object then doesn't really work. i may have poorly conceived of or explained it. the ball should always keep illuminating whatever the rotated lights are aligned to. the identities of lights are rotated and the user is aligning where the leds are in reality by their address"

> "let's make sure the openrgb server is auto started when unavailable"

> "I am angry and don't understand, why no matter what rotation or direction or rotation, I cannot turn the ring and get the lights on the device to correspond to the field I'm seeing"

> "cutting down to one body was extremely informative, the hue deposition angle is wrong and maybe not even the right clockwase if that makes sense. I can align position but hue is not concordant"

> "well the problem wasn't specific to spectrogrpah so that leaves me wondering, also it could be the channel convention swap too"

> "now let's be sure by default identity 0 led is directly leftmost, that's what we have to keep aligning to.  the next big task is going to be device tabs, so we made the controller for a field right, we set in stone the ring. now we nweed to make the ui have buttons at the top that change the view to different physical controllers found, and we're going to make them work the way they are configured even when the tab is not visible, but we'll want to have our ring and our motherboard strip as two tabs, so when I go to the motherboard strip my ring of lightews becomes a vertical line of lights, but the color field math is all the same, see, we just have a strip instead of a ring as the receiving points"

> "now we want to make the configuration backed by a json and persistent and autosaving, and we want to ensure we have a whole independent concurrent system for each light array. i noticed we had one that needed swap and one didn't, and they shared modes and orbiters, it's better if we can handle the configuration of all controllers independently, and separate that task from the question of placing the controllers on the same or their independent light fields, with relative positioning. so like, Iby defautl now we work so tehey can all be fully independent systems. then after that we'll work on how we could get the UI to allow constructing an emitter scene in a light field"

> "okay, so, we want another mode, like physics is a mode (and we want to be able to configure the float ratio between the inner radius of the torus, outer radius of the torus, and radius of the light array ring, so we can place movement outside the ring or any other configuration the new mode, though, beside physics and spirogrpah, and I think we can do this without admin, is to take the speaker out feed and place the emitters according to a live vectorgraph"

> "like a stereo vectorscope, do you know what I mean?"

> " I'm not getting any output on the vectorscope"

> "let's use fft free or freefft, whichever one, it's a subrepo, we should use it to supply at c ingestion the audio signal and it's power of two composition as requested so we can set point color brightness to be a property of bands of color in the vectorscope. this means we put the speaker grab in the c shell and we do an fft decomposition to requested bin count by powers of two of, actually, no I think we have to use power of twos for our thing unless we oversample.... oversample and present as logarithmic music scale divisions across all N nodes in play"

> "let's do something to let us set the angle of the vectorscope in a way that's like the hue vs hue speed sliders, let us do that for the field and the lights, move either through the field with an offset and a rate, and then let's make normalization of lighting optional so we can get genuine dynamic range"

> "field and source rotation was not for the vectorscope it was for all modes"

> "with that in mind, I think we should put strobe on as a source option with a wave shape choice, part of the color chooser set"

> "lets move with one, tempo/phase we'll figure out some other time"

> "subtle change, temp is the speed scaler in the physics? instead I feel we should make temperature an entropic influence on particles not just speed, and I think if it's not hooked to admin to a signal it should be freely adjustable by the user"

> "oh also, while you're working, someone removed the vectorscope tails, that's valuable, vectorscopes are too transient for leds in most cases so trails are how the sound visual evolves"

## Direct ASRock HID transport (2026-10-04)

- The studio's default hardware path no longer launches or connects to an
  OpenRGB server. `run_field_studio.ps1` now builds a native transport and
  closes competing OpenRGB processes before opening HID `26CE:01A2`.
- `native_asrock_hid/asrock_polychrome_hid.cpp` dynamically loads the reviewed
  bundle's `hidapi-hotplug.dll`, selects Direct mode once, and owns persistent
  65-byte `E3`/`E4` reports. Python crosses ctypes once per complete frame.
- The known controller layout remains explicit in the persistent studio JSON:
  245 logical colors across RGB1, ARGB1, ARGB2, PCB, and ARGB3, expanded to the
  firmware-facing 313-color padded transport image and 17 HID reports.
- Native packing preserves OpenRGB's established per-zone R/G behavior while
  updating only changed bytes in the persistent report image. The required
  padded report sequence is still transmitted in full pending evidence that
  this firmware accepts a shorter stream.
- The native DLL compiled successfully with MSVC 19.44. No hardware animation
  or test suite was run during this change, following the user's instruction.
- The frame path was then made non-blocking with a single pending latest-frame
  mailbox. New submissions replace only a not-yet-started frame; a transaction
  already in progress is never mutated. Shutdown drains the final pending
  all-off frame before closing the HID handle.
- The 17 ordered `E3`/`E4` output reports are now issued as a burst, followed by
  draining 17 controller replies. Direct-mode setup commands retain strict
  write/reply ordering. This removes OpenRGB's unproven per-chunk reply barrier
  while preserving report order and explicit failure on missing replies.
- Native counters expose submitted, completed, replaced, failed, busy, and
  pending state. The Pygame status line derives a rolling completed HID-frame
  rate over half-second observation windows, distinct from requested GUI rate.
- The updated asynchronous/burst native DLL compiled successfully. Hardware
  behavior was not exercised by the agent.
- User-observed direct-HID operation established the controller transaction
  ceiling: approximately 15 completed full frames/second at peak and a stable
  10.0 frames/second within about +/-0.1 Hz. These are completed padded
  17-report transactions from the new transport metrics, not requested GUI
  cadence. Higher producer rates are intentionally reduced by the latest-frame
  mailbox rather than accumulated as latency.
- The studio now treats the 17-report padded stream as a hypothesis to bench,
  not a requirement. Live controls independently select normal `E3` then `E4`
  streaming, repeated `E3` restarts, or `E4` continuations; a 1-through-18
  report prefix; and serial, adjustable-window, or burst reply scheduling.
  Rolling throughput sampling restarts after each profile change.
- User benching identified 14-report `stream` + `burst` as the optimum and it
  is now the saved/default production profile. Thirteen or fewer reports do not
  visibly update the device; more than 14 slow the completed-frame rate. Since
  13 reports nominally carry 738 bytes against 735 bytes for 245 RGB entries,
  the extra report is an observed controller latch threshold, not merely RGB
  payload coverage. Acknowledgement windows greater than four reports match
  burst throughput; smaller windows and strict serial reply handling are slow.

## Shared projection and selectable source geometry (2026-10-04)

- Projection is now field-level configuration shared by every engine. Torus
  uses inner/outer/light radius ratios; sphere, Cartesian, spirograph,
  vectorscope, and manual placement use outer/light because they have no
  intrinsic inner hole.
- Physics now has three genuine managed-dt geometries rather than display-only
  labels: the established toroidal Laplace--Beltrami Green system, a spherical
  longitude/latitude metric with the spherical Laplace--Beltrami Green kernel,
  and a softened Cartesian inverse-distance system. Accepted sphere states
  transition across the latitude chart poles while preserving their physical
  point and tangent velocity; periodic coordinates are kept bounded.
- Every spirograph source owns a persisted angular offset and angular-offset
  rate in addition to its own ratio, hue, hue rate, and strobe controls.
- A fourth `manual` engine exposes the same spectral field deposition as the
  other modes. Its numbered emitters are directly draggable, inverse-mapped
  through the shared source rotation and projection scale, and autosaved at
  their released positions.
- The edited Python modules passed bytecode compilation. No hardware run or
  test suite was performed by the agent.
- The sphere and Cartesian symbolic RK4 laws each successfully materialized
  as the expected five eager pieces (four stages plus endpoint) before handoff.

## Deliberate angular aliasing and manual colour-wheel deployment (2026-10-04)

- Whole-field source rotation, per-array receiver rotation, spirograph path
  speed, each spiro source's additional angular rate, and every engine's hue
  angular velocity now each have a persisted `1x..100x` multiplier. The
  original 4096x draft was immediately reduced at the user's request, and
  persisted legacy values are clamped during load.
  Multiplication happens after the ordinary fine-grained rate control; the
  spiro base remains Nyquist-bounded while the multiplier deliberately permits
  operation beyond that bound.
- Manual emitters are no longer tied to the three compiled physics body slots.
  A persisted integer deployment count from 1 through 128 and one explicit
  wipe/deploy button replace the manual scene with equally spaced spatial
  emitters whose hues are equally spaced around the colour wheel. Each result
  remains independently draggable and retains its own hue and hue-rate value.
- Current controller calibration and manually dragged locations remain valid
  through fallback loading; the normal startup autosave migrates them to the
  extended configuration shape.

## Native timed strobe and instantaneous fan RPM (2026-10-04)

- Manual emitters now have the same per-object strobe waveform, rate, and beat
  polarity controls as physics and spiro sources. All four engines expose the
  new `timed` waveform; vectorscope bands intentionally share its existing
  common strobe envelope.
- Timed mode creates distinct full-controller ON and OFF images and queues them
  on the native HID worker at absolute nanosecond-valued steady-clock
  deadlines. Native metrics expose last/max dispatch lateness, queue depth,
  and the most recent complete HID transaction duration.
- A first draft incorrectly used measured continuous throughput to lengthen
  requested timing and snap events to a derived clock grid. The corrected
  mechanism keeps ON and OFF as independent requested deadlines, using
  throughput only to calculate the whole number of source revolutions that
  must be skipped to ensure OFF has completed. It chooses the same phase in
  that later revolution. Each next onset is chained from the prior selected
  revolution with a fresh RPM-derived duration rather than snapped to a global
  time grid.
- `-FanRpm` launches a persistent LibreHardwareMonitor PowerShell worker with
  motherboard/controller support; `-FanSensor` optionally selects an exact
  identifier, otherwise the first positive fan sensor is used. This path may
  require an Administrator launcher.
- Instantaneous RPM determines a period but cannot reveal absolute rotor
  phase. The implementation maintains predicted phase from scheduled periods;
  genuine phase lock would require a timestamped tach edge.
- The native DLL rebuilt successfully after adding timed dispatch. No hardware
  strobe or sensor access was performed by the agent.

## Single-owner HID dispatch correction (2026-10-05)

- The first timed-dispatch implementation launched every scheduled ON/OFF
  transaction in a detached thread. Those threads raced `hid_write` and
  `hid_read_timeout` on one HIDAPI device handle, producing Windows overlapped
  I/O abort `0x000003E3` and sometimes preventing the exit all-off frame.
- Timed ON/OFF requests still retain absolute deadlines and deadline ordering,
  but one native worker now owns every HID transaction. A busy controller makes
  a following event measurably late instead of starting concurrent I/O on the
  same handle. The existing timing metrics report that lateness.
- Exit now cancels queued scheduled events before submitting the final black
  frame. The worker is joined before the HID handle is closed, with no detached
  transmitters left behind.

### Prompt history addition

> "let's fix our program's behavior then, we're at 30hz and we should probably be happy there"

## HSL controls, direct colour engines, and FFT-band envelopes (2026-10-05)

- Added persisted saturation and HSL lightness controls to every physics,
  spirograph, manual, vectorscope, uniform, and direct-site colour source.
  Field sources are still converted into the ray-tracer spectral basis before
  accumulation; direct modes deliberately bypass field accumulation.
- Added `uniform`, one object per configured light array. Its versioned
  `ASRockHIDUniformColor` ctypes/C ABI queues one native `SET_ZONE`/`STATIC`
  transaction through the existing single HID worker. The UI shows the
  configured 65-byte offset/value/meaning packet table.
- Added `sites`, one independently controlled object per configured light
  address (eight on the cooler ring and four on the motherboard strip). It
  uses the established full direct-frame `E3`/`E4` packing and shows that
  packing in the UI. An early interpretation incorrectly treated the user's
  word “zone” as a Polychrome physical-zone ID; it was removed in favor of the
  explicit term “site.”
- Added `audio` beside `beat`: each source chooses a logarithmic fftfree band
  and averaging-window duration, and the measured window mean directly drives
  its strobe envelope. The live status reports these levels.
- Kept the 30 Hz producer, latest-wins frame mailbox, and single native HID
  owner. Added only byte-identical pending/in-flight/resident coalescing plus
  per-array uniform coalescing; no alternate scheduler or I/O race was added.
- The active 30 Hz studio held the production DLL open during the final link.
  It was deliberately left running. The same source linked successfully in
  `build_verify`; because the source remains newer than the production DLL,
  `run_field_studio.ps1` will rebuild the production copy after the current
  studio closes. The launcher stops OpenRGB if present and uses the native HID
  transport; OpenRGB is only a legacy folder name and protocol provenance.

### Prompt history addition

> "okay, we need a few things. all color adjustments should include another row, for a saturation and lightness sliders. look for subtle but powerful ways to streamline and keep live without under the clock allocation the pipeline of messages to the HID, and finally let's add one more mode, uniform, where you have one object, that object representns every light, there is no field accumulation it's just object = light command for all lights, and we use the uniform message style with the message struct, sending signals that all lights change to x, and then we have all our levers bells and whistles for that one object so it can be dynamic and be fed data. then lastly, there should be a new strobe setting, \"audio\" which distinguishes itself from \"beat\" by allowing the selection of a band to pull from fft on the audio signal and report the level to some window average"

> "I thought we were already doing that, don't lose sight of the tasks you have in order to change too completely our methods that it interrupts work"

> "the uniform mode should maybe, if feasible, organize into a nice table the data packet for the HID that is configured, giving one glance instruction on the byte packing for anyone using the program, and it should have a sister node using all the same features, but with 8 zones, it will be like uniform in that it will be a completely straightforward byte packing, but it will use the explicit per-zone control byte packing and have as many objects as there are light zones reported for the device"

> "there's a huge vocabulary problem, when I said zone i ddid not mean the HID \"zone\" I meant the per led byte packets in the free or open or whatever mode we use for individual lights, regions, not zones"

## Native MCU strobe and typed message ABI (2026-10-04)

- OpenRGB's ASRock Polychrome USB source confirms native Strobe `0x03`, Direct
  `0x0F`, speed `0xFF` slowest through `0x00` fastest, and Audio/ARGB3 as
  physical zone `0x07`. A native effect is one `0x10` zone transaction rather
  than a host ON/OFF stream. Optical frequency and pulse width remain empirical
  properties of the MCU.
- Added a bounded ARGB3-only probe with the explicit speed allowlist
  `E0 C0 A0 80 60 40 20 10 08 04 02 01 00`. It captures raw zone state,
  restores it on normal/Ctrl+C/error exits, retains a recovery file after a
  hard termination, never selects another zone, and never sends commit `0x12`.
- Added `ASRockHIDExternalZoneEffect`, a versioned 16-byte ctypes structure,
  and the matching C header. Nested raw color bytes, size, version, and every
  C field offset are explicit. The established compiler ctypes interception
  can consume this live layout without another layout description.
- Python now uses structured read/write exports. The former scalar exports are
  compatibility shims over the same message implementation. The native DLL
  rebuilt successfully and edited Python modules passed bytecode compilation;
  no HID write or optical experiment was run by the agent.

### Prompt history addition

> "can you write us up a ctypes python structure for our python, c, and compiler between the two needs that gives us nice oop feel for the messages involved in this abi"

## Deeper public Polychrome protocol record

- The original `aedalzotto/asrrgb` NUC121ZC2 reverse-engineering document says
  native effects are independently configured per physical controller region,
  but exposes no nested region, segment, or LED selector inside one region.
  `ALL=1` copies one common mode/color/speed to every region.
- Its Music-mode capture resolves an important ambiguity: Polychrome Sync reads
  operating-system audio, calculates an intensity byte, appends it to mode
  `0x06` writes, and transmits every 50--60 ms. Music therefore is not MCU-side
  audio acquisition and does not remove the host audio path.
- Music ignores speed. The normal native packet and readback carry only one RGB
  triple per physical region. OpenRGB marks Strobe as per-LED-color-capable but
  its native update passes only the region's first color; no public source found
  a native per-LED palette or per-LED mode table.
- The older libusb record uses 64-byte endpoint packets and places `ALL` at raw
  byte 9. Current OpenRGB uses a 65-byte HID report and places `allzone` at byte
  `0x10`. This is a real source disagreement, not a field to guess around;
  Music intensity placement on this board requires a vendor capture or binary
  inspection before implementation.

### Prompt history addition

> "i need you to do some web searching deeper into the hardware device's abi/api for more answers online"

## Uniform mode-rate and Sites rotation correction (2026-10-05)

- Reconfirmed from OpenRGB's current `WriteZone` implementation that HID byte
  `0x10` is `allzone`: it applies one mode/color to every physical controller
  region. It does not mean every address inside the selected ARGB region.
  Cooler-only Uniform must therefore retain physical selector `0x07` and
  `allzone=0`; STATIC mode itself fills that selected chain.
- Uniform remains a native `SET_ZONE`/STATIC mode packet. It was not converted
  to the E3/E4 direct framebuffer after the wording was resolved.
- Added a global live controller-output-rate slider. It changes the shared
  producer frequency for all modes so rapid mode-write behavior can be checked
  without pretending output rate belongs to the Uniform source.
- Restored the existing receiver light angle/rate/multiplier controls in Sites.
  Ring output is snapped to exact `360/N` address angles, with a separate
  continuous phase accumulator so slow rates still cross integer steps.
- The drag-to-align ring snap now uses the configured address count instead of
  a hard-coded eight.

### Prompt history addition

> "I thought we were done with openrgb I thought we just sent the packets"

> "the live uniform path is supposed to use the all flag, no? it's supposed to send mode packets? I need you to confirm you know what you're doing because you just were talking about methods of sending"

> "oh good call it's all controllers, the vague wording on this field is killing me"

> "maybe give me a slider on output rate so I can tweak it, maybe the full site deployment can get interrupted if it's messaged at 30hz"

> "don't put it in uniform that's a global, just like rotation should be considering the integer snap on sites"

> "there are already rotation controls you decided to remove them from sites and I keep telling you to integer round them and I can't fucking imagine why you haven't finished already"

### Follow-up correction

- The first Sites restoration snapped only receiver geometry. It did not alter
  the per-address payload and therefore could not rotate the physical device.
  Sites rotation now permutes source colours into controller addresses using
  the configured ring direction; the viewer reads those resulting address
  colours at the calibrated receiver positions.
- Uniform now freezes both spatial rotation integrators. Hue velocity remains
  a colour operation and is intentionally unaffected.
- Uniform gained a separate native hardware-mode button. The typed uniform ABI
  and C validator accept the documented Polychrome effect range `0x00..0x0F`;
  the UI cycles effect modes `0x00..0x0E`, while `0x0F` remains the internal
  transition back to direct streaming. This hardware effect is distinct from
  host-side source strobe modulation.

> "sites rotating is not updating the device"

> "also while rotation is irrelevant to uniform, you still have rotation being applied in that mode. lastly, let's add a mode button like the strobe one for hardware mode, since that only works on uniform messages, it would let the user swap between strobe mode, audio mode, etc."

### Per-region Uniform coalescing defect

The saved state had both the cooler and motherboard fields in Uniform, which
proved that no controller-wide E3/E4 direct frame was present during the
reported cross-device colour glitches. The native coalescer required the whole
worker and every Uniform queue to be idle before recognizing a zone's resident
mode. A changed cooler message therefore re-enqueued the unchanged motherboard
message, which could then re-enqueue the unchanged cooler message: cross-zone
mode-packet ping-pong. Resident-mode coalescing is now scoped by physical zone.
Unrelated-zone activity no longer invalidates it; an active or pending command
for the same zone and any controller-wide direct frame still prevent the drop.

> "it is so curious, I tested it on the motherboard lights and the cooler, in both cases the uniform message can cause seemingly random moments of wrong-color in isolated or total sites"

Uniform submission was then tightened at the producer boundary. Python caches
the semantic `(physical region, hardware mode, packed RGB)` signature and calls
the native ABI only when it changes. Thus ordinary Uniform operation does not
depend on the C coalescer to discard one redundant submission per render frame;
native coalescing remains a defensive guard for ABI callers and races.

> "there is nothing for the condensor to do in uniform are you faking uniform by sending multiple light messages or using a total deployment message and ignoring the task of using actual uniform setting"

### Manual 128-emitter deployment crash

Manual's deployment slider correctly allowed 1--128 and configuration
validation correctly required every per-emitter property array to match the
position count. The deployment function, however, generated the requested
ring of positions only behind an impossible `engine == "vectorscope"` branch.
From Manual mode it therefore retained the old positions while replacing all
colour and strobe arrays, and autosave rejected that malformed transient state.
The branch is now unconditional for every count above one. An in-memory
128-emitter deployment produced 128 entries in the position array and all
eleven parallel property arrays; both edited Python modules compile.

> "when i selected 128 divisions of the color wheel in manual the app crashed:\n\nPS C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell> .\\run_field_studio.ps1 -Fps 30\npygame 2.6.1 (SDL 2.28.4, Python 3.11.7)\nHello from the pygame community. https://www.pygame.org/contribute.html\nerror: field 'ring-field' has invalid manual emitters\nPS C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell>"

### Geometry switching and normalization controls

- Sphere and Cartesian managed states correctly omit torus-only columns that
  their compiled laws do not consume. `_body_coordinates` incorrectly read
  `major_radius` and `minor_radius` before branching on geometry, so changing
  away from Torus crashed the viewer. Those reads now occur only for Torus.
- The existing `sqrt(N)` source-count normalization is now independently
  selectable in Physics, Spirograph, Manual, and Vectorscope. It is applied at
  the common source-strength boundary before spectral accumulation. Existing
  non-vectorscope behavior remains unnormalized by default; Vectorscope keeps
  its prior normalized default. Uniform and Sites remain direct packing paths
  and therefore do not show an inapplicable normalization control.

> "the normalize on off needs to be on every mode that uses normalization, and changing geometry in physics led to this:\n\nHello from the pygame community. https://www.pygame.org/contribute.html\nTraceback (most recent call last):\n  File \"C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell\\openrgb_field_studio.py\", line 3435, in <module>\n    raise SystemExit(main())\n                     ^^^^^^\n  File \"C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell\\openrgb_field_studio.py\", line 2958, in main\n    points = projected_physics_points(field_runtime.physics, field_studio)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell\\openrgb_field_studio.py\", line 991, in projected_physics_points\n    for u, v, _old_x, _old_y in _body_coordinates(physics)[:studio.body_count]:\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell\\openrgb_dt_three_body.py\", line 731, in _body_coordinates\n    major = float(field.state.major_radius[0])\n                  ^^^^^^^^^^^^^^^^^^^^^^^^\nAttributeError: 'PieceState' object has no attribute 'major_radius'\nPS C:\\dev\\Powershell\\speaktome\\tools\\openrgb_sdk_shell>"

### Perceptual color profile

The shared spectral color boundary now uses the existing compiler-owned IEC
61966-2-1 sRGB constants from `src.compiler.spectral_colorimetry`. HSL control
colors are decoded to linear light before nonnegative fitting to the eight-band
ray-tracer basis. Spectral deposition, source overlap, and exposure stay in
linear light; the final LED drive is then sRGB encoded. The symbolic spectral
display law uses the same piecewise transfer, so the managed compiler and NumPy
studio paths have one profile. Direct Uniform/Sites HSL colors are already
sRGB-coded and are not double encoded. Numeric black, breakpoint, midtone, and
white round-trips were checked; edited modules compile. Full symbolic law
construction was stopped after it remained busy for bounded observation, with
no exception emitted.

> "let's put in a perceptual color profile on the color science"

### Helmholtz colour-law boundary

The colour controls now converge through a named Helmholtz law instead of
growing independent per-mode conversions. The honorary equation catalogue has
ten new discovered equations covering Planck spectral radiance, arbitrary
spectral-distribution integration against CIE matching functions, xyY white
points, XYZ-to-linear-sRGB, and the exact piecewise IEC output transfer. The
runtime accepts HSL, HSV, blackbody, named white points, sRGB, linear RGB, XYZ, or a
native eight-band SPD and returns both linear RGB and the shared spectral
representation.

Physics, Spirograph, Manual, Vectorscope, Uniform, and Sites persist a colour
system for every independently controlled source. The UI cycles HSL,
blackbody, and white. Blackbody exposes 1000--40000 K; white cycles E, D50,
and D65. Saturation and lightness remain common controls. Planck spectra use
the public `spectral-analyzer` sampler and remain spectra through field
accumulation; they are not replaced by a hue surrogate. Existing configs load
as HSL through defaults. Syntax compilation, default/current configuration
validation, the three public colour systems, and honorary-group discovery were
checked without starting the hardware studio or compiling its symbolic engine.

> "can you not make things cheaper, just deal with the compile cost. sit on your hands I don't know. also give us a blackbody  scale and a white color chooser, three different color systems"

> "you know maybe we make a light and color theory section in the honorary equations, name the group after something, and then it will just be color law and parameters will pick the system used and it will accept just about anything and then it can convert the linear on the way out"
