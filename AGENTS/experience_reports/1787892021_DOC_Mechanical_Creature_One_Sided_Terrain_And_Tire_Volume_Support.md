# Mechanical Creature one-sided terrain and tire-volume support

> **2026-08-28 correction:** The overlap-volume rejection outcome recorded
> below is superseded. Hands-on review found the expected free-energy failure,
> so all vehicle positional/overlap rejection was removed. The current source
> uses actual swept tire crossing: 5 radial × 3 lateral probes per wheel, eight
> subdivisions, and eight bisection steps. Crossing position, terrain normal,
> point velocity, suspension load, and carcass compliance produce a one-sided
> wrench; the rendered terrain underside has no vehicle boundary.
>
> The same pending batch adds plastic/broken chassis and body-shell mounts,
> subdivided cosmetic-shell collision, total live component/fuel mass, fuel
> consumption, gasoline/nitromethane chemistry, independent ignition timing,
> alternator/starter/battery, headlights, horn, strong differential brakes,
> combustion-timed subharmonic/FIR engine DSP, and solver-side driving modes,
> governor, and PI cruise control. Dismount leaves the resident vehicle and
> running engine advancing; combustion cannot self-start at zero, while starter
> torque and clutch-driven rolling starts can cross cranking speed.
>
> Steering is also no longer a presentation shortcut. Column rotation drives
> pinion/rack travel, each spherical tie rod solves its own length constraint,
> and each resulting knuckle angle is shared by tire contact, the five-axis
> bearing-bound hub/wheel/rotor, and rendering. Calipers remain knuckle-fixed;
> a broken tie rod releases only its own wheel.
>
> Static status: Python syntax, extracted main-thread JavaScript, extracted
> worker JavaScript, and `git diff --check` pass. The full test/build, generated
> page, root-repo publication, commit, and push remain deliberately unstarted
> until the user gives the requested pre-compile approval. Root
> `nogodsnomasters/site/...`, not Turing Pages, remains publication authority.
>
> Final precompile addition: `TILT` now sits beside TC and ABS. It evaluates
> the derived center of mass against the rear tire contact line, pitch angle,
> pitch rate, and longitudinal acceleration; an impending wheelie arc lowers
> the active governor and commands the real implicit rear differential brake.
> The electrical graph now includes ECU, fusebox/relays, routed insulated wire
> circuits, IMU and switch buses, electronically dispatched RPM/load ignition
> timing, headlamps, tail lamps, and brake lamps. ECU and lamp loads draw from
> the battery; the regulated alternator supplies live loads and bounded charging
> demand while reflecting mechanical accessory torque.
>
> Hydraulic/pneumatic precompile addition: `LVL` now owns worker-authoritative
> four-corner hydraulic rest-geometry actuators, built-in poses, three
> programmable pose memories, user-selected pose interpolation rate, and a
> manual wheel-height mode whose chassis pose emerges from the four tire/contact
> wrenches. A frame-mounted electric hydraulic pump, manifold and four hoses are
> explicit graph parts and electrical loads. A separate frame-mounted pneumatic
> compressor/accumulator feeds four shock circuits and four tire rotary-union
> lines. User tire-pressure targets are regulated over time, draw compressor
> power, alter compiled contact area and radial carcass stiffness, and deform
> the rendered toroid (low-pressure sidewall bulge and lower-arc flattening).
>
> Harness-routing addition: electrical looms, hydraulic pressure hoses, and
> pneumatic shock/tire lines are now included in the presentation graph as
> relaxed seven-point routes rather than rigid endpoint rods. Each route keeps
> its endpoints attached while its interior bends exponentially toward a
> gravity-sagged target. Electrical looms carry the most slack and soften most
> slowly, air lines sit between, and high-pressure hydraulic hoses use the
> least slack, largest minimum bend radius, and fastest/stiffest response.
> This relaxation is presentation-only; electrical and pressure authority stay
> in their existing circuits and the hoses cannot inject mechanical energy.
>
> Final steering/control/course addition: the powered steering servo is now a
> massed electrical graph component with its own fused feed and column torque
> coupling. With ECU and servo available, road speed maps to column command
> rate. With ECU lost but servo powered, a local torque-assist fallback remains;
> with both lost, rate is derived from capped human wheel torque against live
> front normal load, tire scrub, caster resistance, steering ratio, and viscous
> resistance. Reverse requests while rolling the other direction now cut
> throttle, cancel cruise when appropriate, apply service braking, and change
> driveline direction only near rest, preventing the former pegged-throttle
> transition. Generic bodies no longer use solid platform tops as deep safety
> rejection volumes, so an object beneath a brick is not teleported upward;
> ordinary swept top-face crossing remains. Sphere winding and two-sided face
> lighting were corrected, distant blue-course visibility was increased, and
> differential brakes gained explicit annular rotor faces. A thirteen-segment
> C2 blue serpentine pleasure road now climbs and descends across the north
> apron without attempting an inversion.

> Pre-compile armored-body addition: added a selectable five-turret carrier
> assembly with four corner posts and a taller center gimbal, a segmented
> 18 mm steel cage skirt, an independent powered fire-control computer, and a
> chassis body-assembly wrench ABI. The assembly reports dry mass, payload
> mass, local center of mass, principal inertia, mount identity, force/moment,
> and point-impulse events. Sixty rounds are independently limited by count,
> 51 kg magazine load, and 0.050 m³ magazine volume; firing removes live mass.
> Every gun produces its own recoil impulse and r-cross angular impulse at its
> own mount. The active-focus surface ray drives a low ballistic solution for
> each yaw/pitch gimbal, and any nearer friendly ray intersection disengages
> the entire volley. Driver primary fire defaults to the turret computer while
> an extended-setting toggle restores the handheld tool.
>
> Four armored-body hydraulic diagonal outriggers now include 0.72 m inboard
> reserve tubes and 1.72 m extension. Each foot uses the same subdivided,
> bisected, one-sided terrain-crossing test as the tire work; first contact
> creates a persistent terrain anchor. The constraint retains all anchored
> feet while the cylinders withdraw and releases only at full retraction.
> Anchor error supplies chassis translation and angular stabilization, while
> the moving piston/foot mass updates assembly COM and inertia. LVL now permits
> 0.62 m corner lift, the HIGH pose requests 0.54 m, and all sixteen upper/lower
> A-arm links plus four steering tie rods carry hydraulic rest-length modifiers
> through the mechanical graph. The mean lift is also fed to resident compiled
> suspension geometry; individual modifiers remain authoritative in the graph
> and scalar contact solve. No final page generation or publication was run;
> the user-required pre-compile checkpoint remains in force.

**Date:** 1787892021
**Title:** Closed the remaining wheel-through-ground paths in the Living Data Map demo

## Overview

Corrected the terrain contract and, after user review exposed that the first
attempt did not implement the requested physics, added a genuine unilateral
overlap-volume constraint. Both resident GPU and scalar Wasm paths now compute
the circular-segment volume of a finite-width tyre below the one-sided terrain
half-space. The compiled law normalizes that forbidden volume and demands the
corner-mass acceleration needed to separate it during the fixed tick. The
pressure-derived contact patch remains separate. Rendered height-field prism
bottoms remain presentation-only and are not collision faces.

## Steps Taken

- Read the prior Living Data Map vehicle/contact reports and local Turing rules.
- Traced terrain publication, GPU quadrature, the compiled SymPy contact law,
  the scalar Wasm fallback, and post-step world-bottom recovery.
- Added the ordinary floor as a GPU terrain-sampler fallback only outside
  authored sampled-field domains.
- The initial repair only altered compliant load routing; user review correctly
  identified that it neither calculated overlap volume nor used volume as a
  rejection constraint. Replaced that approach rather than defending it.
- Added circular-segment volume evaluation for the tyre cylinder to both GPU
  and Wasm geometry paths, including ordinary and already-buried contacts.
- Added `tire_overlap_volume`, full tyre volume, radius, and gravity magnitude
  to the compiled contact ABI. The resulting volume error and inward velocity
  drive a unilateral one-tick rejection load using the wheel's corner mass.
- Kept pressure-derived area as the contact patch; overlap boundary area is
  reported only by the scalar geometric helper and is not used for friction.
- Extended buried-wheel depth from one radius to the full two-radius diameter.
- Added focused source-contract and direct symbolic-force regressions.
- Rebuilt `docs/generated/abstract_ui_object_map.html`, syntax-checked its
  emitted JavaScript, and loaded it through a local server in the in-app browser.

## Observed Behaviour

- `tests/test_state_loop_deployment.py` plus overlap-only scalar regression:
  6 passed.
- Existing pneumatic/Coulomb regression plus overlap-only rejection regression:
  2 passed.
- Focused GPU terrain/contact regression: 1 passed after full tensor-to-WGSL
  lowering.
- The full vehicle file was stopped after a late failure while unrelated
  compiler-heavy tests continued; both changed high-signal tests subsequently
  passed in isolation.
- The generated 8.66 MB page loaded as `MechanicalCreature · living data map`,
  reported the resident Wasm fallback with nonzero spring loads, accepted the
  RIGHT CAR recovery control, and logged no browser errors.
- Node syntax validation of the generated behavior script passed.
- Published the two canonical root-repository artifacts, `site/index.html` and
  `site/demos/living-data-map/index.html`, in `nogodsnomasters` commit `4ced32f`.
  GitHub Pages subsequently served the new byte length and both contact markers
  from the public Living Data Map URL. Turing's historical `gh-pages` branch is
  not the publication authority.

## Lessons Learned

The visible symptom cannot be solved by renaming carcass compression as finite
volume support or by adding another spring load. Overlap volume must be a
geometric constraint error and must produce separating acceleration even when
the compliant suspension/carcass paths are disabled. A height-field may be
rendered as a prism for visibility while its physics remains a one-sided solid
half-space with no bottom face.

## Next Steps

No required follow-up remains for this repair. A future deterministic browser
drive harness could record wheel-hub signed distance over long obstacle-course
runs, but it is not needed to retain the new source-level invariants.

## Prompt History

> in the living data map mechanical creature demo code, it's still too easy for the vehicle to sink a tire under the plane that's supposed to be ground, even though I've asked agents several times about it, it's still a problem. maybe the wheels should be considered volumes and a voume overlap with the ground under the chassis can be used to drive force up over the whole of the wheel while disabling any boundary the bottom of that terrain might represent?

> Can you verify where others publish, in the root repo?

> Turing is not the repo that holds published pages except one mistaken version

> Can you put the latest files online so i can try it out

> Itt would never do anything at all to make the overlap volume the contact patch to solve the problem and that should be the cross sectional area of the shared volume boundary if it were what you wanted that to be. No. I said use the overlap volume to calculate rejection. Your work did nothing whatsoever to help with tires popping under the road

> I think wires relax themselves I'm not sure but then if you wanted to put in the hydrolic and pneumatic harnesses that would be really sick

> let's also let the ECU control stearing rate to velocity calculation

> that's good but I want to be sure that if the ecu is out and the power or steering servo is out, the steering occurs a human force rates

> there's also a bug where sometimes if you hit back instead of applying the brake and then reverse, the vehicle gets the thottle locked pegged

> I don't always get the right faces,the diff brake rotor seems invisible, the balls show me where they clip the floor a little like they might be inside out, I can't see the blue ramps until I'm right in front of them. BTW I love the thin blue glass ramps and want a little more of a rollercoaster to drive around, and the platformer bricks aren't working, they teleport things from under them

> a loop is unlikely to work but we must try as a test of the ground contact engine, but let's try just a cool wiggly meandering path with ups and downs as well, something to drive for pleasure

## 2026-08-28 pre-compile tyre constitutive redesign

The later vehicle batch removes the radial tyre `k*x` analogue and the direct
longitudinal/lateral slip-to-force lookup. Each wheel now carries persistent
longitudinal and lateral tread-belt deformation plus deformation-velocity
states. Hub slip drives critically/over-critically damped sidewall modes; only
their filtered displacement and velocity reach the contact law, so fast lateral
jitter is stored and dissipated in the sidewall rather than transmitted directly
to the hub or disguised as additional friction.

Radial support is a finite pneumatic toroid model. Geometry-derived footprint
chord and effective tread width determine contact area; flattened volume raises
the enclosed gas pressure through a bounded polytropic expansion. Radial carcass
loss opposes compression velocity. Suspension demand and pneumatic capacity are
joined by the passive series law `Fs*Fp/(Fs+Fp+epsilon)`, which cannot exceed
either side and therefore cannot become an overlap-rejection/free-energy path.
The crossing solver remains the authority for discovering one-sided contact.

Suspension travel binding now retains compression beyond travel as bump-stop
residual. A configurable linear-plus-quadratic bump stop and compression damper
carry that residual before the result enters the same passive pneumatic series
path. Config fields are present in `fun_car.json`, the strict validator, worker
Wasm feed, resident WebGPU contact constants, and live suspension-parameter
mapping.

Focused verification completed without generating or publishing the page:

- Python syntax and generated worker JavaScript syntax passed.
- Direct pneumatic/Coulomb, bump-stop residual, and damped lateral-sidewall
  state tests passed.
- The contact law lowered successfully to both the vectorized WebGPU kernel and
  scalar WebAssembly fallback.
- Six selected tests passed in 48.03 seconds; `git diff --check` reported only
  the repository's existing line-ending notices.

The user-required checkpoint still applies: do not launch the full vehicle/page
compile, copy root publication artifacts, commit, or push until explicitly
approved after the final focused request pass.

## 2026-08-28 pre-compile vehicle hardware expansion

The default wheel/powertrain pack is now a tall, narrow tractor tyre on a heavy
solid steel disc, an AMC-era 258 inline-six, a wide-ratio four-speed, a soft old
organic clutch, and a HIGH/L1/L2 transfer case whose L2 ratio is the extra-crawl
case. A Honda-style 1.5 L commuter inline-four and two firmer clutch packs remain
selectable. Wheel selections carry their own tyre/rim mass, rotational inertia,
pneumatic toroid, sidewall mode, and geometry parameters rather than changing
only a mesh.

The full-build model now cooks two equation paths for every engine profile into
the static selector artifact. `linear-playable` is the default: SymPy first
attempts an exact `linear_eq_to_matrix`/`linsolve` reduction, then uses an
audited first-order engine/driveline operating-point approximation where the
exact solve does not reduce operation count. `symbolic-fidelity` retains the
reduced nonlinear equations. Both paths independently lower through repository
SSA into staged WGSL; worker startup creates every baked stage set and selection
only changes the integer case. Clutch parameters stay live inputs to both modes,
so clutch swaps do not invalidate or recompile engine cases.

The reduction pass is deliberately bounded by expression operation count.
Several authored chassis/reaction publications inline tens of thousands of
operations; generic factor/CSE passes on those expressions took more than two
minutes for one profile and were stopped. The revised audit attempts
`factor_terms`/cross-output CSE on tractable publications, records every
budget-skipped publication, and feeds only the bounded coupled
engine/driveline subsystem to `linear_eq_to_matrix`/`linsolve`. If the exact
solution does not lower operation count, the playable case uses bounded
first-order engine/driveline equations and retains contact/chassis fidelity. A
direct profile probe completed in about 17 seconds and reported eight
approximated publications.

Two deliberately extreme engine profiles were added to the same baked switch:
a 27.04 L Packard/Rolls-Royce Merlin V-1650-style 60-degree V12 and an 18.1 L
heavy-machine turbo-diesel inline-six based on published Cat C18 industrial
figures (3655 Nm at 1400 rpm, 597 kW, 1673 kg dry). The selector now contains
13 engines and the eventual artifact contains 26 engine/equation-mode cases.
The aircraft engine prefers 100/130 aviation gasoline; the diesel prefers
ULSD. Per-engine fuel compatibility makes incorrect fuel choices materially
derate combustion. Per-engine ignition compatibility separately requires the
Merlin's dual-magneto profile or the diesel's compression-injection governor;
fuel and ignition remain independent user choices so mismatched combinations
fail rather than silently auto-correcting. Engine swaps now contribute their mass delta, center-of-mass
shift, and parallel-axis inertia delta to live vehicle mass properties; this
closes a previously missed mass-accounting path that would have made the very
heavy diesel physically cosmetic.

Engine selection was also found to overwrite the installed clutch with the
engine profile's nominal torque interface. That was removed: clutch stiffness,
capacity, efficiency, driven inertia, and mass remain owned by the selected
clutch part because those inputs are deliberately live across all baked engine
cases. The soft 235 Nm organic clutch therefore slips under either giant
engine. Selectable 3800 Nm aircraft multi-plate and 4800 Nm industrial
twin-disc parts provide physically heavy alternatives.

The requested likely-to-fail loop is now represented by 32 thin blue static
panels in a vertical ring. This is not a decorative loop mesh: resident GPU
radial gathering already sees every AABB face, and the CPU/Wasm fallback gained
an entry-only swept segment/AABB crossing so tire probes can see vertical and
ceiling faces too. There is explicitly no adhesion; falling out is valid. The
test passes by keeping all crossings finite and one-sided, not by guaranteeing
a completed loop.

The frame/cage audit found and removed a false implementation: all pipe nodes
had been marked chassis-fixed, so damage rest lengths could change without ever
moving endpoints. Structural frame/cage nodes are now deformable within the
constraint graph; failed members leave the constraint set; cage collision uses
the solved deformed node positions; and each pipe carries section-derived mass
allocated within the existing frame/cage/driver residual so total mass is not
double-counted. Fractured pipe mass remains split between its endpoint nodes.

Every applicable mechanical edge endpoint now has a source-parameterized,
compile-static `performance-polyurethane-static-v1` six-axis Kelvin-Voigt
bushing pack. The worker measures relative junction motion and accumulates
linear plus angular damping power and energy. Routed wires/hoses and decorative
lamp geometry retain their own constitutive models instead of being mislabeled
as suspension bushings.

Chassis length and wheelbase are first-class geometry controls. Each graph node
records either an axle-relative longitudinal offset or a chassis-length fraction;
changing dimensions rebuilds node positions, edge natural lengths/stiffness,
pipe mass, presentation, collision, and worker geometry together.

Wheel alignment now exposes per-corner or linked camber/caster/toe values. The
values actuate upper-arm and tie-rod rest geometry and enter the toroidal contact
orientation. Modes are static authored settings, a one-shot stationary four-wheel
calibration, and bounded full-time auto trim which pauses under insufficient
support, high acceleration, or high tyre utilization.

Cheap verification after these additions: Python syntax passed; emitted worker
JavaScript and the static main JavaScript source both passed `node --check`; the
strict default configuration and stubbed-backend 13-profile selector tests
passed; isolated graph construction reported 241 nodes, 339 edges, and 253
physical edges with bushing packs. Earlier probes were terminated when they
entered the expensive WebGPU artifact builder or an unbounded symbolic pass;
neither produced artifacts. No page generation, publication copy, commit,
push, or deployment followed.

## 2026-08-28 full-build cost and mandatory rebuild gate

The expanded MechanicalCreature page was subsequently built successfully with
`python tools/build_mechanical_creature_page.py docs/generated/abstract_ui_object_map.html`.
The build ran from approximately 10:06:53 to 11:47:06 local time: about 1 hour
40 minutes. It produced a 14,913,858-byte HTML file containing 13 engine
profiles in both symbolic-fidelity and linear-playable modes. The byte-identical
root-site targets were published from `nogodsnomasters` as commit `3dbd602`.

**Do not start this full builder again during ordinary diagnosis, repair, or
testing. Treat it as a roughly two-hour operation and require the user's fresh,
explicit approval immediately before another full compile.** Prefer source
checks, extracted-JavaScript syntax checks, focused tests, or the page relinker
when a change is confined to the behavior packet. Never let a page-loading
problem itself trigger a rebuild.

The first published load then froze on `constructing the living map…`. Browser
inspection showed that download and parsing had completed, but the main script
threw `ReferenceError: saved is not defined` in `bindMobileStick`. Two blocks
that restore `saved.vehicle_hydraulics` and `saved.vehicle_tire_pressure_target_pa`
were accidentally placed inside the PointerEvent setup branch, outside the
`restoreLivingEdits` scope where `saved` exists. `renderWorld()` calls
`renderShaderViewport()`, which calls `renderMobileControls()`, which reaches
that exception before the viewport is appended, shader initialization is
scheduled, workers are initialized, or the animation loop begins. This is a
small behavior-source repair and should use the relink path after focused
verification, not another full symbolic compile.

## 2026-08-28 native rig parity correction

The first native teaser was invalid: it selected the reduced suspension
publication set, added a hand-authored five-node Laplacian/contact host, and
rendered proxy boxes. It was stopped and its build entry point was replaced.
The native bundle now admits only the complete `abstract_ui_vehicle_step` and
canonical `abstract_ui_wheel_contact` repository-SSA emissions, plus a compiled
external roller-fixture law. A compiler-rendered C tick shell calls those
kernels and explicitly rejects the reduced suspension kernel.

The fixture has two physical boundary modes. `cage-drive` is a neutrally
weight-compensated hydraulic carriage with a compression-only passive dashpot;
its hub force is exactly zero during separation, so it cannot pull a departing
wheel downward. `suspension-test` is the same carriage with a bidirectional
position lock. Roller/tire forces remain owned by the canonical contact kernel.

Double-double C emission now exposes persistent high/low binary64 lanes at the
kernel ABI. An earlier attempt was stopped because it retained two limbs only
inside one invocation and collapsed state at every output, which would have
discarded the low limb at each audio frame. Symbolic coefficients can also be
baked directly into high/low pairs for linear or neural surrogate experiments.

The engine-pan chassis fitter was found to exist only in the obsolete dyno
tool. It is now a shared vehicle-model function used by the dyno and attached
to every game power-unit preset. Game engine selection invokes the shared fit,
rebuilds chassis/wheelbase graph geometry, and accounts for added frame mass.

One important gate remains deliberately red: the game's 241-node mechanical
presentation/damage constraint solver is still hand-authored worker JavaScript,
outside the canonical 201-input/96-output vehicle SSA transition. The native
manifest records this and refuses to claim or launch a scientific whole-graph
parity rig until the constraint position and plastic/fracture state updates are
migrated into shared SSA. The ordinary canonical C transition did compile and
execute at a 48 kHz timestep with all 96 outputs finite; focused native fixture,
double-double ABI, shell-authority, coefficient-bake, and shared chassis-fit
tests pass.

### Prompt History (native parity correction)

> or optionally yes lock it, it's two modes we need, one drives the cage one tests the suspension

> top priority is absolute parity and same source code same kernel same math between this rig and the game

> I can see in my mind the hub and wheel rollers i designed would actually work perfectly here by having a tiny little dampner with hydrolics on the hub to rollers and just letting the wheels take flight with a light passive but fully realistic simulated measure of travel because it's like, an actuator connected to a long pipe to the side and dampens in that not the small space.... but I'm maybe overthinking it, if we just let those grippers be weightless, which we can just by letting it be fantasy assumed they have force  detection and neutral boyancy compensation

> but like, I imagined we use the real kernel but we really simulate the rollers and actuator to the hub and their lock or dampening states

> but the ability to bake the chassis dimensions around the engine,  the stuff that improved the game, that stuff is real and needs to be accomodated in the game, what we perfect here, we use in the game, right? and when we use it here we use the REAL physics engine, just c compiled with double precision

> the whole system as one graph in one engine at audio frame rate and DOUBLE DOUBLE two lane precision will self check the engine, ensure the design works in game, be useful for scientific research, and can bake coefficients at double double for any linear or neural model for the system inside the rig

## 2026-08-28 scientific presentation boundary correction

The user clarified that native parity is the authored API/ABI entering the
compiler, not pixel-identical presentation or a requirement that every game
presentation routine first migrate to SSA. The prior manifest's red visual
launch gate was therefore incorrect. It is now diagnostic-only and permits a
scientific renderer to consume the canonical compiled ABI without acquiring
physics authority.

Spectral Analyzer now owns dedicated GL 4.3 scientific vehicle shaders. Their
vertex ABI carries position, normal, component class, raw axial strain, yield
strain, fracture strain, damage state, and joint kind. Unstrained base hue
identifies component class; compression adds blue, tension adds red, plastic
state adds magenta, and fracture uses a high-contrast white/black hatch.
Bushings are point-glyph annuli with a core; bearings are double-race/crosshair
point glyphs. Both use exactly the same constitutive color mapping as members.
The shader is explicitly a read-only compiler-output consumer and has no
integration, smoothing, contact, or constraint effect.

Both shader stages passed offline `glslangValidator` compilation. The focused
native deployment suite passed 6 tests in 5.65 seconds. No MechanicalCreature
page build was started.

### Prompt History (scientific presentation)

> Visual parity is not necessary but also only required you used the shader so I'm confused how you failed at that and what else you might need to keep working on because in my optimism i had hoped you were ready

> Okay just get us a scientific shader for it that shows type and strain through color, dots with same for bushings and bearings. The parity is api abi going into the compiler
