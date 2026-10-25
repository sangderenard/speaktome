# Force-driven vehicle chassis pose

**Date/Version:** 2026-08-25 v1

## Overview

Audited and replaced the Living Data Map vehicle's target-speed shortcut with
a wheel-torque/contact-force drivetrain. Added persistent roll/pitch/yaw and
wheel-spin state, compiled chassis force/torque integration, full-pose mesh
presentation, a stick/ball chassis diagnostic, and an isolated Spectral-style
packed BVH compiler utility.

## Prompt History

> okay, I think, the vehicle needs pitch and yaw and so maybe that's a state channel we need to keep around, it doesn't naturally climb but the springs do register the unequal compression as if held level, but for vehicles and the player we need something a little more advanced. you should be able to find some code in turing for a spring system that has parameters for damping and spring coefficients and all manner of things, we could compile it for the chassis and let it be real stick and ball geometry, but it's got to crawl well. then I\"m only seeing max spin and then grip when it's not moving, so that makes me convinved we're not working with the right order, we should be hitting a gas petal, not dictating movement and if it spins it should be spinning kinetic friction which it might be because it feels kinda nice. this whole thing feels suspiciously nice it makes me feel like it's gotta be a trick and not my compiler and math core and entire system. now. do you think you can see to these issues?
>
> you're mentioning some of this i see, but just know we have a lot of great physics in the repo all over the place, spectral analyzer has great bvh/ray material if you could isolate it. the only key is
>
> make sure you use the compiler to turn magic from pure math or python or any convenient language into something real that works on the web, you know?

## Steps Taken

1. Audited the worker and confirmed the contact slip used commanded target
   speed while the scalar chassis Wasm independently accelerated toward it.
2. Replaced the controller with JSON drivetrain torque, wheel inertia, braking,
   rolling resistance, front/rear drive split, and steering-angle parameters.
3. Added four wheel angular velocities and roll/pitch/yaw plus angular velocity
   channels to the compiled chassis ABI and recycled snapshot.
4. Made tire slip use contact-point velocity (including angular velocity) minus
   wheel surface speed and made steering rotate front tire tangents.
5. Reduced four compiled contact force/torque lanes into the compiled chassis
   Wasm within the existing non-overlapping worker tick.
6. Published four force-bearing chassis nodes and six rigid members, rendered
   full chassis pose, and added chassis structure status to the HUD.
7. Isolated the packed BVH layout from Spectral Analyzer into a deterministic
   Python compiler utility without coupling physics to optical materials.
8. Corrected the UI/physics steering handedness while retaining front-axle-only
   steering, raised suspension and angular damping, and added a parametric
   compression-rate clamp so contact acquisition cannot create a damping spike.
9. Replaced the single suspension damper with compiled directional pneumatic
   compression/rebound damping based on Turing's existing classic-mechanics
   contract. Added per-wheel slip history and compiled derivative-sensitive
   throttle traction control and brake ABS, with snapshot/HUD telemetry.
10. Added an 8 m thick world-bottom rejection volume to the shared contact
    contract and common worker post-step, covering platformer, projectile,
    vehicle, and rigid bodies without inventing a vehicle-only floor rule.
11. Added JSON-parametric world tiling and chase-camera response, a WebGL2
    camera-depth prepass, and four state-driven procedural wheel meshes. These
    consume lockstep chassis/wheel state but remain presentation-only adapters.
12. Repaired the resulting mount-time presentation regression by separating
    dynamic GPU uploads from static collider/DOM/portal publication. Moved
    round-wheel spin to a shader tread phase, added world-custody inventory and
    map-marker lifecycle, and retained free-look/tool channels while mounted.
13. Found the remaining invisible-vehicle root cause: wheel construction used
    an undefined `config` before appending any vehicle geometry. Corrected the
    scope and put body, cabin, steel frame, load-colored struts, and round
    tread-phase wheels into the guaranteed main scene mesh.
14. Replaced authored forward/reverse torque constants with a strict JSON
    engine/powertrain configuration and compiled an explicit engine, clutch,
    transmission, final-drive, front/rear differential, and half-shaft torque
    graph. Engine position/orientation, rotating inertia, mass, displacement,
    BMEP, ratios, and efficiencies now affect compiled outputs and chassis
    reaction torque.
15. Extended the lockstep snapshot with fourteen powertrain channels and made
    a mechanical cutaway (low floorpan, roll cage, engine, gearbox, shaft,
    differentials, half-shafts) plus HUD torque graph consume those channels.
    Corrected the presentation-only axle inversion: the worker already steered
    the front named lanes, while rendered front/rear local-X signs were swapped.
16. Replaced the courtyard gradient-test wedge with a 49x33 sampled mud-oval
    terrain. Its inner band is a physical trench and its adjacent outer band is
    a berm; the rendered two-triangle cells and shared platformer/vehicle
    piecewise-planar sampler use the same 1,617 height samples. Published the
    standalone generated page from an isolated `gh-pages` branch at
    `https://sangderenard.github.io/Turing/`.
17. Strengthened terrain legibility with world-grid and per-cell height/checker
    coloring, added the slot-10 depth-map tool (lower/raise and middle/texture
    growth modes), and added worker-authoritative `RIGHT CAR` recovery. Reduced
    and raised drivetrain presentation parts, corrected visible axle-member
    labels, and changed crank/mount reactions to be explicitly resolved and
    accumulated in chassis-local coordinates. Published Pages commit `694b209`.
18. Fixed sampled-terrain crest tunneling by replacing the worker's hard
    `bodyY + 0.08` surface ceiling with declared suspension reach and enforcing
    a four-corner travel-stop constraint after each compiled pose. Removed the
    rendered floorpan and seat blocks; retained silver frame/cage members,
    yellow suspension, black shaft-like drivetrain members, and explicit engine
    and transmission mount crossmembers. Published Pages commit `f3969bc`.

## Observed Behaviour

- Sixty-four focused vehicle and Living Data Map tests passed, including direct
  execution of emitted vehicle Wasm proving throttle spins wheels rather than
  assigning chassis speed, and proving a worsening high-slip wheel independently
  reduces throttle and brake authority.
- Direct emitted-Wasm execution also verifies clutch loss, gear and final-drive
  multiplication, engine angular acceleration, and the configured 42/58 axle
  differential split.
- The generated JavaScript passed Node syntax validation.
- The live arbitrary-mesh BVH traversal and physics-material path remains
  explicit follow-up work; analytic gradients and box tops remain live today.

## Lessons Learned

A convincing speed servo can hide a broken drivetrain ordering. A gas pedal
must influence axle torque, not chassis velocity or a synthetic tire target.
The correct lockstep invariant is a single state owner and a joined tick
barrier; the presence of async GPU APIs neither proves nor disproves useful
parallelism. For a crawling vehicle, rigid chassis members plus compliant
suspension are more stable than an unnecessarily deformable chassis.

## Next Steps

- Compile BVH traversal and triangle closest-feature contact behind the shared
  platformer/projectile/vehicle contact-surface ABI.
- Compile the same contact SSA to batched Wasm for WebGPU-unavailable parity.
- Add GPU-resident force/torque reduction and measure it against per-tick CPU
  mapping before changing scheduling.
- Replace circular chassis side contact with an oriented multi-contact solid
  manifold and eventually move unrestricted tumbling to quaternions.
