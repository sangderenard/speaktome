# Mechanical Creature model-readiness audit

**Date:** 1787965531
**Title:** Audited the Mechanical Creature model, inverse route, and native/browser readiness

## Scope

Audited the accumulated Mechanical Creature requests and implementation in
Turing, with particular attention to real tire contact, closed-graph wheel-end
physics, parametric equipment and mass accounting, native/game equation parity,
the tape-free inverse/Adam route, two-limb arithmetic, and publication readiness.
No long Mechanical Creature page build or publication was authorized or run.

## Methodology

Read the current conversation requirements, prior guestbook reports, vehicle
configuration, canonical symbolic equation, worker state loop, native deployment
manifest, inverse compiler, precision implementation, and focused tests. Compared
claimed features with executable authority, separating equation integration,
worker-only behavior, contracts/manifests, and source-presence tests. Ran focused
precision, inverse, native, state-loop, PCM/Wasm, canonical compilation, and
20-second engine-off quiescence validations.

## Detailed Observations

- The authoritative readiness table is recorded in
  `turing/docs/vehicle_readiness_audit.md`.
- The model is the equation. JSON values remain runtime parameters and are never
  frozen by the symbolic-constant high/low splitting helper.
- Variable real powers now execute on the two-limb type through limb-aware
  logarithm/exponential operations. A generated static reverse graph plus
  functional AbstractNN Adam executes as a small tape-free two-limb proof.
- The vehicle equation now accepts three body-frame drag vectors with runtime air
  density, direction, drag coefficient, and reference area. It evaluates the
  signed quadratic drag force along each yaw-rotated vector.
- Generic corner wrench attachments, shock-mounted heavy bumpers, and
  density-sized ballast enter mass, COM, inertia, and the mechanical graph.
- Default damage state is live parametric/no-prebake, and live parameter changes
  retain the same ABI without requiring a kernel rebuild.
- Focused validation is green: the combined precision/inverse/native/state-loop/
  PCM suite accounts for 51 passing tests after correcting one stale source-text
  assertion; canonical vehicle compilation, drag evaluation, attachment/ballast
  validation, and the 20-second gross-mass quiescence test also passed.
- `git diff --check` is clean apart from expected LF/CRLF notices.

## Analysis

The system is not ready for a long page build, publication, or a claim of native
and browser physical parity. Browser contact still uses sampled radial crossing
rather than the native analytic torus contact arc. The canonical chassis equation
still consumes a lumped wheel-end wrench, and differential rotor inertia is not
in an authoritative wheel-end/driveline mass matrix. Mechanical constraint,
plastic/fracture, recoil, bumper, and outrigger authority remains partly in the
handwritten worker instead of shared compiled SSA. The native rig cannot yet load
and rebuild a complete equipment selection, and the canonical vehicle objective
has not traversed reverse graph, two-limb promotion, and AOT C end to end.

## Recommendations

Proceed in the order recorded by the readiness audit: analytic finite-triangle
torus contact in the browser/shared equation; authoritative wheel-end and
driveline mass matrix; shared compiled mechanical/damage graph; native equipment
loader; canonical inverse two-limb AOT; behavioral scenarios; only then the long
browser build, real-browser validation, and root-repository publication. The
continuation is captured in `speaktome/todo/1787965531_mechanical_creature_model_readiness.stub.md`.

## Prompt History

> I've asked for a lot of things and I expect a high degree of accuracy, hidden incomplete or test states hide real important details systems will depend on, I need you to do an audit on the conversation, my requests, and the readiness to move forward. I would like you then, if the siituation is clear, move forward

> sorry that last comment was staged, uh, on the last thing, I meant, float powers you wanted right? and we don't have a double for that? can you make one?

> ONE LAS TTHING and then we are ready I swear, the rig needs basic coefficients of drag for a set of vectors

> continue your audit but keep in mind that has to be fixed, we need those pieces

## Tire soft/hard contact continuation

The follow-up replaced the instantaneous pneumatic normal-force evaluation with
a three-stage implicit-midpoint radial-mode solve inside the shared SymPy
contact equation. The returned terrain reaction is the mode's outward impulse
divided by `dt`. Zero carcass loss conserves local quadratic energy to numerical
precision; positive carcass loss removes exactly the accumulated
`h*c*v_mid^2`. The unilateral force clamp can discard attraction but cannot
create it. Effective radial mass is a live JSON fraction of unsprung mass.

The resident GPU source was re-audited and found already to contain the
continuous concentric torus/local-plane arc. The scalar browser fallback was
the remaining 5x3 radial maximum-penetration path; its authoritative call now
uses the continuous torus arc and swept ring-support crossing as well. Exact
finite-triangle face/edge/vertex branch selection remains outstanding.

The direct lossless/damped energy identity passed. The modified contact kernel
lowered through native C, scalar Wasm, and vectorized WebGPU. Worker JavaScript
syntax and all five state-loop tests passed, the native shell/torus tests passed,
and the 20-second engine-off gross-mass quiescence test remained green. No page
build or publication was started.

> make sure if we can improve the conservation of our penetration response in the tire, even with a little series ring down solve for the tire system's transients, we should do it now before we move, in addition to taking care of anything you have identfied

> if it helps, you're essentially being charged with making a high performance pure math verifiable soft vs hard body collider

## Balloon-skin direction correction

The user explicitly cancelled the analytic torus/finite-triangle collider as
the production direction. The tire must instead be a closed pressurized
soft-body skin. Work began in `turing/src/compiler/vehicle_balloon_tire.py`:
compile-static closed topology, per-face StVK strain energy, Kelvin dissipation,
global polytropic gas law, bead/rim equal-opposite wrench equations, and
deformed vertex/triangle unilateral contact are authored for the existing
SymPy -> process graph -> SSA pipeline. `fun_car.json` now has a parametric
`compiled-balloon-skin-v1` record. The old torus/radial deployed code is a
known transitional blocker, not completed authority.

> no, not a torus finite triangle collider. not anymore, not with how angry I am. make it a balloon skin soft body system
