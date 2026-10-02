# Living Data Map vehicle WebGPU physics

**Date/Version:** 2026-08-25 v1

## Overview

Added a JSON-configured vehicle slot to Turing's generated AbstractUI Living
Data Map. The implementation shares a general analytic support-surface contract
with the platformer and projectile paths, compiles scalar chassis physics to
Wasm, compiles four tire/contact lanes from SymPy through ProcessGraph/SSA to
WGSL, and coordinates both stages inside the existing 120 Hz world worker tick.

## Prompt History

> check out the recent markdown that goes over where our page is compiled and where it is generated and where it is hand authored, and I want you to do an additional audit on giving players a vehicle slot, which lets them use different physics for their controls, a modular changeable slot for, at the moment let's say, a car. well,&#x20;
>
> then I want to be able to use a json configuration for the physics of the car object and I want it to be fun, let it do jumps on ramps(work out z buffer's gradient traversal rules) and we'll work a parallel spring compute into the physics engine
>
> All while increasing the degree to which the site is designed in languages of convenience, like python, and got compiled mostly automatically into a website that does all these things
>
> There are a lot of ways to make bad decisions because this is a vulnerable place that can determine the future of a lot of stuff, so be careful and thoughtful and unafraid to ask questions

> now be careful because whatever you do for ramp capability has to be something done for platformer physics in general

> I want parametric air pressure weight etc. responsive contact patches calculated against whatever mesh and then friction in static and kinetic calculated carefully, and forces communicated between the chassis contact points and wheels, and i want you to use  this repo's profound ability to compile to webgpu compute shader a parallal physics kernel baked to purpose with parametric inputs allowing a wide variety of vehicle configurations to merely be precompiled into a shader to plug in and go

> I noticed an async, I don't know if we're winning performance with more async scheduling, I was going to suggest the vehicle be a hook in the physics that already exist, so it functions in lock step with the world physics

> if you are saying "so we won't do it" re: getting the shader in place, no, we will just compile what needs to be compiled and coordinate it well through javascript, assembly, and shader

## Steps Taken

1. Read the compilation/productive-world audit and traced Python model,
   generated page, bespoke host JavaScript, Wasm, SSA, and WebGPU boundaries.
2. Added strict vehicle JSON loading, a modular player vehicle slot, a shared
   support-surface ABI, a four-lane contact kernel, and scalar chassis Wasm.
3. Hooked vehicle advancement into the existing fixed-step worker with a
   non-overlap guard and an explicit GPU-result barrier before snapshot publish.
4. Added both worker-local WebGPU and page-WebGPU bridge paths; both preserve a
   single authoritative worker tick and fall back to Wasm on bounded failure.
5. Generated the portable HTML projection and exercised it in Chrome.

## Observed Behaviour

- Chrome reported `lockstep-worker-webgpu+wasm` for the mounted Springtail.
- After exposing contact telemetry in the shared snapshot, all four settled
  spring lanes reported approximately 1520 N and the local page logged no errors.
- Focused vehicle/state-loop/physics tests passed. The broader symbolic compiler
  suite still has two current-tree expectation failures recorded separately.

## Lessons Learned

WebGPU availability differs between the in-app browser and desktop Chrome, so
capability fallback must be explicit and visible. Async GPU submission is not
itself a second schedule: the important invariant is that one worker owns the
world tick, permits no overlap, and publishes no state until every selected
stage has joined the barrier.

## Next Steps

- Replace CPU readback/reduction with a second GPU-resident chassis-reduction
  stage while retaining the same tick barrier and deterministic fallback.
- Replace the analytic plane adapter with a compiled triangle/BVH mesh sampler
  behind the same platformer/projectile/vehicle support-surface ABI.
- Extend the tire model with wheel angular inertia, relaxation, temperature,
  camber, differential, anti-roll coupling, and aerodynamic downforce.

