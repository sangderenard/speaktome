# Turing Perforated AbstractNN and LLVM Reverse

**Date:** 2026-09-14
**Title:** AbstractTensor perforated linear comparison and tape-free LLVM VJP

## Overview

Added a dendrite-augmented `PerforatedLinear` whose eager forward is authored
entirely on the AbstractTensor surface, a reproducible Adam comparison against
the ordinary `Linear`, and a compiler product that emits an isolated LLVM
forward plus a combined forward/VJP contract. The native reverse comes from
the ProcessGraph backward generator and does not capture or replay the eager
gradient tape.

The work also repaired the compiler-visible matmul adjoint. The analytical VJP
is now authored directly as transpose/matmul/unbroadcast operations in the
backward rule, avoiding nested aggregate and generic-transpose helper frames.
Aggregate result metadata is correlated by the linker's proven return position
rather than by function-local integer IDs, and public LLVM argument shapes now
come from the root function's authored ABI rather than same-numbered helper
storage.

## Steps Taken

- Read `turing/AGENTS.md`, `TEST_BASELINE_AND_HAZARDS.md`, and the graph-adjoint
  LLVM experience reports before editing.
- Added `src/common/tensors/abstract_nn/perforated.py` and exported
  `PerforatedLinear`.
- Added `demo_perforated_regression.py` for NumPy, Torch CPU, or Torch CUDA
  execution using the existing AbstractTensor Adam optimizer.
- Added `src/compiler/perforated_network_llvm.py`, which emits forward and
  ProcessGraph-generated forward/VJP DLLs plus a versioned JSON contract.
- Represented dendrite-to-neuron aggregation as an explicit fixed routing
  matrix, keeping the compiled numerical path in matmul/elementwise form.
- Added eager/backward/training comparison and native LLVM numerical tests.
- Compiled final artifacts under `.turing-cache/perforated-llvm-final`.
- Replaced a preliminary hand-written engine formula with a distillation of
  `engine_toy`'s real LDT-465 cycle simulation. The input contains the complete
  numeric engine specification and physical state/control surface, and the
  target is the complete moving state delta.
- Kept load-aware idle adjustment owned by `ecu.EngineControlUnit`; the Turing
  benchmark drives `EngineCycleSim.known_accessory_shaft_load_w` and records
  the ECU's own feedforward output instead of duplicating the law.
- Jointly trained every ordinary and perforated parameter with AbstractTensor
  Adam across five fuels and five load bands.
- Added `CompiledPerforatedAdam`, which uses the exact isolated LLVM forward and
  ProcessGraph LLVM VJP for every update while retaining only Adam moments in
  Python, plus a live OpenGL/audio runner and a truly headless PNG/WAV path.
- Added startup persistence for the real-simulator dataset union and both LLVM
  artifacts. Cache manifests retain the full native buffer ABI and validate
  dimensions, branch count, engine/capture settings, and source fingerprints;
  `--no-cache` forces both stages cold.

## Observed Behaviour

- The NumPy held-out comparison (`samples=96`, phase steps `35/90/25`, seed
  `1729`) produced ordinary MSE `0.116097` and perforated MSE `0.0605191`, or
  1.92x lower held-out error for this deliberately nonlinear synthetic target.
- The LLVM test compares prediction and all five learned-parameter VJPs to
  independent NumPy formulas. It passes for the explicit upstream prediction
  adjoint contract.
- A singleton compile (`batch=2`, `in_dim=1`, `out_dim=1`, two dendrites) also
  completed, so the temporary dimensions-at-least-two restriction was removed.
- Focused results: 6 passed for the new perforated/native-runtime group; 9
  passed and 1 skipped for backend, batched-matmul, BPID, bias-broadcast, and
  existing ProcessGraph linear reverse checks.
- The full-engine CUDA comparison used 240 real-sim rows, 381 inputs, and 99
  moving outputs. Across seeds 1729/1730/1731, ordinary mean normalized MSE was
  `0.3562057` and perforated was `0.3426717` (4% lower). Perforated error was
  notably lower at medium load (`0.180219` vs `0.245303`) and heavy load
  (`0.083710` vs `0.096613`), slightly higher at idle (`0.273652` vs
  `0.262146`), and nearly tied under overload.
- The full-dimension headless compiled run used a `[2,381]` input, 99 outputs,
  198 dendrite branches, and all five parameter gradients with tape disabled.
  Forty native Adam steps reduced sampled training loss from `0.854999` to
  `0.380176` and emitted the requested PNG, stereo WAV, metrics, and contract.
- The long batch-16 run performed 3,000 compiled updates. Minibatch loss fell
  from `1.07403` to `0.129688`; validation improved from `1.17674` to a best
  `0.311003` at step 1,000, then drifted to `0.322508`. The runner restored and
  saved the step-1,000 parameters and emitted an 18.75-second stereo soundtrack.
- Native post-fix batch sweep on the exact 381-to-99 network: batch
  1/2/4/8/16/32 training throughput was 168/267/368/461/514/420 samples/s;
  forward throughput was 1058/1184/1263/1329/1330/1158 samples/s. Batch 16 is
  the clear learning knee and batch 8 is already at the inference plateau.

## Lessons Learned

- A correct eager VJP is not sufficient when its source rule returns another
  tuple-returning helper: aggregate projections must survive every compiled
  call frame. Owning the two matmul gradient expressions directly in the rule
  is both simpler and more readily isolatable.
- SSA integer IDs are local to functions. Once call linking has established an
  ordered physical return contract, downstream type propagation must use its
  positions; looking those caller IDs up inside the callee can select an
  unrelated helper constant.
- Public inputs require their root declaration shape. Whole-module storage
  analysis legitimately sees many local values with the same integer ID and
  therefore cannot define the external root argument ABI by ID alone.
- A fixed branch-routing matrix expresses the dendritic reduction with ordinary
  differentiable tensor algebra and avoids dynamic reduction-shape frames in
  the isolated native backward.
- Index-modulo holdout was a bad engine-evaluation split: it isolated rare
  startup/event deltas in the idle bin and divided by near-zero train variance.
  Stratifying within each fuel/load cell and applying a physical-range floor to
  delta normalization retained those outputs without letting a numerical
  denominator dominate the score.
- Batch 1 revealed and motivated a singleton reverse-ABI repair: the `[1,99]`
  base-bias gradient had resolved to same-numbered helper storage shaped
  `[381,198]`. Public statically shaped outputs now prefer the settled root ABI
  shape; exact 381-to-99 batch-1 inference and Adam both execute correctly.
- Direct rate comparison at 5 ms per engine transition: real `EngineCycleSim`
  21.48 samples/s, engine plus synchronous audio 21.19/s, compiled singleton
  inference 1,131.6/s (52.7x), and compiled batch-16 inference 1,339.3/s
  (62.4x per sample). Batch 16 remained the learning knee at 514 samples/s.

## Stateful machine-replacement addendum

The first multifuel dataset and live display were not adequate as a stateful
machine-replacement demonstration. They shuffled correlated rows from one long
simulator trajectory, split neighboring ticks across train and validation, and
ran audio from a separate live simulator rather than from learned state. The
load signal also described shaft power to the ECU while applying an unrelated
fraction of peak torque to the dyno.

The revised capture and runtime now:

- generate independent episodes for starting, idling, idle recovery,
  compensated idle load, load recovery, high-end behavior, upshifts, and
  downshifts across every multifuel profile;
- reach high/shift operating points through real `EngineCycleSim` dynamics,
  cloning genuine captured hidden state into independent trajectories rather
  than setting a fabricated RPM;
- perturb only player throttle and electrical/environmental demand using
  seeded correlated noise, leaving state and targets as simulator output;
- synchronize/ramp the real clutch and pair ECU-known shaft watts with the
  station's `torque = power / transfer_omega` mechanical reaction;
- add a compact previous-transition feature surface while leaving deployed
  inference feed-forward and LLVM-friendly;
- hold out complete episodes and schedule every row exactly once per pass with
  distinct-episode-first batches;
- define one epoch as configurable `N` complete dataset passes and mask fixed
  batch padding out of both loss and the compiled VJP seed;
- predict the stable complete 212-field numeric state-delta ABI, explicitly
  naming crank/RMS/accessory/compression-brake/load-shaft torque and power
  outputs instead of dropping fields constant in one capture;
- drive interactive/headless audio from a closed-loop batch-1 compiled network
  state, with keyboard throttle in the OpenGL runner, rather than a live
  physics engine;
- time the actual user-driven baked network path in the live display, reporting
  isolated LLVM inference throughput, complete transition-loop throughput,
  5 ms real-time factor, deadline misses, and numerical divergence. No shadow
  simulator or held-out replay runs in the interactive loop.

The generic perforated AbstractTensor/LLVM system was also advanced to contract
version 2. `PerforatedLinear.forward` accepts a runtime dendrite mask. The
compiled equation adds per-output network authority and a same-shape simulator
fallback delta, permitting scene-graph subgraphs to hand authority between
learned and simulated implementations without recompilation. Those gates live
inside the AbstractTensor graph, so ProcessGraph VJP gradients are naturally
gated too. Dense masking preserves stable latency but does not elide work;
actual compute removal should dispatch separately compiled machine islands.

Focused results for the addendum:

- A real 80-row regime probe reported no stalled or static-target rows. Genuine
  high-end captures were around 2,424 rpm and shift captures around 1,215 rpm;
  starting began at 0 rpm and idle regimes remained around 627-632 rpm.
- The complete `416 -> 212` headless contract compiled both batch-4 training
  and batch-1 rollout artifacts. A deliberately tiny one-pass smoke run reduced
  loss from `1.00595` to `0.165165`; its large closed-loop drift (normalized
  RMSE `7.26`, final crank-torque error `251.7 N m`) confirmed that the audit
  exposes compounding error instead of presenting one-step loss as stability.
  The live/headless rollout now detects non-finite or physically runaway state,
  marks the network as diverged, and resets it between longer-run epochs.
- A corrected live-performance smoke measured the complete 416-to-212 singleton
  LLVM call at 2.73 ms (367 transitions/s) and the full feature-build/inference/
  state-update path at 3.38 ms (296 transitions/s), or 1.48x its 5 ms real-time
  budget. These are short rolling measurements, not a stabilized benchmark.
- Contract-v2 throughput was 376.1 singleton transitions/s (2.659 ms/call) and
  456.7 samples/s at batch 16 (35.037 ms/call), versus 25.24 physical engine
  transitions/s in that short diagnostic run.
- Focused dataset/epoch/load tests passed 6/6. Eager mask plus LLVM forward,
  generated VJP, hybrid simulator fallback, singleton ABI, and weighted-padding
  tests passed 7/7. The final combined focused suite passed 13/13 after the
  full-output and graph-authority extensions.

The later union-ABI/realtime pass generalized the same transition system:

- explicit starter-signal and ignition-enabled inputs now join throttle,
  direct brake, known accessory shaft load, electrical load, gear, clutch, and
  fuel; these commands remain distinct from the state they cause;
- the live default adds 3,000 real-simulator randomized transitions, and an
  executed LDT capture produced exactly 3,000 rows across 60 random episodes
  with throttle spanning 0.00047-0.99977 and shaft load 53.7-24,443 W;
- `make_engine_union_data` pads heterogeneous engine configuration/state paths
  into one ABI and adds a profile selector. The first validation ladder spans
  the LDT multifuel piston engine, dual-motor EV, and AGT1500 Abrams turbine;
- the OpenGL controls now include engine selection, throttle, brake, gear,
  clutch, electrical demand, starter, ignition, and compatible fuel;
- a shadow engine receives the same commands in a background worker, feeds an
  8,192-row live replay buffer, and supplies periodic state correction, while
  the compiled network remains the foreground 5 ms state/audio owner;
- captured-data and live-replay Adam steps also run in that worker and publish
  completed weight snapshots only between foreground transition ticks;
- 2% mean-value information dropout (excluding commands/profile selectors) and
  2% dendrite-branch dropout train tolerance to missing union-ABI information.

The LDT/EV/AGT1500 union compiled as contract-v2 `442 -> 212`. Its short
headless smoke reduced loss from 0.79405 to 0.25244 and measured 3.36 ms for
native singleton inference, 4.17 ms for the complete maintained-state path,
or 1.20x the 5 ms real-time budget. The final focused suite passed 16/16.

A subsequent large-episode invocation exposed that `random-coverage`, which is
a reporting label with its own stochastic sampler, had accidentally been added
to the modulo cycle used for scripted regimes. `NAMED_TRANSITION_REGIMES` and
`_named_regime_for_episode` now isolate the eight scripted cases; a 512-episode
regression plus the three-profile union regression pass.

The startup-cache smoke captured a 50-transition LDT dataset in 9.537 seconds
and reopened it in 0.037 seconds (260.6x faster). The pickle-free dataset
round-trip regression and the native cold-then-cached execution regression
both pass; the latter executes the reopened DLL and compares its output with
the original compiled prediction.

The shared `engine_toy.engine_sound` synthesis boundary now applies a default-on
4 kHz-knee spectral loudness compressor to both gas-turbine header harmonics
and its intake whine. One module constant disables it for reference renders;
the high voices remain present behind a 0.20 gain floor. Both focused tests
pass, and an actual AGT1500 stereo render produced a finite 2,048-frame block.

## Next Steps

None required for this request.

## Prompt History

> can you go into turing, use the backend numpy, or torch on the gpu, and iterate on a basic linear model on adam, there might be a few laying around as demos or w/e. then I want you to make a perforated network system that can be pitted against the abstract tensor ordinary model. be sure to use abstract tensor so the code is translateable and backwardable

> go ahead and see if you can compile a perforated network to llvm using the compiler, with a contract

> oh sorry, uh, very important and it's slipping my mind what it is.... uh... you will want to make use of the process graph backward generator and forward isolation so you can make the forward and backward much faster than tape based

> can you please, you can keep your new method if it's performant, but can you please just define the backward for the matmul or otherwise elegantly fix the problem you discovered

> was the test not tuned well oor did we make any mistakes in our dendrite usage? can you run a slightly more thorough test, try to simulate the multifuel engine under various loads

> make sure the load idle adjustment that was put in recently ends up in your usage of the system, and if it doesn't, make sure the general system of the engine engine obtains ownership of that faculty

> can you please do what I fucking said and put all parameters in to try and train the whole engine

> if you could run the engine engine with it's audio output and perform the training while the screen were an opengl window of a dot field representing the dendrite network developing, along with metrics like loss, etc. and make it so it can be headless and return an image that would be helpful too, be sure to compile the network that's going to do the learning

> can you run it and show me the result of a long long run trying to see if we can cook something really good and also where batch inference runs at an advantageous rate or one near that that might compile into llvm better

> can you fix the one batch we need to know the rate difference between the engine engine and the compiled inference rate

> the engine is not getting the load assist on the idle like it gets in the station work happening concurrently in the repo, or else a load is applied that kills it, the engine state seems to be determining the training examples, which is odd because how are we batching if we only have one engine state? that shouldn't be possible right? and we're not using enough examples to train effectively. our network should be reasoning based on state transitions as well, this is a stateful creature it's trying to learn, so let's brainstorm some slight elegant adjustments that won't pile on latency

> proceed to make the required changes

> basically we want samples of starting, idling, idle recovery, idling with load with compensation and that system recovreing, high end behavior, and up and down shift transitions. it may be better to use data we capture from the engine itself running varrying conditions, a latent noise of player input and environmental response. think about it without obligation

> one epoch should be N x the entire data set, that changes how we express certain things, and explains how deeply we want to try to train the engine process, we may even need new abstract tensor network structures for sequence learning

> then we should replace any live audio input for a lone thread running the current weights, with keyboard throttle, so we can see, just how would this perform if it was given the entire job, and where does error accumulate and by how much

> make sure we're providing the system output torques you know, uh, the goal here is the network can fully replace an engine or machine

> are perforated tensor networks suitable for graph network topology, such as, can we take a total graph based description of the scene's engines and machines layer, and fluently turn subgraphs on and off the network, being on the simulator when off

> extend or accessorize the abstract tensor perforated network system for this task

> let's feed several random sets that still come from a real sim, let's get some broad coverage and look for about 3k more samples to train off of from that kind of material

> okay I misspoke earlier too, in addition to what you're saying and doing, let's do put in the shadow sim for the live and then correct it every N seconds and then train on the actual behavior, and give a broader set of controls, like the engine demo/browser has for the brake, gear shifting, as much as seems like it's ready or can be elegantly used

> make sure you can parameterize, like, starter signal, ignition on, ignition off, the settings it will need to be aware of and know the interference from states

> this can be traned to different goals but I would like us to focus on making a realtime system on engine transforms like this

> oh, you mentioned some important t hings, would you be willing to teach the network different engine profiles, so the demo could train on all the engines at once and the user could test it on any engine

> this is a union abi experiment

> if we put a tiny bit of dropout in we could train masks to be effective at missing information

> if you get those two to work, the next should be the abrams turbine

> alright as soon as we can can you get engine selection and the new controls into the live visual demo

> is the profoundly long beginning to these anything we can cache?

> okay, so, can we by default but opt-out by constant definition in the code, loudness compression on turbine whines over a certain frequency
