# Turing vehicle tensor/WebGPU transition handoff

**Date/Version:** 2026-08-26 v1  
**Repository:** `C:\dev\Powershell\turing`  
**Status:** locally testable transition checkpoint; **not published**  
**Public baseline intentionally preserved:**
<https://sangderenard.github.io/Turing/?build=cc58b9a>

## Executive summary

The Springtail vehicle remains live on GitHub Pages at the last known-good
published build. This work did not overwrite it. The local source now has a
real compiler-owned path from the SymPy tire/contact law, through generated
AbstractTensor Python, through the repository numerical-region planner and
tensor SSA lowering, to a one-dispatch four-wheel WGSL kernel.

The existing browser runtime ABI is still preserved: the live-compatible
contact shader exposes four output buffers (force x/y/z and contact area).
Separately, an **opt-in** version of the same compiler artifact exposes those
outputs in one component-major GPU buffer. A second compiler helper proves
that a SymPy matrix product becomes AbstractTensor `matmul`, retains its
`blas.gemm` semantic identity, and is substituted by the existing tiled
WebGPU GEMM backend.

The transition is therefore halfway across the important boundary:

1. pure math -> AbstractTensor -> one WGSL contact dispatch is working;
2. packed GPU output is working and opt-in;
3. backend GEMM substitution is working;
4. the generated page model carries both artifacts;
5. the browser worker does **not yet chain** the packed contact buffer into the
   GEMM buffer without CPU reduction;
6. the worker still maps all wheel outputs and performs `r x F` and summation
   in JavaScript before calling the compiled chassis Wasm.

The next agent's principal job is to complete item 5 without weakening the
compiler, changing tick ownership, or regressing the current driving behavior.

## Non-negotiable intent from the user

- Trust Turing's compiler. Slow compilation is not evidence that a shortcut is
  needed.
- Author physics as pure math/SymPy or another language of convenience, generate
  AbstractTensor Python, and let the compiler select backend work.
- Prefer backend GEMM/matmul intrinsics. Do not replace a visible matrix product
  with hand-written scalar reduction merely because it is small.
- Keep the vehicle hook in the existing lockstep world physics. WebGPU APIs may
  be asynchronous, but there must not be a second independent vehicle schedule.
- Do not add a worker merely to hide GPU synchronization.
- Avoid CPU/GPU juggling and unnecessary world uploads.
- Preserve parametric JSON vehicle configuration.
- Changes in support/terrain traversal must remain general platformer/rigid-body
  physics, not ramp-specific vehicle rules.
- Forces should integrate smoothly. Do not introduce first-order force or
  control discontinuities where second-order/C2 channels now exist.
- Do not touch or weaken compiler semantics to make a build finish sooner.
- Particle-driven sediment/depth-map redistribution is explicitly deferred.

## Repository and working-tree safety

`C:\dev\Powershell` is a junk-drawer parent, not a monorepo. Work only in the
`turing` child repository for implementation. Read:

- `C:\dev\Powershell\turing\AGENTS.md`
- `C:\dev\Powershell\turing\TEST_BASELINE_AND_HAZARDS.md`
- this report and the three vehicle reports immediately preceding it.

The Turing worktree is heavily dirty and many of the vehicle files are
untracked. They contain substantial user work. Do not clean, stash, reset,
checkout over, or assume an untracked file is disposable.

Current relevant status at handoff:

```text
 M src/compiler/ssa_wasm_backend.py
 M src/compiler/ssa_webgpu_backend.py
 M src/compiler/tensor_ssa_lowering.py
 M tests/test_backend_identities.py
 M tests/test_symbolic_fluid_direct_backends.py
 M tests/test_webgpu_ssa_backend.py
?? configs/vehicles/fun_car.json
?? src/compiler/abstract_ui_div_map.py
?? src/compiler/abstract_ui_vehicles.py
?? src/compiler/state_loop_deployment.py
?? tests/test_abstract_ui_vehicles.py
```

The isolated Pages staging repository used for the last publication is:

```text
C:\Users\alber\AppData\Local\Temp\turing-pages-publish-20260826
```

Its known public baseline commit is `cc58b9a`. Do not publish the current
transition merely because generation succeeds. First finish and browser-test
the GPU-resident chain described below.

## Current architecture

### Mathematical authority

`src/compiler/abstract_ui_vehicles.py` owns:

- strict JSON loading and vehicle configuration;
- the scalar SymPy chassis/drivetrain/state equations;
- the per-wheel SymPy pneumatic contact/Coulomb-friction equations;
- the generated four-wheel AbstractTensor precompile;
- the SymPy matrix -> AbstractTensor -> backend GEMM helper;
- vehicle mechanical, torque, suspension, and presentation models.

The contact law currently publishes:

```text
chassis_force_x
chassis_force_y
chassis_force_z
contact_area
```

It consumes 43 runtime-parametric four-wheel inputs. The equations include
smooth compression limits, directional pneumatic damping, active damper scale,
pressure-derived contact area, load-sensitive static/kinetic Coulomb friction,
and smooth force-limit blending.

### Generated AbstractTensor source

`compile_wheel_contact_abstract_tensor(*, packed_outputs=False)`:

1. obtains the SymPy equations;
2. derives the stable sorted input ABI directly from free symbols;
3. runs SymPy CSE;
4. prints supported intrinsics as native AbstractTensor methods (`.sqrt()`,
   `.tanh()`), not tiny Python helper functions;
5. invokes the existing `compile_ast_aot` frontend;
6. lowers the planned numerical program to repository SSA;
7. requires exactly one numerical region;
8. purpose-specializes every tensor value to the fixed four-wheel shape;
9. performs ordinary tensor SSA lowering;
10. emits WGSL with either the legacy output ABI or opt-in packed outputs.

This orchestration correction mattered:

- first observed compile: **269.14 s**, 106 functions, 48 numerical regions;
- removing a redundant full scalar compile: **11.51 s**;
- printing `.sqrt()`/`.tanh()` directly instead of helper calls: **5.08 s**,
  59 functions, **one numerical region**, 28 generated source lines.

This was a precompile-source improvement. No compiler semantic shortcut was
used.

### Production-compatible contact artifact

`compile_wheel_contact_webgpu()` now returns the non-packed artifact from the
AbstractTensor pipeline. It no longer asks the old direct scalar
SymPy->SSA->WGSL path to emit the production shader. That old path is retained
as an oracle but cannot currently emit `Tanh` to WGSL.

The non-packed artifact deliberately preserves the worker's established ABI:

- one packed feed buffer (43 components x 4 lanes);
- four output storage buffers;
- workgroup `(4, 1, 1)`;
- one dispatch group.

### Opt-in packed contact artifact

Calling:

```python
compile_wheel_contact_abstract_tensor(packed_outputs=True)
```

produces one output storage binding with component-major layout:

```text
component 0 lanes 0..3
component 1 lanes 0..3
component 2 lanes 0..3
component 3 lanes 0..3
```

`ssa_webgpu_backend.emit_module(..., packed_outputs=False)` retains old
behavior by default. Packed emission refuses mixed output dtypes and records:

- `packed_outputs`;
- `output_span` (SSA value ids in component order);
- one output binding in `io_layout`.

Do not make packed outputs the compiler-wide default.

### Backend GEMM transition

`compile_sympy_matrix_to_abstract_tensor_backend(...)` is the requested helper
for this route:

```text
SymPy MatrixExpr
  -> generated Python using AbstractTensor.matmul
  -> AOT numerical region
  -> fixed application-owned shape specialization
  -> repository tensor SSA
  -> backend identity selection
  -> WGSL artifact
```

`compile_vehicle_wrench_reduction_webgpu()` currently authors:

```text
[1 x 4] unit_row @ [4 x 6] wheel_wrenches -> [1 x 6] chassis_wrench
```

The resulting artifact is complete and reports:

```text
variant: webgpu_tiled_gemm
shader contains: var<workgroup> tile_A
```

The helper shape-specializes the planned matmul region because the legacy AOT
capture drops concrete shapes. That specialization belongs to this pre-baked
vehicle program, not to the generic compiler.

### Compiler changes, all narrow/default-preserving

1. `src/compiler/tensor_ssa_lowering.py`

   A planned `Call` already selected as `matmul_double` can retain the
   `AbstractTensor.matmul` semantic identity when its existing attributes say
   `tensor_operation == "matmul"`. This lets backend identity selection replace
   it with `blas.gemm`. Unmarked calls are unchanged.

2. `src/compiler/ssa_webgpu_backend.py`

   Added `packed_outputs=False`. Default behavior is unchanged. Same-dtype
   results can opt into one component-major storage buffer.

3. `src/compiler/ssa_wasm_backend.py`

   Added optional `work_contract=None` to choose a named contract for one
   emission without mutating process-global compiler state. Default resolution
   is unchanged. Added the missing deploy/fast spelling of exponent `-0.5` as
   `1 / sqrt(x)`, guarded by `contract.inexact_identities`. Exact develop/prove
   contracts still refuse it.

4. Vehicle scalar Wasm explicitly requests `work_contract="deploy"` because
   the smooth physical laws deliberately use the documented sqrt-family
   bounded identities. This is a call-site license, not a global weakening.

### Extended precision

`extra_precision_closure(function, limbs=2)` wraps arguments with the existing
repository `Precision` expansion and collapses results at the closure boundary.
It is tested for ordinary one-limb behavior and two-limb sqrt/arithmetic.

Important limitation: this closure is **not yet injected into the generated
contact AOT source**. It is a working opt-in callable seam, not proof that the
WGSL contact shader is already running expanded precision. Keep GEMM outside
the closure so the visible matmul identity remains available to backend
substitution. If the next agent integrates precision into compilation, do it
as a separately selected program variant and measure dispatch/limb cost.

## Browser/worker state at handoff

The existing fixed-step worker remains the sole state owner. It awaits contact
compute before compiled chassis Wasm and snapshot publication. Do not introduce
an independent vehicle clock.

The worker currently does this every vehicle tick:

1. assemble four `wheelContactRecords` in JavaScript from terrain sampling and
   mechanical graph state;
2. pack 43 x 4 float inputs;
3. upload the contact feed;
4. dispatch the non-packed AbstractTensor WGSL contact shader;
5. copy and map all four output buffers;
6. calculate each `attachment x force` and sum all wheel forces/torques in
   JavaScript;
7. write six scalar contact-wrench inputs into the chassis Wasm arena;
8. execute the compiled vehicle/chassis Wasm;
9. publish the recycled lockstep snapshot.

The CPU reduction in step 6 is explicitly temporary. It is located in
`emit_javascript_physics_worker()` after `dispatchVehicleContacts()`.

The browser initialization also still assumes four independent output buffers:

```javascript
outputs=program.outputs.map(()=>device.createBuffer(...))
reads=program.outputs.map(()=>device.createBuffer(...))
```

Do not simply point that runtime at the packed artifact; its binding and
readback assumptions must be changed coherently.

## Exact next implementation sequence

### Phase 1: make the packed contact buffer a complete wheel-wrench matrix

Extend the SymPy contact publication with torque components calculated from the
already-present attachment parameters:

```text
torque = attachment cross chassis_force
```

Desired packed component order:

```text
force_x[4]
force_y[4]
force_z[4]
torque_x[4]
torque_y[4]
torque_z[4]
contact_area[4]
```

This is a 7 x 4 component-major output. The first 24 floats are already a
`[6 x 4]` wheel-wrench matrix. Keep `contact_area` afterward as telemetry.

Do not compute `r x F` in hand-written WGSL. It belongs in the SymPy-generated
AbstractTensor contact graph.

Maintain the non-packed four-output compatibility artifact until the new chain
is browser-tested. One safe approach is to parameterize the publication set so
the legacy runtime variant stays force+area while the packed pipeline variant
adds torque. Do not casually change the live-compatible output list underneath
the existing worker.

### Phase 2: orient GEMM to consume packed storage directly

The current mathematical helper uses `[1 x 4] @ [4 x 6]`. For direct packed
consumption, prefer the equivalent visible product:

```text
[6 x 4] wheel_wrenches_transposed @ [4 x 1] unit_column
    -> [6 x 1] chassis_wrench
```

This matches the component-major output without a CPU transpose. Change the
SymPy MatrixExpr and fixed shapes, then prove the backend still selects
`webgpu_tiled_gemm`. Do not replace it with six scalar sums.

The unit column is a tiny immutable GPU buffer `[1, 1, 1, 1]` created during
vehicle GPU initialization.

### Phase 3: build one joined GPU pipeline object

Create both compute pipelines and persistent buffers during
`initializeVehicleGpu`:

- contact feed buffer;
- packed contact/wrench/area buffer, with storage usage allowing it to be read
  by the GEMM dispatch;
- immutable unit-column buffer;
- six-float reduced chassis-wrench buffer;
- minimal staging/read buffers only for the data the current Wasm/snapshot
  boundary actually consumes.

In one worker tick, encode contact dispatch followed by GEMM dispatch. Keep the
existing tick await/barrier. A second compute pass in the same command encoder
is the conservative initial ordering. Validate WebGPU storage visibility rules
against the actual implementation rather than assuming an implicit cross-
dispatch barrier.

Do not create another worker. Do not let a previous tick overlap the next.

### Phase 4: remove the JavaScript force/torque reduction

After the chained output is numerically checked, delete the temporary loop that
sums forces and calculates attachment cross force. Populate only:

```text
contact_wrench_force_x/y/z
contact_wrench_torque_x/y/z
```

from the six-float reduced result at the Wasm boundary.

For the first correctness checkpoint it is acceptable to continue reading
per-wheel `contact_area` and force telemetry so the current HUD and traction
logic remain intact. The essential transition is that chassis wrench authority
comes from the compiler-generated GPU contact+GEMM graph, not JavaScript.

Then profile. If telemetry mapping dominates, move the traction/utilization
math into the same compiled tensor graph or publish a compact telemetry buffer.
Do not remove diagnostics blindly.

### Phase 5: prove numerical and behavioral parity

Add a deterministic CPU oracle test using the same four records:

1. evaluate the SymPy force and torque expressions;
2. sum the four wrench rows in Python/NumPy;
3. verify the compiled GEMM artifact metadata and, where the test environment
   permits, execution result;
4. require the same six chassis-wrench values within the documented f32 bound.

Add tests that ensure:

- contact still lowers to one numerical region;
- packed output order is exactly documented;
- GEMM variant remains `webgpu_tiled_gemm`;
- non-packed default ABI remains available;
- JavaScript no longer contains the temporary force/torque summation loop;
- the worker still has one joined vehicle tick and no extra scheduler;
- no full-world upload or DOM telemetry loop is introduced.

### Phase 6: browser-test before publishing

Generate the page locally and test in a real WebGPU-capable browser:

- start mounted in Springtail;
- frame, wheels, double-wishbone suspension, drivetrain and engine visible;
- front wheels are the animated steering axle;
- chase camera works;
- throttle is torque demand, not velocity assignment;
- automatic second-gear start can downshift into crawler first under sustained
  integrated demand;
- right-car and respawn controls work;
- no freeze on mount;
- no new jitter or vehicle disappearance;
- contact patches remain correct on rear tires and disturbed terrain;
- upside-down driving is resisted by cage contact and orientation/traction,
  not a hidden gimbal;
- compact HUD remains shader-rendered;
- DOM stats update only when explicitly expanded;
- no console or WebGPU validation errors.

Compare feel and behavior against the live `cc58b9a` baseline before
publishing. If the new pipeline is technically correct but materially less fun
or stable, do not publish it; diagnose the numerical/ordering difference.

## Vehicle controls and smoothness already in local source

The JSON and worker contain second-order/C2 work that must not be forgotten
while finishing GPU integration:

- second-order throttle, steering, and brake channels;
- second-order engaged transmission ratio;
- integrated downshift demand and crawler-first entry;
- second-order friction-utilization/traction response;
- smooth C2 support engagement and travel-stop correction;
- smooth active damper scaling;
- directional pneumatic compression/rebound damping;
- cage static/kinetic friction and cage wrench application;
- substantial engine, transmission, differential, rim, and tire masses;
- pitch/yaw/roll and angular velocity state;
- wheel rotational inertia and explicit drivetrain torque graph.

The user is particularly sensitive to discontinuous correction/jitter. Do not
reintroduce direct threshold impulses where the current source uses
`secondOrderChannel`, `c2Unit`, `c2Positive`, or `c2Clamp`.

## DOM/HUD audit status

The compact four-corner vehicle instruments are rendered in the viewport
shader. The old detailed DOM contact graph still exists as an explicit `STATS`
diagnostic. It now synchronizes only while its root has class `expanded`.
Mount/dismount only toggle its visibility at those discrete events.

Transmission DOM controls live in the general settings rows and update only
when transmission state changes.

There are still other general page DOM updates, including the main viewport
readout, map marker transforms, device indicators, and non-vehicle telemetry.
Do not claim the whole page is DOM-free. If frame pacing remains problematic,
profile those call sites individually. The vehicle contact panel's former
unconditional per-animation-frame call is already removed.

## Tests and measurements completed

### Green focused compiler/vehicle gate

```text
14 passed, 1 warning in 9.29s
```

Covered:

- backend GEMM identity recovery;
- opt-in packed WGSL output;
- exact WASM sqrt-family refusal;
- explicit deploy reciprocal-sqrt emission;
- one-region packed vehicle contact artifact;
- explicit extended-precision closure.

### Generated vehicle page-model integration

```text
1 passed, 1 warning in 36.39s
```

`test_living_map_has_a_vehicle_slot_not_a_car_specific_control_mode` verifies
the vehicle model, packed contact artifact, tiled GEMM artifact, generated
page JavaScript, shader HUD, smooth support/steering channels, terrain,
mechanical graph, controls, and worker source.

### JavaScript parsing

Generated page JavaScript and generated worker JavaScript both passed:

```text
[('page', 0, ''), ('worker', 0, '')]
```

Use an explicit UTF-8 pipe on Windows; the generated page contains Unicode and
CP-1252 input to `node --check` fails before Node sees the source.

### Full WebGPU backend file

```text
10 passed, 4 failed
```

The four failures are the deprecated-AOT quartet already listed in
`TEST_BASELINE_AND_HAZARDS.md`:

- `test_ast_generated_float32_program_emits_wgsl_compute`
- both `test_ast_generated_loop_uses_structured_wgsl` cases
- `test_float64_is_a_named_webgpu_core_shortfall`

Direct inspection showed the first receives an empty fused `kernel` and the
WGSL emitter correctly reports `compute module has no named output`. This
occurs before the packed-output changes and is the recorded persistent AOT
capture/checkpoint problem. Do not weaken WGSL output validation to make it
green.

### Syntax/static checks

- `python -m py_compile` passed for all touched compiler/vehicle Python files.
- `git diff --check` passed for tracked touched files; only expected Windows
  LF/CRLF warnings were printed.

## Tests to run next

Do not run the full suite; it is documented not to finish. Prefer:

```powershell
$env:PYTHONPATH='C:\dev\Powershell\turing'
& 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' -m pytest `
  tests/test_backend_identities.py `
  tests/test_webgpu_ssa_backend.py::test_same_typed_outputs_can_publish_one_component_major_gpu_span `
  tests/test_symbolic_fluid_direct_backends.py::test_exact_contracts_forbid_the_private_sqrt_spellings `
  tests/test_symbolic_fluid_direct_backends.py::test_wasm_reciprocal_sqrt_is_an_explicit_deploy_identity `
  tests/test_abstract_ui_vehicles.py::test_sympy_contact_precompile_is_one_opt_in_packed_tensor_dispatch `
  tests/test_abstract_ui_vehicles.py::test_vehicle_tensor_precision_closure_is_explicit_and_collapses_at_boundary `
  -q
```

Then run the vehicle projection test separately and allow it to finish:

```powershell
& 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' -m pytest `
  tests/test_abstract_ui_vehicles.py::test_living_map_has_a_vehicle_slot_not_a_car_specific_control_mode `
  -q
```

The scalar SymPy SSA oracle test can take several minutes. Silence is not proof
of a hang. Do not stop it merely because the tensor precompile is much faster.

## Known risks and traps

1. **Do not publish the staged transition yet.** The packed buffer and GEMM are
   independently real but are not yet chained in the worker.
2. **Do not remove the non-packed runtime variant first.** It is the behavior-
   preserving bridge and rollback surface.
3. **Do not hand-write a six-sum WGSL kernel.** The visible matrix product and
   compiler-selected GEMM are the point of the demonstration.
4. **Do not confuse async WebGPU APIs with permission for an independent
   schedule.** Keep one lockstep owner and joined tick.
5. **Do not restore raw control thresholds.** The current local source moved
   steering/throttle/brake, downshift demand, ratio changes, and traction
   response toward second-order integration.
6. **Do not delete user changes in untracked vehicle files.** They are the
   active implementation.
7. **Do not use `git stash`, `git reset`, or checkout-over-file to establish a
   baseline.** Use an isolated worktree if comparison is truly required.
8. **Do not infer compiler failure from compilation time.** The 269 s run
   completed successfully and exposed a redundant precompile orchestration
   call; the correct fix reduced the source work, not compiler fidelity.
9. **Packed contact telemetry and GEMM have a layout contract.** Record the
   component order explicitly and test it before wiring buffer aliases.
10. **WebGPU buffer offsets have alignment constraints.** Avoid assuming the
    trailing `contact_area` segment can be rebound at an arbitrary byte offset;
    design binding ranges against actual WebGPU limits.
11. **The current scalar chassis remains Wasm.** A six-float readback is an
    acceptable transition boundary. Compiling a larger state graph to WGSL can
    follow after contact+GEMM is correct; do not make that expansion a reason to
    stall the first coherent GPU chain.

## Definition of done for this transition

The transition is complete when all of the following are true:

- the wheel contact force, torque, and area originate from the SymPy-generated
  AbstractTensor WGSL kernel;
- one packed GPU buffer carries four wheel wrench rows/components;
- the compiler-selected tiled GEMM reduces those wheel wrenches on GPU;
- JavaScript no longer performs the authoritative wheel force/torque sum;
- only the reduced chassis wrench crosses into scalar chassis Wasm, plus
  deliberately retained compact telemetry;
- one worker tick remains the sole scheduling authority;
- default compiler behavior remains unchanged unless an explicit packed,
  precision, or deploy-contract option is selected;
- the generated page and worker parse;
- focused tests pass, with only documented baseline failures;
- a real browser run shows no mount freeze, disappearance, validation error,
  driving regression, or new jitter;
- the result is compared against `cc58b9a` and only then published from the
  isolated Pages staging repository.

## Closing assessment

This is a strong compiler demonstration already: a complicated pressure,
spring/damper, and Coulomb contact law written as SymPy is CSE-reduced into
compact AbstractTensor Python and becomes one four-wheel WebGPU dispatch. A
separately authored SymPy matrix product already reaches Turing's genuine
tiled GEMM backend. The remaining work is not to invent more math or another
scheduler; it is to connect those two compiler products with a persistent,
tested buffer ABI while preserving the unusually good current vehicle feel.

