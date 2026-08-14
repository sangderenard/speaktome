# Turing parametric multi-card dt-system fluid capture handoff

**Date:** 2026-08-05
**Version:** 1
**Title:** Full-physics repository fluid capture through AbstractTensor, hierarchical cards, SSA, Fortran, and the native C shell

## Abstract

This session replaced the earlier application-specific fluid approximation with
an attempt to compile the repository's real `VoxelMACFluid` and adaptive
`dt_system` source through the normal AST/AOT compiler. The work deliberately
kept viscosity, scalar diffusion, pressure projection, adaptive subdivision,
and compiler validity checks enabled. It also began a target-neutral parametric
card-program representation that retains the hierarchical `ControlProgram`, its
per-region `FusedProgram` cards, public ports, and exact resident-memory alias
edges instead of flattening the entire source into a single numerical program.

The full 4x4 diagnostic build now captures the real higher-order
`fluid_advance` callback and reaches precompile-to-SSA lowering with a 297-value
hierarchy. It does not yet emit Fortran. The remaining blocker is incorrect
hierarchical identity projection for six loop-carried update edges in the real
CG viscosity/pressure solvers. The latest checkpoint contains source graph,
frontend, compiled plan, and captured program, so investigation can resume
without losing the captured hierarchy when the implementation hash is stable.

## Repository and ownership state

- Repository: `C:\dev\Powershell\turing`
- Branch: `codex/recursive-reduction-bridge`
- Starting HEAD: `b3773f30a4391b1cff92b6a590872fc409e3cc26`
  (`Compile captured programs through resident native arenas`)
- The worktree was already heavily dirty and continued to receive unrelated
  changes during this session. Do not stage all files, reset the tree, or infer
  ownership from `git status` alone.
- Generated probe artifacts are under ignored `build/` paths. They were not
  force-added, and no log/image output was added to the Turing repository.
- No commit or push was made during this session.

Files materially changed or added by this effort include:

- `src/compiler/parametric_card_program.py` (new)
- `tests/test_parametric_card_program.py` (new)
- `src/compiler/fortran_c_shell.py`
- `src/compiler/native_voxel_fluid.py`
- `src/cells/bath/voxel_fluid.py`
- `src/common/tensors/abstraction.py`
- `src/common/tensors/abstraction_methods/reduction.py`
- `src/compiler/glsl_deployment_strategy.py`
- `tests/test_fortran_c_shell.py`
- `tests/dt_system/test_voxel_fluid_engine.py`
- `tests/test_canonical_forward_and_backward.py`
- `tests/test_loop_composer.py`
- `tests/test_process_graph_shell.py`

Some of those files also contain concurrent work not authored in this session.
Review hunks, not whole files, before committing.

## User intent and non-negotiable constraints

The requested architecture is not a one-shot trace specialized to one sample.
It is a system of sequential cards whose inputs and outputs alias shared arena
addresses under an outer coordinator. Classes, modules, higher-order functions,
the adaptive dt controller, and AbstractTensor operations should remain
traversable program structure. A backend should be able to select parameters,
lock others, cache cards, and execute the same retained program parametrically.

For this fluid demonstration specifically:

- use the repository `VoxelMACFluid`, not a bespoke solver;
- use the actual `dt_system` adaptive controller and microstepping;
- ingest Python source through the compiler process;
- do not disable physics, validation, or difficult loops to obtain an image;
- produce native Fortran through the existing registered backend and common C
  shell;
- preserve preallocated arena semantics;
- keep compilation checkpoints and allow long runs to finish;
- do not retain generated logs or images in Turing.

## Implemented architecture

### 1. Parametric card program

`src/compiler/parametric_card_program.py` adds a backend-neutral retained
program model:

- `CardPort` describes typed/shape-aware card inputs and outputs.
- `ProgramCard` retains one control or numerical card.
- `CardAliasEdge` records canonical producer-to-consumer shared-memory aliases.
- `ParametricCardProgram` retains the outer coordinator, region programs,
  map-derived connection graph, public ports, feedback rotation policy, and
  address policy.
- `build_parametric_card_program(...)` builds the representation from an
  `AOTCompilation` without replaying or flattening the captured program.
- `capture_ast_card_program(...)` is the Python-source entry path.

The mapping ABI is `turing.parametric-card-program.v1`. The coordinator policy
is outer-coordinator address ownership; state feedback is represented as
resident-memory rotation. Region feeds with no producer are promoted to
coordinator/public ports rather than silently baked from the observation.

`compile_ast_fortran_c_shell(...)` now retains this card program in API
metadata by default (`retain_card_program=True`) and accepts
`mutable_parameters`. The native artifact is still emitted as the existing
composed control routine plus numerical region routines; the card object is a
retained execution/linking contract, not yet a second runtime.

### 2. Fortran/C multidimensional arena boundary

The common shell now maps file-order C/NumPy row-major indices to Fortran
column-major storage when reading multidimensional initial arenas, and maps
back when writing multidimensional final outputs. Resident feedback remains in
Fortran storage order and rotates without a copy.

This is a shell ABI correction, not a change to source tensor semantics. The
semantic shape remains Python/NumPy ordered. A real compiled 2x3 Fortran module
test changes logical element `[0, 1]` and verifies that the output file reshapes
correctly in NumPy.

The shell also restores source names for promoted hierarchical inputs using
the compilation identity table, so a generated `t<ID>` parameter can recover a
public spelling such as `dt` where the provenance is unambiguous.

### 3. Real fluid metrics and source configuration

`VoxelMACFluid.compute_metrics(prev_mass)` no longer returns placeholder zero
divergence and mass error. It now computes:

- maximum absolute staggered velocity;
- MAC divergence from face differences times `inv_dx`, zeroed in solids;
- divergence infinity norm;
- relative salinity-mass error against `prev_mass`.

The native source captures previous mass before advancing and asks the real
engine for metrics afterward.

The native demo no longer disables viscosity or scalar diffusion and no longer
sets the pressure solver to one iteration. It uses the repository defaults,
with:

- grid `nx=height`, `ny=width`, `nz=1`, `dx=0.025`;
- gravity `(9.81, 0, 0)` because solver axis zero maps to display rows;
- one Gaussian salinity/temperature plume near the lower center;
- `Targets(cfl=0.5, div_max=1e-2, mass_max=1e-3)`;
- `STController(dt_min=1e-6, dt_max=0.025)`;
- frame window `0.025`, initial dt `0.01`;
- `dt_initial -> dt_out` resident feedback;
- `dt_initial` declared mutable at the compiler boundary.

A Python reference probe at 16x16 completed three accepted microsteps
`[0.01, 0.01, 0.005]`, advanced exactly `0.025`, and reported approximately:

- `max_vel = 1.443e-4`
- `div_inf = 4.63e-13`
- `mass_err = 1.185e-4`
- maximum pressure magnitude near `1839`

Controller fields other than `dt_initial` are not yet fed back through the
native shell. In particular, `acc`, `max_vel_ever`, and the controller's
updated `dt_max` remain an open class-state/card-port design issue.

### 4. Pressure visualization

The RGB source uses salinity as dye and normalized pressure magnitude as a
darkness multiplier. The scale is the larger of actual maximum pressure and a
hydrostatic floor `rho0 * 9.81 * dx`, preventing numerical dust from being
amplified into severe banding. This code has not yet reached a new native
window because the full program is blocked in SSA lowering.

### 5. NumPy reduction protocol for AbstractTensor

Full capture first failed because `np.sum(AbstractTensor)` called
`AbstractTensor.sum(axis=..., out=...)`, while the tensor dialect only accepted
`dim` and `keepdim`. The abstraction now normalizes NumPy and tensor spellings
for `sum`, `mean`, `min`, `max`, and `prod`:

- `axis` aliases `dim`;
- `keepdims` aliases `keepdim`;
- a requested `dtype` becomes an explicit tensor cast before reduction;
- non-`None` `out` is rejected because returned tensors must be explicitly
  arena-aliased rather than silently mutating an unknown NumPy destination.

This preserves canonical autograd metadata (`axis`, `keepdim`) and delegates
only the normalized operation to each backend.

### 6. Higher-order function-table linkage

After reduction was fixed, the captured dt program exposed `t263`, a synthetic
external scalar corresponding to `metrics.mass_err`. The producer was missing
because `fluid_advance` crossed into `run_superstep` as the first-class
`advance` parameter. Runtime discovery knew it was a `_CompiledStructuralFunction`,
but hierarchy planning had no `PlanCall` for `advance(state, dt)`.

The planner now treats `StaticReference` function values as opaque
`FunctionReference` addresses during safe callsite specialization. A reference
can propagate through specialized `Input` parameters. A call whose function
name is bound to such a reference receives an ordinary `callee_ref` with
`callee_resolution = bound-function-parameter`. No Python callable is stored
or invoked to establish the edge.

A focused `root -> apply(value, increment) -> operation(value)` test verifies
that the specialized `apply` card contains a real callsite shell for
`increment`.

This change expanded the fluid hierarchy candidate from 11 values to 297 and
eliminated the fabricated `t263` public input. It is the most important proof
this session that non-single-shot, higher-order multi-card capture is viable.

## Full-physics compile chronology

Command used for every diagnostic attempt:

```powershell
python -m src.compiler.native_voxel_fluid `
  --output build/native-voxel-fluid-full-physics-probe `
  --width 4 --height 4 --compile-only
```

### Attempt A: NumPy reduction boundary

- Duration: 117.9 seconds in the prior run; the direct rerun reported 190.3
  seconds after checkpoint/hash changes.
- Saved source graph, frontend, and compiled plan; later saved captured program.
- Failure: `AbstractTensor.sum() got an unexpected keyword argument 'axis'`.
- Correction: normalize the NumPy reduction method protocol in AbstractTensor.

### Attempt B: missing higher-order callback producer

- Duration: 190.3 seconds.
- Capture completed and saved `captured_program`.
- Hierarchy candidate: `native_voxel_frame`, 11 values.
- Failure: compiled input `t263` had no value in feeds or region cache.
- Endpoint path identified `step_with_dt_control_used`, tuple unpack of the
  callback result, field `mass_err`.
- Correction: propagate opaque function-table references through specialized
  callback parameters and create a real callsite edge.

### Attempt C: full callback subtree, first loop-carried failure

- Duration: 273.9 seconds.
- Planning grew to 297 hierarchical values.
- Capture and captured-program checkpoint completed.
- SSA lowering reported eight loop-carried producer shortfalls:
  `363`, `368`, `373`, `6429`, `9757`, and three reports for `6238`.
- Inspection showed loop carried pairs had collapsed to `(363,363)`,
  `(368,368)`, `(373,373)`, and repeated same-ID pairs for the pressure loop.
- Numerical regions did contain the body updates, but under distinct values
  such as `386`, `793`, `1200`, and related pressure-solver products.

### Attempt D: partial loop identity correction

- Duration: 278.6 seconds.
- Latest checkpoint key: `d26b194456cf5421f135cc8a2f96103e86dee93368515f992b94186f70beee0c`.
- All four checkpoint stages were saved before failure.
- A generic nested retained-loop regression passes and proves distinct update
  and initial IDs in that simpler hierarchy.
- The full fluid shortfalls fell from eight to six: values `6429` and `9757`
  were repaired; `363`, `368`, `373`, and three duplicate `6238` reports
  remain.
- Therefore the current exclusion of loop-updated endpoints from ordinary
  `control_alias_sources` is directionally correct but incomplete.

Optional GLSL hierarchy emission also reports an unsupported/invalid `pad`
fragment for a `(4,4,1) -> (5,4,1)` shape. This is explicitly optional on the
Fortran precompile path and is not the current terminal failure. Do not confuse
it with the SSA loop-carried blocker.

## Checkpoints and resume points

All checkpoints are ignored build artifacts under:

`C:\dev\Powershell\turing\build\native-voxel-fluid-full-physics-probe\aot-checkpoint\aot-checkpoints`

Keys present at handoff:

| Key | Files | Approximate bytes | Notes |
|---|---:|---:|---|
| `a5cf45807f70757328b61f97114eeb159cb7dd8bea94deaaaee6022c813d2256` | 6 | 60,589,631 | earlier source/frontend/compiled plan |
| `0c7333d396f244050481c7e9791201e421571f10fe493d1a8baeadfecd6a7cd9` | 8 | 62,582,884 | reduction-fixed captured program, before higher-order linkage |
| `c48d863eff5074f7d65f6f321e643d523e1c29d76121a994f5bde1b4941cef91` | 8 | 92,390,393 | higher-order callback linked, eight loop shortfalls |
| `d26b194456cf5421f135cc8a2f96103e86dee93368515f992b94186f70beee0c` | 8 | 92,389,637 | partial loop correction, six loop shortfalls |

Each complete key has sidecar JSON and pickle payloads for `source_graph`,
`frontend`, `compiled_plan`, and `captured_program`. Implementation hashes are
part of the key. Editing capture/planning code will intentionally miss the
old key and create a new one; reverting to a matching implementation allows a
resume. Do not delete these until the full compile is working or the user asks
for cleanup.

## Loop-carried identity analysis

The precompile SSA lowerer is behaving correctly when it reports the current
failure. For each `LoopBlock.carried_aliases` pair it creates:

1. the preheader initial value;
2. a distinct backedge `updated_value` object;
3. a header Phi selecting initial or updated;
4. a validation that some body instruction produced that exact updated object.

The hierarchy composer currently creates `control_alias_sources` from
`ControlProgram.value_aliases`. Ordinary aliases can be collapsed. A loop
backedge cannot: it may share storage with the initial value, but it is a
distinct SSA version whose producer is inside the loop.

The first repair excluded endpoints named as `LoopBlock`/`WhileBlock` updates
from `control_alias_sources`. This fixed pressure values `6429` and `9757`, but
the remaining diagnostics reveal two additional forms:

- In `_cg_helmholtz_face`, global values `363`, `368`, and `373` correlate the
  caller result, local `copy`, `IndexedStore`, `LoopResult`, and nested
  `_helmholtz_face_apply` input. The actual body result values include `386`,
  `793`, and `1200`. A `LoopResult`/`IndexedStore` control alias is still being
  selected as the canonical update before the lexical loop pair is globally
  rewritten.
- In `_cg_poisson_cc_rhs`, `6238` correlates `zeros_like`, `nan_to_num`,
  `IndexedStore`, `LoopResult`, and the caller result. Three carried entries
  collapse to the same global value. Regions 207, 210, and 212 produce `6238`,
  but not necessarily inside the lexical body/path for every collapsed entry.
  The pair multiplicity must be retained until Phi construction rather than
  deduplicated by global storage identity.

The next investigation should compare the local pre-composition
`LoopBlock.carried_aliases` with the projected pairs inside
`compose_hierarchical_control`, specifically through:

- `control_alias_sources` construction;
- `leaves()` handling for `LoopResult` and `IndexedStore`;
- `canonical_global()` and projection redirects;
- hierarchy `value_table` correlation of loop update endpoints;
- `ControlProgram.value_aliases` versus lexical `carried_aliases`.

The desired invariant is:

```text
storage alias: initial arena address == updated arena address
SSA identity:  initial value version != updated value version
control edge:  Phi(initial, updated) owns the temporal alias
```

Do not repair this by suppressing the shortfall, treating the update as an
external argument, lowering pressure/viscosity iteration counts, or assigning
the captured value as a constant.

## Tests run and results

Passing focused tests during this session:

- `tests/test_canonical_forward_and_backward.py`: 19 passed.
- NumPy reduction protocol test covers `np.sum`, `np.max`, and `np.mean` with
  `axis`/`keepdims` on AbstractTensor.
- Higher-order function parameter linkage and literal callsite specialization:
  2 passed.
- Nested retained-loop distinct backedge/SSA test: 1 passed.
- Parametric card program plus existing card graph and WASM class coordinator:
  15 passed.
- Voxel fluid engine focused tests: 4 passed.
- Fortran/C layout and feedback tests: 2 passed.
- Early-return source-name test: passed.
- Dotted state-feedback test: passed after source-name correction.

The full repository suite was not run. The full native fluid compile remains
red at precompile SSA loop-carried validation, so there is no new executable or
visual result to claim.

## Known limitations and risks

1. The parametric card program is retained metadata and validation today; the
   native shell still invokes the composed compiled entry point rather than a
   separately schedulable card-runtime ABI.
2. Controller state beyond `dt_initial` is not persistent between native
   frames. Full class-state capture should expose mutable controller fields as
   resident ports and feed them back explicitly.
3. AbstractTensor reduction `out` remains intentionally unsupported. A proper
   arena-output contract should model aliasing explicitly rather than emulate
   arbitrary NumPy mutation.
4. The latest loop-identity patch is partial. Keep its focused regression, but
   do not describe the full solver issue as fixed.
5. The optional GLSL path still cannot emit at least one `pad` stage with the
   required shape transformation. It does not block Fortran precompile today.
6. The worktree contains unrelated concurrent changes in compiler and machine
   runtime files. Broad staging would mix projects and authorship.
7. A 4x4 compile takes roughly 4.5 minutes after the hierarchy grew to 297
   values. Use the small target until SSA and Fortran emission are green.

## Recommended next steps

1. Load or reproduce the latest captured program and log local and global
   carried pairs for the four remaining solver loops before changing code.
2. Preserve lexical backedge identities through `leaves()` and hierarchy
   redirects; let the loop Phi, not ordinary alias reduction, merge storage.
3. Add a focused regression modeled on `_cg_helmholtz_face`: nested call in a
   retained loop, indexed/in-place update, caller result alias, and distinct
   body-produced backedge.
4. Add a second regression modeled on `_cg_poisson_cc_rhs` where multiple
   carried variables share an arena lineage but must retain separate versions.
5. Rerun the 4x4 compile and require zero SSA shortfalls. Then inspect emitted
   Fortran control, especially the outer `run_superstep` while loop and CG loop
   Phis, before launching.
6. Run one bounded native frame and compare all state outputs and `dt_out`
   against the Python source for the same 4x4 initial state.
7. Only after parity, build the high-resolution continuous window and extend
   feedback to controller class state.
8. Commit by explicit path/hunk after separating concurrent work. Keep build
   checkpoints ignored and do not commit logs or images into Turing.

## Resume commands

Focused regressions:

```powershell
cd C:\dev\Powershell\turing
python -m pytest `
  tests/test_canonical_forward_and_backward.py `
  tests/test_loop_composer.py::test_first_class_function_parameter_becomes_a_parametric_callsite_edge `
  tests/test_process_graph_shell.py::test_nested_retained_loop_preserves_distinct_backedge_ssa_identity `
  tests/test_parametric_card_program.py `
  tests/dt_system/test_voxel_fluid_engine.py -q
```

Full diagnostic compile (allow up to one hour):

```powershell
cd C:\dev\Powershell\turing
python -m src.compiler.native_voxel_fluid `
  --output build/native-voxel-fluid-full-physics-probe `
  --width 4 --height 4 --compile-only
```

## Prompt history

> "1. you should've implemented the span operator present in other backends if you wanted to fix that, not change how my code works, it works the way it works specifically to use only preallocated arenas. when tensor accomodation is set up like it is for some other backends, you won't compile with billions of literals you'll just get a zero init so you need to undo what you did and then fix the fortran backend to handle span and indexing"

> "2. use the compiler process not bespoke garbage"

> "okay part of this is I didn't know there was a difference, forget columnar, let's use the repo fluid sim okay and make it into a fortran and see how it looks, using the dt system"

> "did you ignore me and not use dt manager, the adaptive dt system that subbatches to error metrics, and did you use the real repo fluid sim or did you make one"

> "it could take up to an hour it will not finish if you don't let it"

> "attempt to address these failings and improve your adaptation, do not disable things, and consider the possibility of designing how, perhaps, you might capture all of dt system into one program, all of abstract tensors, etc. using non-single-shot capture like for classes, constructing a system of cards that can run the whole thing not just faithfully but parametrically. it would give you a chance to iron out problems that might exist with the multi card coordinator engine"

> "can you now transition to a handoff document and extensive session logging in speaktome"

