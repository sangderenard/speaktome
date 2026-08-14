# Turing full-physics capture attempt log

**Date:** 2026-08-05
**Title:** Four 4x4 repository-fluid AST/AOT capture attempts

## Command

```powershell
cd C:\dev\Powershell\turing
python -m src.compiler.native_voxel_fluid --output build/native-voxel-fluid-full-physics-probe --width 4 --height 4 --compile-only
```

The command intentionally uses the ordinary Python AST/AOT compiler, registered
Fortran backend, common C shell, real `VoxelMACFluid`, and `dt_system`. It does
not request a bespoke Fortran implementation.

## Condensed chronological log

### Reduction protocol failure

```text
full-physics capture reached VoxelMACFluid.compute_metrics
TypeError: AbstractTensor.sum() got an unexpected keyword argument 'axis'
```

Resolution: normalized NumPy `axis`, `keepdims`, `dtype`, and `out` method
keywords at the AbstractTensor reduction surface. Canonical tensor/autograd
suite: 19 passed.

### Missing callback result producer

```text
0.760s aot: applying constant map
0.773s aot: building process graph (second, independent build)
4.437s aot: saving source-graph checkpoint
4.602s aot: reducing abstract tensor topology
5.526s aot: propagating bound planner specializations
5.527s aot: building map dependency regions
5.533s aot: saving frontend checkpoint
5.698s aot: strategizing glsl deployment (scheduling/planning pass)
21.908s aot: compile_process_graph (usually the largest phase)
22.573s aot: saving compiled-plan checkpoint
85.193s aot: capturing fused programs
175.196s aot: hierarchy candidates
  (native_voxel_frame, 11 values, 10 outputs)
175.217s aot: rebuilding region program feed provenance
175.218s aot: saving captured-program checkpoint
ValueError: compiled input 't263' (()) has no value in feeds or the captured region cache
endpoint field_path=('mass_err',)
function=step_with_dt_control_used
```

Inspection established that `advance` arrived as a
`_CompiledStructuralFunction`, but the call `advance(state, dt)` had no
hierarchy callsite or result bindings. Resolution: propagate an opaque
function-table reference through safe callsite specialization.

### First full callback hierarchy

```text
0.621s aot: applying constant map
0.635s aot: building process graph (second, independent build)
5.084s aot: saving source-graph checkpoint
5.259s aot: reducing abstract tensor topology
6.220s aot: propagating bound planner specializations
6.221s aot: building map dependency regions
6.225s aot: saving frontend checkpoint
6.430s aot: strategizing glsl deployment (scheduling/planning pass)
31.874s aot: compile_process_graph (usually the largest phase)
32.815s aot: saving compiled-plan checkpoint
122.558s aot: capturing fused programs
253.123s aot: hierarchy candidates
  (native_voxel_frame, 297 values, 10 outputs)
253.172s aot: rebuilding region program feed provenance
253.185s aot: saving captured-program checkpoint
273.9s total

precompile-to-SSA lowering shortfalls:
- loop_carried updated 363: no producer inside body
- loop_carried updated 368: no producer inside body
- loop_carried updated 373: no producer inside body
- loop_carried updated 6429: no producer inside body
- loop_carried updated 9757: no producer inside body
- loop_carried updated 6238: no producer inside body (reported three times)
```

Relevant captured identities:

```text
_cg_helmholtz_face loops:
  carried ((363, 363),) while body call result includes 386
  carried ((368, 368),) while body call result includes 793
  carried ((373, 373),) while body call result includes 1200

_cg_poisson_cc_rhs loop:
  carried ((6429,6429), (9757,9757),
           (6238,6238), (6238,6238), (6238,6238))
```

Region ownership confirmed that these were not absent numerical computations.
For example, region 35 produces `363`, region 38 consumes it, and loop-body
regions produce the later update lineages. The failure is lexical SSA identity
projection, not missing physics.

### Partial loop alias correction

The hierarchy builder was changed so endpoints explicitly named as lexical
loop updates are not folded by ordinary `ControlProgram.value_aliases` during
`control_alias_sources` construction.

Focused regression:

```text
tests/test_process_graph_shell.py::
  test_nested_retained_loop_preserves_distinct_backedge_ssa_identity
1 passed
```

Latest full run:

```text
0.650s aot: applying constant map
0.660s aot: building process graph (second, independent build)
4.621s aot: saving source-graph checkpoint
4.824s aot: reducing abstract tensor topology
5.960s aot: propagating bound planner specializations
5.961s aot: building map dependency regions
5.968s aot: saving frontend checkpoint
6.151s aot: strategizing glsl deployment (scheduling/planning pass)
33.474s aot: compile_process_graph (usually the largest phase)
34.248s aot: saving compiled-plan checkpoint
129.535s aot: capturing fused programs
258.507s aot: hierarchy candidates
  (native_voxel_frame, 297 values, 10 outputs)
258.549s aot: rebuilding region program feed provenance
258.556s aot: saving captured-program checkpoint
278.6s total

precompile-to-SSA lowering shortfalls:
- loop_carried updated 363: no producer inside body
- loop_carried updated 368: no producer inside body
- loop_carried updated 373: no producer inside body
- loop_carried updated 6238: no producer inside body (reported three times)
```

The change repaired `6429` and `9757`, reducing eight reports to six. It did
not fix the complete solver hierarchy.

## Checkpoint inventory

```text
0c7333d396f244050481c7e9791201e421571f10fe493d1a8baeadfecd6a7cd9
  8 files, 62,582,884 bytes
a5cf45807f70757328b61f97114eeb159cb7dd8bea94deaaaee6022c813d2256
  6 files, 60,589,631 bytes
c48d863eff5074f7d65f6f321e643d523e1c29d76121a994f5bde1b4941cef91
  8 files, 92,390,393 bytes
d26b194456cf5421f135cc8a2f96103e86dee93368515f992b94186f70beee0c
  8 files, 92,389,637 bytes
```

Complete keys contain `source_graph`, `frontend`, `compiled_plan`, and
`captured_program` JSON/pickle pairs.

## Prompt history

> "it could take up to an hour it will not finish if you don't let it"

> "attempt to address these failings and improve your adaptation, do not disable things, and consider the possibility of designing how, perhaps, you might capture all of dt system into one program, all of abstract tensors, etc. using non-single-shot capture like for classes, constructing a system of cards that can run the whole thing not just faithfully but parametrically. it would give you a chance to iron out problems that might exist with the multi card coordinator engine"

> "can you now transition to a handoff document and extensive session logging in speaktome"

