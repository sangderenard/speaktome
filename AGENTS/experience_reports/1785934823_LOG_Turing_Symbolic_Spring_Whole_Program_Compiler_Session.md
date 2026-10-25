# Turing symbolic spring whole-program compiler session log

**Date:** 2026-08-05  
**Epoch:** 1785934823  
**Repository:** `C:\dev\Powershell\turing`  
**Status:** active handoff; generic runtime-string compilation is not proven

## User objective and architectural constraints

The requested stress test is to compile one ordinary Python entrypoint which
does all of the following: accepts an arbitrary SymPy expression string,
parses and simplifies it, forms the bound-spring ProcessGraph, initializes the
physics/image runtime, discovers the shader code actually compiled by OpenGL,
and runs the program. The application must enter the compiler once through its
Python AST. Compiler improvement must be general; no expression-specific
equation, reduced numeric trace, custom harness, or alternate control-flow
interpreter is acceptable.

The user repeatedly clarified these points:

> "all you do is put the python function that initiates the image with the
> spring graph and physics the shader and the solve from sympy, all of it starts
> from one function, and you compile that function from ast"

> "you are NOT to make bespoke harness ... your job is SPECIFICALLY to make
> this work through first principles improvements of the general automatic
> case"

> "Can you please compile it in general, not as a baked in sympy equation? so
> it will work with anything you provide it?"

> "another agent is currently working ... so that individual classes like the
> dt system can be captured in general, in entirity, not as a reduced run"

The user also requested planning, a persistent goal, frequent path checks, and
resume saves after long compilation attempts.

## Entry point established

`src/rendering/symbolic_spring_image.py` contains:

```python
def run_symbolic_spring_image(expression_text: str) -> None:
    expression = sympy.sympify(expression_text, evaluate=False)
    source_graph = ProcessGraph(materialize_memory=False)
    source_graph.build_from_expression(expression)
    process_graph, _reduction = symbolically_reduce_process_graph(
        source_graph,
        aggressive=True,
    )
    shader_sources = load_fluxspring_graph_shaders()
    run_precompiled_graph(
        process_graph.graph_accessor(),
        duration=math.inf,
        shader_sources=shader_sources,
    )
```

This is the sole application entrypoint. Backend lowering is required to begin
from its source AST.

## Chronological work log

### 1. Existing compiler and application discovery

The repository already had the necessary conceptual route: ProcessGraph AST
construction, loop and branch planning, hierarchical `PlanClosure`/`PlanCall`,
backend-neutral `ControlProgram`, captured numerical cards, precompile-to-SSA,
Fortran, GLSL, WebGPU, and native shell infrastructure. The work therefore
focused on exposing the whole entrypoint to that route rather than constructing
an application-specific backend.

The live FluxSpring shader source was found inside
`src/common/tensors/autoautograd/spring_async_toy.py`, where
`LiveVizGLPoints` calls PyOpenGL's `compileShader(vsrc, GL_VERTEX_SHADER)` and
`compileShader(fsrc, GL_FRAGMENT_SHADER)`.

### 2. Shader extraction first tranche

`src/compiler/shader_extractor.py` was added earlier in the session to parse
Python source and extract literal strings at actual `compileShader` call sites.
`src/rendering/opengl_render/fluxspring_shader.py` loads the live vertex and
fragment shader without importing the large toy runtime. The application
entrypoint calls this loader and passes the resulting pair into the existing
renderer.

This was verified against the real FluxSpring source, not a copied fixture.

### 3. Wrong structural compiler turn and full reversion

A separate structural AST/SSA layer was briefly introduced in
`structural_ast_ssa.py` and `structural_program.py`. The user correctly objected:

> "what are you doing introducing interpretation of control flow that already
> exists in the compiler"

That design duplicated semantics already owned by ProcessGraph control
planning. It was removed completely, including tests and AOT/Fortran
integrations. A repository search found no remaining references. This is a
durable negative decision: future work must repair the existing hierarchy and
card export rather than rebuild Python control elsewhere.

### 4. Mutable runtime ABI instead of a baked discovery value

The initial successful Fortran compilation used one sample expression and was
therefore not generic. The compiler was corrected so a mutable function
parameter is not propagated as a planner specialization. Checkpoint identity
for such a parameter uses its runtime ABI description instead of its concrete
sample value.

An additional AOT invariant now checks whether each declared mutable public
parameter remains represented in the executable public-input value IDs. A
numeric parameter survives; a structural string that was evaluated and erased
raises a clear `RuntimeError` saying it was specialized out. This prevents a
captured trace from being called a general compiler result.

### 5. Durable phase checkpoints

The compilation now saves and resumes source/frontend, compiled-plan, and
captured-program phases. Checkpoints include schema and compiler implementation
digests. A reducer was added for SymPy's `_global_parameters` singleton after a
640 MB compiled plan initially failed to pickle.

The important mutable identity is:

```text
4bf0715a2f7fe214ffebff6dd719b57accd9097e45e715a9f698e162a17a2c6a
```

It contains 24,698,061-byte frontend, 640,547,324-byte compiled plan, and
233,525-byte captured program payloads. Different valid discovery expressions
mapped to the same checkpoint identity, proving the identity is no longer
keyed by the sample string.

The earlier baked checkpoint is:

```text
800c06c5a54cd927c6577c69e95413d1c17799383baab9cfcb357f6bee769327
```

It remains useful only for diagnosis. Its compiled Fortran DLL must not be
reported as satisfying arbitrary runtime expressions.

### 6. Projected iterable ABI

Retained loops over `enumerate` and destructured resident tuples/rows needed an
explicit backend-neutral projection. `ControlProgram.projected_iterable_bindings`
was added and propagated through control overlays, hierarchy namespacing, SSA,
Fortran, and GLSL row loads. Tests cover enumerate counter/value projections
and shared GLSL control-shader row ABI.

This work extended the existing loop representation; it did not add control
interpretation.

### 7. Generic capture exposed a false hierarchy success

After resuming the large compiled plan and capturing the application, hierarchy
composition logged:

```text
planned calls reference enclosing loops absent from closure control: (161, 182)
```

`compile_ast_aot` caught that exception and selected a fallback shell with only
seven straight-line numerical regions. The returned artifact claimed no
control shortfalls but had no public inputs or outputs even though
`expression_text` appeared in function metadata. This is not a valid generic
artifact.

The exact existing seam is:

- `_build_shell_hierarchy_plan` tags each `PlanCall` with loop node IDs whose
  bodies contain the callsite;
- `compose_hierarchical_control` holds pending calls until their scheduled
  region or lexical loop end;
- pending calls for loop IDs 161 and 182 found no matching loop induction in
  that closure's `ControlProgram`;
- AOT's recomposition catch allowed degraded continuation.

The correct investigation is closure ownership/namespacing in these existing
objects, not a new AST runner.

### 8. Concurrent retained-loop version correction

The user warned that another agent was concurrently refining loop control and
asked for change-date inspection before continuing. File timestamps showed the
core loop composer/hierarchy files last changed around 06:27, while active
edits landed in `glsl_deployment_strategy.py` at 07:53 and
`tests/test_process_graph_shell.py` at 07:54.

The concurrent agent supplied this diagnosis:

> "hierarchy identity reduction treated a loop's backedge alias like an
> ordinary storage alias and collapsed (updated, initial) into
> (initial, initial)"

Its correction keeps loop versions distinct in SSA while preserving resident
arena reuse as an allocation policy. The focused test checks both distinct
carried IDs and absence of `loop_carried` lowering shortfalls. The agent then
started a complete unreduced 4x4 viscosity and pressure capture. This session
did not stop that process or edit over its live ownership area.

The backedge-alias defect may influence the whole hierarchy but is not
automatically identical to the missing lexical loop IDs. Reassess only after
the full corrected run reports.

### 9. Shader wrapper extraction work-in-progress

After separating from the concurrent loop work, this session began generalizing
shader discovery from direct PyOpenGL helpers to raw OpenGL APIs. A 337-line
internal abstract-flow implementation was added to
`src/compiler/shader_extractor.py`. It can represent function parameters,
literal strings, shader stages, tuples, alternatives, shader handles, helper
summaries, and recognized calls to `glCreateShader`, `glShaderSource`, and
`glCompileShader`.

The user then requested that effort focus on handover and extensive logging.
At that moment the new machinery had not been wired into
`extract_shader_compile_calls()`. It is therefore explicitly WIP and not an
implemented extractor feature. Existing direct `compileShader` behavior still
passes. A next agent should either connect it with tests for raw and nested
wrapper APIs or remove the dead WIP cleanly.

## Files changed by this line of work

Relevant tracked modifications at the snapshot include:

- `src/common/tensors/accelerator_backends/aot_checkpoint.py`
- `src/common/tensors/accelerator_backends/aot_compile.py`
- `src/common/tensors/accelerator_backends/glsl_backend.py`
- `src/compiler/control_source.py`
- `src/compiler/glsl_deployment_strategy.py`
- `src/compiler/hierarchical_control.py`
- `src/compiler/loop_composer.py`
- `src/compiler/precompile_to_ssa.py`
- `src/compiler/ssa_fortran_backend.py`
- `src/compiler/shader_extractor.py`
- `tests/test_aot_checkpoint.py`
- `tests/test_glsl_fused_network.py`
- `tests/test_loop_composer.py`
- `tests/test_precompile_to_ssa.py`
- `tests/test_process_graph_shell.py`
- `tests/test_shader_extractor.py`

`src/compiler/parametric_card_program.py`, its tests, `fortran_c_shell.py`, and
many reversible-machine/system-port files have concurrent owners. Preserve
them. The worktree is intentionally dirty and shared; no staging, reset, or
checkout was performed.

## Verification record

The latest small command was:

```text
py -3.11 -m pytest -q tests/test_shader_extractor.py tests/test_aot_checkpoint.py
3 passed in 4.40s
```

Earlier focused mutable-parameter, projected-iterable, SSA, and GLSL tests
passed. The structural compiler reversion was followed by two focused passing
tests. No full test suite was run because two other agents and a long capture
were active.

Environment note: `turing/.venv` and the shared historical venv launchers point
to a removed Python 3.10 installation. No manual package installation or venv
repair was attempted. The existing Windows Python 3.11 launcher had pytest and
the needed dependencies for the focused tests.

## Required next evidence

1. Record the other agent's full 4x4 retained-loop result and its checkpoint
   key.
2. Resume the symbolic spring compile against the updated compiler.
3. Demonstrate that `expression_text` is a public runtime input of the complete
   hierarchy/card artifact.
4. Compile that normal artifact to Fortran.
5. Invoke the same compiled ABI with at least two different valid expressions;
   do not recompile between them for the proof of generality.
6. Extract the actual shader sources through both `compileShader` and raw
   `glShaderSource/glCompileShader` wrapper forms.
7. Package them through the existing `turing.shader-component.v1` ABI, then
   select the supported shader-language or native raster deployment boundary.

## Lessons learned

- A successful native compiler invocation proves nothing about generality if a
  mutable structural input disappeared during discovery.
- Zero shortfalls can be false when hierarchy composition failed and a fallback
  shell was silently selected.
- SSA version identity and arena reuse are distinct concepts, especially on a
  loop backedge.
- The compiler already has control-flow ownership. Adding another AST evaluator
  obscures the actual defect and violates the requested design.
- Large compiler sessions need phase checkpoints whose implementation digest
  includes semantic helpers, not just the public orchestration function.
- In a shared worktree, timestamps and focused diffs are part of correctness.

## Final continuation addendum — 2026-08-05 08:10 America/Chicago

After the first handoff was filed, implementation continued on the independent
shader extractor seam. The user subsequently stated that this continuation was
coasting on the persistent goal and ordered: file experiences and continuation,
then log off. No further feature work or tests were run after that instruction.

The post-handoff work did the following:

1. Connected `_function_summaries()` and `_ShaderFlow` to the public
   `extract_shader_compile_calls()` result.
2. Added focused raw OpenGL and nested-wrapper tests.
3. Found and fixed loss of direct sites caused by indexing all methods only by
   simple name (`__init__` collisions). Every function definition is now
   analyzed independently; only uniquely named functions participate in
   interprocedural helper resolution.
4. Found and fixed skipped nested compile calls inside unknown host calls such
   as `compileProgram(compileShader(...))` by traversing argument expressions
   without interpreting the outer API.
5. Found and fixed lazy-import placeholders such as
   `GL_VERTEX_SHADER = None` shadowing recognized stage enum identities.
6. Verified the actual FluxSpring and raw renderer extraction paths:

   ```text
   py -3.11 -m pytest -q tests/test_shader_extractor.py
   5 passed in 5.38s
   ```

7. Ran a read-only real-source coverage scan. It extracted 2 FluxSpring
   shaders, 6 canonical renderer shaders, 7 OpenGL demo shaders, and 4 particle
   shaders. Dynamic/generated-source hosts correctly emitted no static literal
   artifacts.
8. Added a final packaging patch with `ExtractedShader.to_mapping()`,
   `ExtractedShaderBundle`, deterministic manifest serialization, selection by
   stage/language, and recursive `discover_shader_compile_calls(root)`.

The packaging patch in item 8 was applied immediately before the user's stop
instruction and has **not been tested**. The first continuation action is the
single focused command:

```text
py -3.11 -m pytest -q tests/test_shader_extractor.py
```

If it fails, repair only the bundle/discovery patch. Do not resume the long
symbolic-spring capture until the concurrent retained-loop owner reports its
full 4x4 result and file timestamps are checked again.

Final architectural state remains unchanged: one Python entrypoint, existing
ProcessGraph control/hierarchy/cards, general mutable runtime expression ABI,
normal SSA/Fortran lowering, and shader discovery from actual compiler calls.
No bespoke host harness or second AST control interpreter is authorized.
