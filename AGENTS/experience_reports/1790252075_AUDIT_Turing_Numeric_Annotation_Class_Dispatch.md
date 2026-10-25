# Turing numeric annotation class dispatch

**Date:** 2026-09-24
**Title:** Composite annotations now enter the existing class dependency graph

## Scope

Audited the source-to-SSA boundary for a repeated division whose operands are
annotated `ComplexRationalPrecision[2]`. The question was whether complex,
rational, and precision behavior should be reconstructed in a new SSA algebra
pass or obtained by compiling the existing wrapper dependency graph.

## Methodology

Traced the annotation from `ProcessGraph.build_from_ast` through lexical value
normalization and `lower_class_operator_calls`. Compared the normalized
numeric descriptor, class navigation table, input-node identity, and selected
operator reference. Then ran the sanctioned `lower_ast_source_to_ssa` entry on
the direct repeated-division probe and retained the first downstream failure.

## Detailed Observations

The annotation spelling and normalized `NumericFeatureDescriptor` survived
ingestion, but `_normalize_lexical_values` assigned parameter class identity
only for a bare `ast.Name` naming an already ingested class. A generic external
annotation such as `ComplexRationalPrecision[2]` therefore never stamped
`result_class_ref` on its input nodes. The existing receiver-edge operator
dispatcher consequently could not select `__truediv__`.

Numeric annotations now use their normalized descriptor's `type_name` when
that exact class is present in the class navigation table. Input nodes publish
the class identity, descriptor receipt, and Precision width and commit the
same fact to source-value class concordance.

The compiler entry now admits the exact eager numeric class named by an
annotation and its source MRO. Class-table construction carries inherited
methods without replacing derived overrides. This is required because
`ComplexRationalPrecision` owns storage and construction while
`_ComplexRationalBase` owns `__truediv__`.

Retained external class methods initially lost their defining module's free
names when source pursuit installed `self`/`cls`. That made the real
`_composed_binary` dependency appear dynamic. Retention now preserves the
class's defining-module lexical bindings and adds owner context to them.

The focused reducer regression proves that both annotated inputs and the
division result are `ComplexRationalPrecision` values with two limbs and the
complete feature descriptor. The division is a `Call` to the inherited
`__truediv__` function-table reference. It is no longer an ordinary scalar
`Div`.

The first full direct-wrapper compile faithfully pursued `_composed_binary`,
`_promote_numeric`, `_same_binary`, the rational classes, and Precision. That
unspecialized closure reached `_tensor_descriptor_rule`, where a heterogeneous
structural dispatcher tuple was mistaken for possible numeric literal data.

The follow-up now specializes retained operator source only when every
observed wrapper use of that operation has identical descriptors on both
operand edges. For the repeated CRP division, the retained inherited dunder
routes directly to the existing `_ComplexRationalBase._binary_same` method.
The generated repository module contains `_binary_same` dependencies and no
`_composed_binary` function. Mixed/unproved operations retain the general
promotion route. The tensor classifier was not changed.

The compile now completes repository SSA. Its remaining expected failure is
the physical composite ABI: the root still has wrapper-shaped parameters and
results instead of 16 input limbs and 8 output limbs, so the inner Precision
operations do not yet reach the established limb transaction.

Numeric type facts now cross an authored call edge through
`source_value_class_concordance`. The caller's concorded class identity and
limb width are authoritative; copied node attributes are only a view. This
found a real stale-width case immediately: a call-result node advertised one
limb while its receipt said two. The transfer now reads the receipt and
commits the same fact for the callee's exact `self` and `other` formals.

The repository precision lowering no longer imposes the former f64/f32 hard
ceilings of four/eight limbs. Its expansion algorithms were already generic
in width. The old ladders remain conservative automatic-planning defaults and
performance guidance, while authored `Precision[n]` accepts wider `n`.
Automatic/tuned precision remains optional: the default search bound stays
four, and a caller must explicitly supply a wider `max_limbs` policy.

## Verification

```text
py -3.11 -m pytest -q tests/test_rational_compiler.py -rxX
11 passed, 1 xfailed in 9.77s

py -3.11 -m pytest -q tests/test_abstract_tensor_topological_reducer.py::test_method_resolution_follows_an_authored_function_returned_class tests/test_rational_compiler.py::test_generic_numeric_annotation_binds_the_existing_operator_method
2 passed in 2.79s

py -3.11 -m pytest tests/test_rational_compiler.py -q
11 passed, 1 xfailed in 10.72s

py -3.11 -m pytest tests/test_precision_pipeline.py::test_precision_pipeline_lowers_authored_width_beyond_policy_ladder tests/test_precision_pipeline.py::test_automatic_precision_ceiling_is_an_explicit_optional_policy_bound tests/test_precision_pipeline.py::test_float32_limb_sections_widen_to_eight_and_split_with_4097 -q
3 passed in 3.06s

py -3.11 -m pytest tests/test_abstract_tensor_topological_reducer.py::test_annotated_receiver_discovers_authored_method_without_instance tests/test_abstract_tensor_topological_reducer.py::test_classmethod_receiver_uses_concorded_static_class_identity tests/test_abstract_tensor_topological_reducer.py::test_complex_precision_classmethod_receiver_uses_same_concordance_rule tests/test_abstract_tensor_topological_reducer.py::test_calls_reference_separate_local_function_subgraphs tests/test_abstract_tensor_topological_reducer.py::test_bound_method_dereference_is_an_explicit_ssa_accessor tests/test_abstract_tensor_topological_reducer.py::test_method_resolution_follows_an_authored_function_returned_class -q
6 passed in 2.99s
```

The passing dependency-specialization regression proves the classifier symptom
is gone and repository SSA completes. The strict expected failure now names
only composite parameter/result ABI expansion.

## 2026-09-24 Re-audit: Interpolant Source Surface

The earlier inventory saying the rational classes lacked the interpolant's
six operations is stale. The live `extended_precision.py` now provides
rational slicing and concatenation, sign and comparisons, feature-aware
`where`, leading-axis cumulative sum, tridiagonal solve, and limb readout.
The live interpolant consumes those operations directly through
`RationalPrecision`; it does not reproduce the formulas in a backend-only
lane.

The source implementation passed its rational unit suite and a small run of
the existing exact-oracle interpolant sweep. All four interpolation methods
returned correctly rounded doubles for values, first derivatives,
antiderivatives, and bin averages in that run:

```text
py -3.11 -m pytest tests/test_rational_precision.py -q
11 passed in 17.36s

py -3.11 tools/interpolant_error_wells.py --trials 1 --limbs 2
4 methods x 4 outputs; 0 misses (39s)
```

The compiler has also advanced: component projection preserves a seven-limb
descriptor, projected Precision coefficients enter ordinary precision
lowering at width seven, and same-type rational dunders reach the existing
wrapper dependency body. The complete compiler suite now reports:

```text
py -3.11 -m pytest tests/test_rational_compiler.py -q -rxX
14 passed, 1 xfailed in 96.84s
```

The remaining xfail is still real. Running it without xfail reaches complete
repository SSA, but the module has no precision-pipeline receipt for the
composite entry point. Therefore the root does not expose the required 16
scalar inputs and eight scalar outputs. Physical composite parameter/result
ABI expansion remains the compiler-side blocker; the source-side interpolant
requirements are present.

## Recommendations

- Expand composite parameter/result boundaries from the descriptor's exact
  coefficient paths, then feed their Precision leaves into the established
  limb transaction.
- Preserve the now-compiled authoritative eager `_binary_same` dependency
  graph; do not copy its formulas into an SSA-only algebra.
- Extend identity concordance auditing to read numeric parameter/result class
  and descriptor receipts.
- After specialization, resume the direct ABI goal: 16 scalar input
  components and 8 scalar result components for width-two CRP.

## 2026-09-24 Re-audit: False Construction Identity

The completed repeated-division compile exposed 5,158 root formals, 5,156 of
them propagated private call-frame storage.  Two concordance consumers were
writing the wrong graph view of a known class identity:

- numeric parameter ingestion stamped an Input with both `result_class_ref`
  and `class_ref`;
- actual-to-formal numeric specialization repeated the same pair on `self`
  and `other` Inputs.

The compiler's established distinction is exact: `result_class_ref` says a
value is an instance, while `class_ref` says the node constructs one.  The
second spelling caused deployment to synthesize `__init__` calls recursively
through the retained wrapper dependency graph.  Both writers now retain only
`result_class_ref`; the existing `source_value_class_concordance` remains the
authoritative class/width record.

The root no longer calls `ComplexRationalPrecision.__init__`.  Its propagated
frame fell from 5,158 to 1,058 formals, the `__truediv__` call frame fell from
2,578 to 530 operands, and the direct compile fell from roughly 85 seconds to
roughly 40 seconds in the focused diagnostic.  The remaining 1,056 root
formals are declared compiler-owned workspace and are allocated internally by
the native wrappers; they are not public buffers.

Verification after the repair:

```text
py -3.11 -m pytest tests/test_rational_compiler.py tests/test_ssa_call_input_adapters.py -q -rxX
27 passed, 1 xfailed in 73.63s
```

The xfail remains the separate physical composite ABI expansion: the public
entry still accepts two scalar wrapper placeholders rather than sixteen
coefficient limbs and returns one placeholder rather than eight limbs.

## Prompt History

> right now and too some extent i expect this from the work, complex and precision are partly handled merely from faithful reproduction of the dependency graph

> proceed

> if you wanna unlock the limb count be my guest

> yes, all auto precision or tuned precision needs to be optional

> make sure the types flow through the concordance, I think this might be some of the most complex typing we ever deal with

> check if your needs from the compiler or source might be already achieved
