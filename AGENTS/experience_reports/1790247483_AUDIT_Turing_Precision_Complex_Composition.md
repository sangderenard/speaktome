# Turing precision and complex composition audit

**Date:** 2026-09-24
**Title:** Precision and complex systems as the basis for rational composition

## Scope

Audit the eager and compiler-facing `Precision` and `ComplexPrecision`
systems in `turing` before adding rational tensors. The governing requirement
is that precision, complex, and rational features can occur in any unordered
combination, with one canonical concrete type for each feature set and
operand-order-independent promotion.

No implementation was changed during this audit. Existing unrelated working
tree edits in `src/transmogrifier/graph/graph_express2.py` and
`tests/test_abstract_tensor_topological_reducer.py` were left untouched.

## Methodology

- Read the workspace, Turing, tensor, compiler, testing, and guestbook agent
  instructions.
- Read `src/common/tensors/extended_precision.py`, the AbstractTensor dispatch
  seam, the topological reducer's authored-precision lowering, the compiler
  precision policy and exact lowering transaction, and their focused tests.
- Inspected the commit that introduced `ComplexPrecision`.
- Exercised the existing eager promotion matrix for `AbstractTensor`,
  `Precision`, and `ComplexPrecision` across `+`, `-`, `*`, and `/` in both
  operand orders.
- Ran the focused precision and complex tests with the installed Python 3.11
  environment after discovering that the repository `.venv` points at a
  removed Python 3.10 interpreter.

## Detailed Observations

### 1. Precision owns representation, width, and numerical truth

`Precision` is not an `AbstractTensor` subclass. It owns a leading limb axis
whose remaining axes retain the caller-declared shape. `Precision.of` is the
promotion boundary; `collapse` is the named rounding boundary. Width
combination is explicitly "widest".

The eager basic arithmetic is centralized in `Precision.dispatch`. Ordinary
tensors and scalars become a leading value limb plus exact-zero tail limbs.
Addition and subtraction use error-free summation and renormalization;
multiplication uses every limb pair and an error-free product; division uses
expansion long division. The extra endorsed surface (`abs`, selection-based
minimum/maximum, floor, square root, power, proof-core transcendental
functions, sum, and mean) is written in terms of whole expansions or
individual limbs where the operation is exact.

The type deliberately refuses unendorsed tensor methods rather than exposing
its packed representation to shape-unaware operations.

### 2. AbstractTensor knows how to yield to wrapper types

`AbstractTensor` has one explicit reflected-dispatch seam:
`_defers_to_reflected` returns true for both `Precision` and
`ComplexPrecision`. Thus an ordinary tensor on the left returns
`NotImplemented`, allowing the wrapper's reflected operator to absorb it.

The internal `_apply_operator` understands `Precision` width and invokes
`Precision.dispatch`. It does not understand complex composition directly.

### 3. ComplexPrecision is a composite of two Precision coefficients

`ComplexPrecision` holds `real` and `imag`, each normalized to the widest
`Precision` coefficient width. It refuses native complex numbers as
individual coefficients because the real error-free transforms cannot be
applied to an already-compound native complex multiplication.

`ComplexPrecision.of` accepts:

- another `ComplexPrecision`, preserving the widest requested width;
- a two-item real/imaginary pair;
- a Python complex scalar;
- a native complex tensor, split through `AbstractTensor.real/imag`; or
- a real tensor/scalar, promoted with an exact zero imaginary component.

Its `_pair` method is the local swallowing rule. Basic complex algebra is then
expressed through the two owned precision coefficients. This is already the
manual-composite-type pattern required for the unordered set
`{complex, precision}`.

### 4. Existing eager promotion is asymmetric

The observed promotion matrix for all four basic operators is:

| Left | Right | Result |
|---|---|---|
| tensor | precision | `Precision` |
| precision | tensor | `Precision` |
| tensor | complex-precision | `ComplexPrecision` |
| complex-precision | tensor | `ComplexPrecision` |
| complex-precision | precision | `ComplexPrecision` |
| precision | complex-precision | failure |

Every `Precision <op> ComplexPrecision` case failed with
`AttributeError: 'ComplexPrecision' object has no attribute 'track_time'`.
`Precision.__add__` and its siblings commit immediately to
`Precision.dispatch`; they do not yield to the more expressive composite.
`Precision.width_of` also recognizes only `Precision`. Consequently the
current pair is not a commutative promotion system even though the complex
wrapper can swallow precision when complex is the left operand.

### 5. Compiler support belongs to Precision, not directly to ComplexPrecision

The source reducer recognizes both class names during general call-result
classification, but `lower_python_precision` intentionally tests only real
`Precision`. It converts `Precision.of`, arithmetic, selected methods, and
`collapse` into the repository's `precision_*` vocabulary.

The production compiler transaction `apply_precision_pipeline` carries limb
facts through SSA, applies only catalogued exact identities, records section
contracts and backend obligations, expands each abstract precision operation
to ordinary SSA values (one per limb), refreshes call records, and refuses to
return if a `precision_*` operation survives.

The test named
`test_source_compiler_lowers_complex_precision_as_two_real_expansions` does
not compile `ComplexPrecision`. Its source signature contains four
`Precision[2]` parameters (`ar`, `ai`, `br`, `bi`) and manually spells complex
multiply and divide. Its ABI likewise declares four values whose Python type
is `Precision`. This proves that complex algebra manually decomposed into
real precision coefficients can reach LLVM; it does not prove direct
`ComplexPrecision` source, ABI, value identity, return, or call-boundary
support.

### 6. Focused verification status

Command:

```powershell
py -3.11 -m pytest -q tests/test_precision_surface.py tests/test_complex_precision.py -k "not torch"
```

Result: 34 passed, 1 failed, 1 deselected in 20.54 seconds.

The failure is the compiler-facing complex-precision test. Its receipt reports
zero lowered `precision_div` operations where the assertion expects two. The
test stopped at that receipt assertion, so this run does not establish LLVM
emission or execution for the current working tree. The eager precision and
non-Torch eager complex tests passed.

The repository `.venv` is not runnable on this machine because its
`pyvenv.cfg` names a removed Python 3.10 installation. No package installation
or environment rewrite was performed; the already-installed Python 3.11
environment supplied pytest, NumPy, and NetworkX.

## Analysis

The sister systems establish two reusable rules.

First, a numerical modifier must own every operator that understands its
representation and expose explicit promotion and collapse boundaries.
Second, a composite type should be algebra over already-correct component
types: `ComplexPrecision` does not duplicate limb arithmetic; it composes two
`Precision` values.

The missing rule is canonical cross-wrapper promotion. Pairwise Python dunder
methods currently decide precedence locally. That makes the result depend on
which wrapper happens to be the left operand. Adding rational methods in the
same isolated style would multiply these gaps.

For the three requested features, order is disregarded and the concrete
non-empty feature sets are:

1. `{precision}`
2. `{complex}`
3. `{rational}`
4. `{complex, precision}`
5. `{rational, precision}`
6. `{complex, rational}`
7. `{complex, rational, precision}`

The existing `ComplexPrecision` is case 4. Rational's stated precedence means
that any mixed operation containing the rational feature must normalize to
case 3, 5, 6, or 7 as appropriate. "Swallow" must work in both operand orders.
Precision width remains an independent widest-width fact inside cases 4, 5,
and 7; it must not compete with rational or complex as though only one wrapper
may win.

## Recommendations

1. Pin the seven canonical feature sets and the complete bidirectional
   promotion matrix in eager tests before adding rational arithmetic.
2. Repair the existing `{precision}` + `{complex, precision}` asymmetry as the
   first promotion regression. This is the smallest proof that normalization
   is independent of operand order.
3. Implement rational as the owner of numerator/denominator algebra and make
   the rational-containing composite types absorb non-rational operands as
   denominator-one values, matching the user's stated precedence.
4. Keep precision arithmetic in `Precision`; composite types should delegate
   coefficient work to it as `ComplexPrecision` already does.
5. Give each concrete feature-set type explicit source descriptors, ABI
   decomposition, call/return identity, and compiler tests. Do not treat a
   manually expanded component program as proof that the wrapper itself is
   supported.
6. Resolve or baseline the current zero-`precision_div` compiler receipt
   before using the complex compiler test as the rational-composition model.

## Prompt History

> abstract tensor needs a rational type, similar to how we did complex. two tensors hold the numerators and the denominator and the rationals take precedent and swallow non rationals, they override some operators. are you with me?

> start with an audit of the precision and the complex systems, they are sister functionalities to rationals

> the tricky thing is that precision, complex, and rational need to be able to be combined in any set

> i guess we could do this sort of easily if we make each object manually of the 3! possibilities, i mean, the set disregarding order
