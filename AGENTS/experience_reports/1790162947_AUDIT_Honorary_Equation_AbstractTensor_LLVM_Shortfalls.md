# Honorary equation AbstractTensor-to-LLVM shortfall audit

**Date:** 2026-09-23

## Scope

Read-only compiler audit of selected difficult equations from
`engine_toy/honorary_engine_equation_catalogue.py`. Product and compiler source
were not changed. Each applicable equation was tested through the established
route:

1. `compile_sympy_equations` to repository symbolic SSA;
2. `symbolic_abstract_tensor_source` to materialize the AbstractTensor Python;
3. `lower_ast_source_to_ssa` under `batch_contract(..., batch=4)` so the input
   shape is explicit;
4. `emit_ssa_function_to_llvm`.

An initial direct symbolic-SSA emitter probe was stopped when the user clarified
that the AbstractTensor stage and its shape parameter are mandatory.

## Results

Complete through LLVM with zero shortfalls:

- `eq_B12_2` (Cunningham slip correction; `exp` and reciprocal)
- `eq_F17_12` (Brillouin magnetization; `coth` exact respelling)
- `eq_NS10_1` (compressible orifice flow; variable powers and square root)
- `eq_H4_1` (many-body Hamiltonian; symbolic sums and absolute values)

AbstractTensor materialization produced invalid Python parameter spellings:

- `eq_HO1_1`: names such as `C_{in}` and `F_{Far}`
- `eq_LA3_17`: name `omega_{pe}`
- `eq_O1_1`: name `\\theta`
- `eq_N6_1`: names such as `r_b^0` and `theta_a^0`; the materialized source
  literally contains a signature parameter `r_b^0` and expressions using `^`.

The symbolic SSA itself was valid in all four cases. The failure is at the
SSA-to-AbstractTensor Python naming boundary, which currently uses authored
SymPy display names as Python identifiers without a reversible name mapping.

Symbolic-ingestion refusals:

- `eq_T4_1`: an `Integral(..., x)` over a bare domain survives symbolic
  integration. The compiler correctly refuses to guess a measure and requests
  finite bounds or a declared `Domain`.
- `eq_LA11_17`: no SymPy-to-ProcessGraph translation for `ImaginaryUnit`.
- `eq_F17_14`: `M_s` is both output and RHS input; the simultaneous equation
  compiler refuses implicit recurrence.
- `eq_F18_20`: differentiating the opaque applied function `M_12(z_c)` reaches
  ProcessGraph autograd, which has no graph-native adjoint rule for `call`.

Whole-program AbstractTensor lowering refusals:

- `eq_NS6_3`: the undeclared applied function `rho(t)` materializes as a Python
  call. Its value becomes an undefined feed to a planned region. The full-native
  contract reports value id 14 in `if_merge.1` rather than letting it pass.
- `eq_B4_1`: the undeclared applied function `p_s(T_s)` similarly becomes an
  undefined planned-region feed (value id 21 in `entry`).

LLVM shortfall after successful AbstractTensor lowering:

- `eq_F19_8`: one shortfall, operation `cross`: `operation has no repository
  LLVM emission`. The catalogue's `cross` is an untyped placeholder over
  scalar-shaped inputs, so it carries no vector-axis contract into LLVM.

## Interpretation

The sampled hard scalar/transcendental laws are healthier than their printed
complexity suggests. The most reusable frontier is the identity boundary
between authored SymPy names and legal Python parameters: it needs reversible,
concordance-visible name transport rather than equation-by-equation renaming.
Opaque applied functions need either linked implementations or explicit input
parameters; derivatives of such calls additionally need an adjoint contract.
The domain, recurrence, and vector-shape refusals are useful semantic refusals,
not arithmetic gaps to paper over.

## Prompt History

> can you go through some difficult equations in the honorary equation set in engine toy i'm the root repo and hand pick some you think won't compile, see where their shortfalls are

> the way these compile is by going to abstract tensor first to get their shape parameter and then to llvm

