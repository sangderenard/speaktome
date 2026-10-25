# Machine-code C/SSA round-trip prototype

**Date:** 2026-08-02
**Project:** `C:\dev\Powershell\turing`

## Activity

Added a deliberately narrow x86-64 machine-code raising prototype. GNU
`objdump` provides instruction decoding; an explicit, fail-closed reference
vocabulary lifts two-operand `imul`, affine `lea`, and `ret` into the existing
repository SSA dataclasses. The module also emits its supported scalar SSA
subset back to C and projects SSA dependencies into a `MultiDiGraph`, retaining
repeated operand edges.

The source-language and raised graphs use integer token IDs as comparison
identity. The actual C fixture is parsed into a token/dataflow graph, and an
atlas-style deduplicator assigns the same path token to the two identical
multiplication expressions. Strings remain diagnostics only. This mirrors
Nodus HSPIR/KPath's `TokenId` + edge-path model and leaves stable numbering to
the canonical/atlas vocabulary owners.

The test compiles `(x * y) + (x * y)` from C at `-O2`. GCC performs common
subexpression elimination and emits one `imul`, one `lea`, and `ret`. The
machine code lifts into SSA, emits back to C, recompiles, relifts, and agrees
behaviorally with the original shared library over signed test inputs.

The syntactic source graph and raised graph are intentionally non-isomorphic:
the former contains two multiplication nodes while the optimized raised graph
contains one reused multiplication. Their label-independent topology score is
approximately `0.66`, demonstrating similarity without demanding replication.
An explicit CSE quotient then contracts the two equal multiplication-token
nodes; the quotient is token-conditionally isomorphic to the machine-raised SSA
graph. Recompiled C is relifted and isomorphic to that raised token topology.

## Verification

```text
python -m pytest tests/test_machine_code_lifting_roundtrip.py \
  tests/test_llvm_repository_ssa.py tests/test_precompile_to_ssa.py -q
17 passed in 8.55s
```

## Next Steps

- Separate ISA decoding from platform ABI descriptions.
- Extend the reference vocabulary with move, compare, branch, and explicit
  flag semantics, retaining exact widths and signedness.
- Add memory SSA before accepting stack-bearing or pointer-bearing functions.
- Replace the hand-authored source topology fixture with a C frontend graph
  when a suitable existing frontend is selected.

## Prompt History

> theorize our ability to raise machine code from reference vocabularies to llvm, our ssa, fortran, or anything else we can process into our multi-graph ir

> set this as a goal and use a test where we round trip through c and can demonstrate a topological similarity (not exact replication)

> how curious, part of me thought they'd be simple topology or knot topology

> uhhh, fucking funny you say that because we're going to use tokens instead of strings and I expect to examine the similarities in language and fully compiled topology
