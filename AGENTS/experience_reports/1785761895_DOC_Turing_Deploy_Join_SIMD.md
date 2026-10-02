# Turing Deploy/Join and Host-Adaptive SIMD

**Date:** 2026-08-03
**Title:** First scale-1 deployment consumer in the Turing compiler

## Activity

The adjacent `turing` repository gained a backend-neutral deployment frame
shared by Control IR and SSA. Structural `Deploy` and `Join` handlers now mark
lexical parallel regions; reduction joins refer to established operators such
as `Add` rather than creating numerical SSA operations.

An LLVM consumer lowers a legal scale-1 `float64` Add reduction into a native
SIMD cohort with horizontal join and scalar tail. Floating regrouping requires
explicit reassociation permission. Unsupported frames are rejected so the
retained serial CFG remains the fallback.

Host selection was verified on the local `bdver2` CPU. Its preferred width is
128 bits (two `float64` lanes); AVX2 is absent and is not required or emitted.

## Verification

Focused Control/SSA/LLVM suites: 51 passed. Broader graph, AST/SymPy,
operator-catalog, fusion, GLSL, Control, and loop-composer suites: 131 passed.

## Prompt History

> "is ther eany way we can run agoal to set this up and then hav eone backend we outfit with simd reduction as scale 1 deployment, breaking into apropriate lane widths"

> "make sure it tests, we don't have AVX2 on this system"
