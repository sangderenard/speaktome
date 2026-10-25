# Turing ProcessGraph to SymPy control algebra

**Date:** 2026-08-01
**Title:** Solver-oriented ProcessGraph equations for control flow and BitOps

## Scope

Implemented a reverse ProcessGraph-to-SymPy route in the independent `turing`
repository. Git index, branch, and history operations were deliberately
avoided because other agents were working concurrently.

## Work completed

- Added an equation-preserving `SymbolicProcessModel` alongside the existing
  compact expression projection.
- Encoded `select`, `Phi`, and Turing `mu` merges as the exact polynomial mux
  `false + condition * (true - false)` with Boolean-domain constraints.
- Encoded scalar NOT, AND, NAND, OR, and XOR as multilinear polynomials.
- Added little-endian bit recombination and bounded simultaneous transition
  unrolling for loop/state equations.
- Kept unknown operations explicitly uninterpreted and reportable instead of
  assigning them invented mathematical semantics.
- Replaced the deferred design note with the implemented model, its solver
  usage, and exactness boundaries for vector lanes, effects, and dynamic
  recurrences.
- Added the public, inspectable `SYMPY_PROCESS_GRAPH_TRANSLATIONS` reverse
  table. Its rules recover canonical arithmetic, comparisons, math calls,
  logical operations, indexing roles, and nested three-input `Select` nodes
  from SymPy `Piecewise`; undefined functions become explicit `Call` nodes
  carrying their callee name.
- Added strict/fallback auditing to SymPy ingestion and exact re-projection
  checks after reconstruction. This caught and fixed a condition bug where a
  relational SymPy predicate was incorrectly compared with numeric zero.
- Added a compact AST-ingestion/precompile/simplify/rebuild/precompile length
  comparison and a stress version using the real `encode_jfif_resident`
  pixel-to-JFIF source hierarchy. Neither test uses the fused-program route.

## Verification

The active system Python had the required project packages; the repository's
`.venv` was stale and referenced a removed Python 3.10 interpreter. Without
installing dependencies, focused verification passed:

- nine focused non-stress symbolic projection/relation tests (excluding one known unrelated
  Mandelbrot dispatch-region regression);
- the focused branch-to-Phi reducer test;
- the BitOps translator test;
- Python bytecode compilation and `git diff --check`.

The corrected AST-ingested JPEG stress test built a 5,760-node source module
and a 108-node reduced encoder. The first precompile emitted 109 instructions
and added one graph node. The compact SymPy expression remained exactly 132
operations with a 5,778-character `srepr` after the aggressive pass. The full
program form contained 107 equations and 65 uninterpreted operations; its
per-equation aggressive pass changed zero equations. Reconstruction mapped all
109 modeled nodes with zero translation fallbacks and restored every ordering
edge, producing 143 nodes before and after the second topology reducer. Final
precompile emitted 145 instructions and left 145 graph nodes. The passing run
took 50.47 seconds and
had no test-level timeout.

This supersedes the earlier 109-to-48 observation. That smaller graph rebuilt
only the compact selected output and omitted effect/control equations, so it
was not a valid whole-program comparison. The full result demonstrates that
JFIF really crosses both SymPy representations and comes back through the
reducer/precompiler; it also demonstrates that the current SymPy strategies do
not optimize this encoder and that explicit mathematical lowering makes it
larger.

The homepage's separate count of 647 uninterpreted operations was audited as
322 `minimum`, 322 `maximum`, and 3 `tanh` operations in a 3,096-node
Mandelbrot/color model, not the JFIF graph. None had been removed before
SymPy; they were explicit equations classified as opaque because backend
spellings were absent from the semantic registry. Registered
`minimum`/`maximum` aliases and `tanh`, plus JFIF lowercase constants and
indexed access. Focused tests now require those known mathematical operations
to produce no uninterpreted entries.

The compact AST test reduced direct precompile length from 16 instructions to
4, ProcessGraph size from 15 nodes to 4, and SymPy operation count from 11 to
1; the final expression was `3*right`.

The full symbolic file still exposes the existing Mandelbrot filtered-region
failure: its extracted boundary input is named `value_24` and its index is a
constant zero, while the test supplies `floatframes` and expects runtime
`floatframe_index`. The same original projection already produces
`value_24[0]`; this was not caused by the new control algebra and was left out
of scope.

## Exactness boundary

The relational model is complete for a live slice when it reports no
uninterpreted operations and retained loops/effects have explicit transition
relations. Multi-quantum BitOps nodes must expose scalar lanes before the
projector may honestly claim a polynomial model for every bit.

## Prompt History

> you're working concurrently so be very careful about git usage. I want you to work on how we can go from programmatic concepts in a process graph back to sympy by using clever tricks, some might be described in some file, bitops maybe, where you can technically fully describe programmatic control flow with pure math that we can use for the symbolic solving capability of sympy

> if you didn't make one already I'd be interested in a test that precompiles a demo function and then moves it back to process graph to sympy, performs aggressive simplification attempts, then goes back to process graph, and then precompile, and we compare the lengths of the two versions

> go big, try to do it on the code that we have that goes from pixel data to jpeg data, see if it locks up like it does with the fully fused mandlebrot

> I don't think you can use fused program for that, that's not what that's for

> you have to use the ast ingestion

> don't use a timeout it does finish just let it

> oh yeah sometimes you really have to wait for those

> oh yeah we made the things to go to sympy now we need to make the things to come back from it

> some translation table or something

> did you understand the task? why aren't you reporting what they were when they got converted back from sympy all wthe way back down to reduction, and why aren't you doing sympy on the jiff encode, or am I not understanding.  this is probably great but not quite there yet

> the ui said there were 647 explicit uninterpreted operations, and that makes me wonder does that mean they were resolved out as unimportant before the sympy? why are we holding anything in an "uninterpreted" mode
