# Turing Native SSA C Backend Mid-Flight Audit Repairs

**Date/Version:** 2026-08-31 / working tree
**Title:** Native SSA C parity repairs while vehicle compilation work was active

## Overview

Audited the new `src/compiler/ssa_c_backend.py` in the middle of another
agent's vehicle-program iteration, comparing its contracts with the native
LLVM and Fortran SSA lanes. Applied the outstanding safe repairs without
discarding or resetting the heavily dirty shared worktree.

## Prompts

> in turing, in the compiler, there are several backends to drop to. C is new and we have been using llvm and fortran through c but now we have a c backend. It's not performing exactly perfectly and it's unclear moment by moment what the next problem will be. an agent is currently working on passing a program, iterating on the c backend and compiling in general to tackle the problem. I want you to do an audit right in the middle of their work now on c backend shortfalls compared to llvm and fortran

> proceed to make all listed changes that have not already been carried out

> I'm not certain you're supposed to use pow don't we have a reduction identity for that

> float pow is different though we can't identity that

## Steps Taken

1. Re-read the live backend and focused tests rather than relying on the
   earlier audit snapshot; the concurrent worker had already repaired the
   shaped-formal aggregate case.
2. Repaired native storage and ABI generation: conservative shaped spans,
   invocation-local scratch, no unsound `restrict`, runtime extent vectors,
   and row-major multi-axis address calculation.
3. Added exact-width integer C types and LLVM-compatible signed/unsigned
   casts, comparisons, shifts, and wrapping arithmetic.
4. Kept generic floating `Pow` as C `pow()`. Constant-exponent identities
   remain owned by the shared `reduce_constant_exponent_pow` pass rather than
   by the native module emitter.
5. Added a canonical `emit_ssa_to_c` entry point, migrated module consumers,
   published scalar/tensor capability inventories, and changed the machine
   inventory from the legacy captured-tape C lane to the native SSA C lane.
6. Added compiled runtime regressions for dynamic 2-D indexing, extent
   propagation, aggregate storage, integer overflow, extension/truncation,
   unsigned comparison, alias qualifiers, scratch lifetime, and capabilities.

## Observed Behaviour

- The focused C/backend suite passed: 38 tests in the final combined gate,
  plus the repaired inventory test independently.
- An earlier broader backend gate passed 44 tests. Python syntax compilation
  and `git diff --check` passed for the touched implementation/test surface.
- The repository `stack_double` helper closure emitted without C shortfalls
  and compiled as a native library.
- A canonical balloon-tire build remains blocked before C emission by active
  upstream SSA lowering/shape errors (missing ranks and incompatible tensor
  shapes). This was not attributed to the C backend or modified here.
- Truly dynamic compiler-owned temporaries still fail explicitly with a C
  emission shortfall because no workspace-size/ownership ABI exists. Dynamic
  public/feed spans and their extents are supported; the backend no longer
  guesses or silently allocates one element.

## Lessons Learned

The largest parity gaps were ABI gaps, not operator spellings. A backend can
appear complete on scalar helpers while corrupting a linked program through
undersized spans, alias promises, static scratch, or missing runtime shape
data. Capability reporting also has to name the actual SSA backend; retaining
the legacy captured-tape C inventory made preflight selection misleading.

Floating power needs the semantic split the user identified: approved
constant-exponent rewrites belong to the shared identity policy, while a
surviving variable floating exponent is not reducible and must remain `pow()`.

## Next Steps

## Canonical balloon-tire takeover update

Later the same day, the canonical balloon-tire lowering reached complete C
emission and exposed four concrete native defects. They were repaired at their
shared compiler boundaries rather than patched into the authored vehicle
program:

1. Shaped semantic Boolean tensors were registered as one-byte storage even
   though repository comparison, broadcast, and where kernels use double-backed
   0.0/1.0 buffers. Tensor lowering now stamps the physical dtype before tensor
   descriptor registration, and registration sizes from that physical dtype.
2. A scalar `sum_double` return carried a stale 8,192-element call-site view.
   C publication copied 8,192 doubles from one stack scalar. Publication count
   now follows the physical Ret address: an emitter-local `&tN` is one scalar.
3. `balloon_tire_gas` wrote its `(8,4)` `out13` result (32 doubles) into a
   caller allocation selected from a stale `(8,1)` projection (8 doubles).
   C activation allocation now takes the maximum of its local tensor descriptor
   and the backend-neutral interprocedural storage requirement.
4. `IndexedStore` was documented and aliased as a new SSA version of one
   resident arena, but tensor lowering used functional `index_set_double`,
   leaving every update in discarded temporaries. `IndexedStore` now lowers to
   the existing in-place `index_assign_double`; functional `index_set` remains
   copy-producing.

The build tool and standalone C compiler now accept an explicit optimization
level. The verified debugging command was:

```text
python tools/build_balloon_tire_native.py --output build/balloon-tire-c-resident-o0 --backend c --optimization O0 --batch-size 8 --frames 1
```

The clean compiler-generated executable completed one and two frames with exit
code zero. After two frames, all 226,372 public values were finite, all 24,576
state cells changed, 431 output cells changed, and no generated source was
manually patched. Focused regressions for physical Boolean storage,
interprocedural projection capacity, scalar publication extent, resident
IndexedStore semantics, and the declared vehicle indexing path passed 5/5.

Optimized compilation was deliberately not attempted after the user requested
that correctness be established at `O0` first.

## Next Steps

The canonical C program now runs and updates finite state at `O0`. Before
enabling optimization, compare its material outputs with the reference oracle
under the same initialized state and tolerance contract. Truly dynamic
compiler-owned temporaries still require a sized caller-owned workspace ABI;
the backend should continue rejecting those rather than guessing capacity.
