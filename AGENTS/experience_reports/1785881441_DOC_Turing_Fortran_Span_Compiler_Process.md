# Turing Fortran span compiler process

**Date:** 1785881441
**Title:** Fortran span/index lowering and compiled forward/reverse sorting learner

## Overview

Corrected the native sorting learner architecture so it uses Turing's normal
captured Graph IR, reverse derivation, control scheduling, SSA lowering,
Fortran backend, and common C shell. Added backend support for preallocated
span initialization and generic indexed arena access instead of materializing
large Fortran literal constructors.

## Steps Taken

1. Preserved rank-zero metadata and canonicalized captured dtype names at the
   precompile-to-SSA boundary.
2. Implemented `Handler.Fill` whole-span assignments plus generic
   multidimensional `GetElementPtr`/`Load`/`Store` lowering in the Fortran
   backend.
3. Added result-directed branch promotion for Fortran `merge` expressions.
4. Captured an eight-wire sorting network, derived its reverse correction
   graph, and scheduled it with an IR-authored RGB graph renderer.
5. Changed common C-shell feedback from byte copying to rotation of two
   preallocated arena addresses.
6. Compiled and ran the resulting native executable through both the new entry
   point and the historical `native_affine_learner` compatibility command.

## Observed Behaviour

- The focused backend/host suite passed 27 tests.
- The forward/reverse/compiler regression suite passed 46 tests.
- The end-to-end CLI produced and ran
  `sorting_process_learning_window.exe` successfully.
- Generated image planes use scalar whole-array initialization and sparse
  indexed assignments. No training set or image is emitted as a Fortran list
  of literals, so gfortran's 65,535-element constructor ceiling is irrelevant.

## Lessons Learned

Fortran was not the source of the literal-volume problem. The compiler had
failed to lower its existing span-memory abstraction at this backend boundary.
Keeping tensor accommodation metadata intact also matters at scheduled region
boundaries: a one-element span is not interchangeable with a scalar ABI value.
Feedback is best expressed as an outer arena-address policy, leaving the
compiled numerical regions free of runtime-specific copying.

## Next Steps

None required for this correction.

## Prompt History

> "1. you should've implemented the span operator present in other backends if you wanted to fix that, not change how my code works, it works the way it works specifically to use only preallocated arenas. when tensor accomodation is set up like it is for some other backends, you won't compile with billions of literals you'll just get a zero init so you need to undo what you did and then fix the fortran backend to handle span and indexing"
>
> "2. use the compiler process not bespoke garbage"

> "that's my bad be sure to filter those artifacts through gitignore and untrack them or whatever we need to do when you finish your work, i didn't want logs or images"
