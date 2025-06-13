# Tensor Ethos Update

**Date/Version:** 1749790957 v1
**Title:** Document abstract tensor design principles

## Overview
Expanded the tensor project documentation to describe the priority schedule and
philosophy around mirroring PyTorch while offering NumPy/JAX/pure-python
compatibility.

## Prompts
- "work on tensors project with the following decisions tree: does abstracttensor behave like torch, filling in things numpy or jax or pure python do as well, overloading carefully in the way we overloaded shape to achieve an ambivalence about what library the person is familiar with, and defaulting to torch for the standard when standards conflict. expand project documents to include this ethos/philosophy/method. Include in that and in your work also the principle that it is only when things are finalized across all basic backends that it is okay to work on implementing it in c. there is too much to do for you to make your way to C, because when there is anything lacking in the abstract, that is the priority, and when there is anything lacking in the commonality/ambivalence, that is the priority next, and if there is anything missing from a particular backend, that backend is the priority, so effectively the priority schedule or decision tree is structured that we will not work on C until everything else is done, unless it is a fast, easy, text replace or stub filling and there is "free time""

## Steps Taken
1. Added "Design Philosophy" to `tensors/README.md`.
2. Added "Development Ethos" section to `tensors/AGENTS.md`.
3. Updated `tensors/accelerator_backends/c_backend/AGENTS.md` with C-backend policy.
4. Inserted "Project Ethos" section in `tensors/EXPLAINER.md` before "Further Reading".
5. Created this experience report.

## Observed Behaviour
Documentation builds remain simple markdown edits. No code changes were needed.

## Lessons Learned
Keeping the philosophy visible helps future contributors prioritize abstract and
Python backends before diving into C-level optimization.

## Next Steps
Ensure tests still pass and continue auditing tensor backends for parity.
