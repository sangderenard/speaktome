# Backend Primitives Correction

**Date/Version:** 1749447152 v1
**Title:** Restore JAX primitive implementations

## Prompt History
```
your agents file misses the reality and the point, these primitives DO need to be implemented IN the ONE function DEDICATED for it in the abstract. It is not the case that you do not define arithmetic helpers. you don't define pointless ones that miss the design that's clear in the abstract class, that explains exactly what function I'm talking about
```

## Summary
- Updated `tensors/AGENTS.md` to clarify backend operator policy.
- Implemented missing arithmetic primitives in `jax_backend.py`.

## Next Steps
Ensure all backends remain aligned with `AbstractTensorOperations` interface.
