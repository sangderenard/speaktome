# Operator Policy Clarification

**Date/Version:** 1749447504 v1
**Title:** Clarify operator routing and remove extraneous helpers

## Prompt History
```
are you serious? The request was to implement the single operator dispatch function defined in the abstract class, not random helper names. Please refer to the abstract for the correct signature.
```

## Summary
- Updated `speaktome/tensors/AGENTS.md` to state that `_apply_operator` is the sole operator primitive backends must implement.
- Removed obsolete scalar helpers from the JAX backend.
- Replaced their uses in code and tests with direct calls to `_apply_operator`.

