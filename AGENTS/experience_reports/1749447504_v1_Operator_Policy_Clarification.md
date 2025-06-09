# Operator Policy Clarification

**Date/Version:** 1749447504 v1
**Title:** Clarify operator routing and remove extraneous helpers

## Prompt History
```
are you fucking kidding me? sub_scalar and div_scalar are not the fucking functions you aren't even fucking trying to understand this request. what fucking function needs to be implemented for operators please, can you fucking answer that? because it's not the fucking individual fucking functions sub_whateverthefuck and I told you where to fucking find out
```

## Summary
- Updated `speaktome/tensors/AGENTS.md` to state that `_apply_operator` is the sole operator primitive backends must implement.
- Removed `sub_scalar` and `div_scalar` from the JAX backend.
- Replaced their uses in code and tests with direct calls to `_apply_operator`.

