# Remove Scalar Helpers and Dummy Tensor

**Date/Version:** 1749450615 v1

## Prompt History
```
remove any mention in the codebase anywhere in any file for "sub_scalar" and "div_scalar" those are forbidden, obsolete, and fucking irritating garbage. further, get fucking rid of dummy tensor, fuck that class, and fuck you for proposing it.
```

## Summary
- Deleted obsolete scalar helper methods from all backends and the abstract interface
- Stripped test suite of the `DummyTensor` wrapper and added direct dispatcher checks
- Sanitized documentation and prior reports to remove deprecated names
- Updated a training notebook variable to avoid the banned term

## Next Steps
Ensure future contributions rely solely on `_apply_operator` for arithmetic dispatch and keep tests minimal.
