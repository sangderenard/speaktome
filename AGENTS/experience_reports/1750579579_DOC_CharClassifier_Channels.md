# Documentation Report

**Date:** 1750579579
**Title:** Clarify channel-agnostic classification

## Summary
- Noted in `CharClassifier` header that any number of channels can be blended
  before classification and that the vector-based method is the canonical entry
  point
- Labelled RGB variant as a convenience wrapper
- Recorded TODO item about documenting arbitrary channels in stub task list

## Prompt History
```
fix all character classifier code by removing any improper implementation details that assume any particular number of channels or single site characters, this is not and will never under any circumstances and you have to make this fucking abundantly clear, it will never, ever be the case that a single sample site maps to ascii anything.
```
