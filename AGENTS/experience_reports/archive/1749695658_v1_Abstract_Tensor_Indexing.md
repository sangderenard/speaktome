# Abstract Tensor Indexing

**Date/Version:** 1749695658 v1
**Title:** Abstract tensors as indices

## Overview
Explore how abstract tensors behave when used as indices and update the code to unwrap indices properly.

## Prompts
- "Make sure the abstract tensors can use abstract tensors for indices, is there any special function for being reduced to indices? Check wrap and unwrap i think maybe we just aren't using those for the task"

## Steps Taken
1. Read AGENTS instructions and examined tensor backends.
2. Added unwrapping logic to indexing methods and backend operations.
3. Expanded tests to cover tensor-based indices.

## Observed Behaviour
Tests now show that indexing and assignment using abstract tensor indices work across backends.

## Lessons Learned
Unwrapping indices inside `__getitem__`, `__setitem__`, and backend helpers keeps the API consistent and avoids advanced indexing surprises.

## Next Steps
Continue auditing tensor utilities for other missing unwrapping operations.
