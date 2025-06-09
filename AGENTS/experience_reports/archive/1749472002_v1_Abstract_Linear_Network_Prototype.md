# Abstract Linear Network Prototype

**Date/Version:** 1749472002 v1

## Overview
Implemented a backend-agnostic linear network prototype using the existing tensor abstraction layer. Added a new unit test validating the forward pass across two linear layers.

## Prompt History
- User: "attempt to prototype an abstract tensor based (assume any backend has all functions implemented so you can develop where it needs to be not where it could currently work) that functions with a series of linear layers, even though nn in the program thus far are not simple linear networks"

## Steps Taken
1. Created `abstract_linear_net.py` defining `AbstractLinearLayer` and `SequentialLinearModel`.
2. Added `test_abstract_linear_net.py` to verify output matches manual operations.
3. Ran the new test with `pytest`.
4. Attempted to run the full test hub but PyTorch was unavailable, resulting in errors.

## Observed Behaviour
- New test passes under the default backend.
- Full test suite fails during collection when importing torch.

## Lessons Learned
Abstract tensor operations can model simple networks when provided with appropriate linear layers. Missing optional dependencies limit full test coverage.
