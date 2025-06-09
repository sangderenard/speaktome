# Abstract Linear Layer Activation

**Date/Version:** 1749474298 v1

## Overview
Expand the prototype linear network to support an optional activation function and provide a convenience constructor for multiple layers.

## Prompts
- User: "expand on the complexity supported by the abstract linear layer and code the wrapper for multiple abstract layers becoming a model"

## Steps Taken
1. Added `activation` parameter to `AbstractLinearLayer` with ReLU implemented via `clamp`.
2. Implemented `SequentialLinearModel.from_weights` helper.
3. Extended tests to cover the new functionality.
4. Ran the full pytest suite (failures due to missing dependencies).

## Observed Behaviour
- New tests pass under the default backend.
- Several unrelated tests fail when the suite imports optional dependencies.

## Lessons Learned
Backend-agnostic networks can include simple non-linearities without touching each tensor backend by leveraging existing operations.

## Next Steps
Consider additional activations once supporting operations exist across backends.
