# Backend returns wrapper

## Overview
Adjusted backend conversion helpers to return `AbstractTensor` objects instead of raw primitives. Updated `to_backend` to handle either wrapped or unwrapped results.

## Prompts
- "In all your backends you are returning data not tensors holding data."
- "Check methods and methodology between abstract tensors and the different backends ... make sure that an abstract tensor behaves and returns in all ways like a pytorch tensor."

## Steps Taken
1. Modified `to_backend` to detect and accept wrapped tensors.
2. Updated `from_numpy`, `from_torch`, `from_pure`, and `from_jax` in each backend to create a new wrapper.
3. Attempted `bash setup_env_dev.sh --prefetch --headless` but script failed due to missing modules (`AGENTS`, `tomli`).

## Observed Behaviour
The code compiles, but environment setup remains incomplete.

## Next Steps
Investigate environment bootstrap failures so tests can run.
