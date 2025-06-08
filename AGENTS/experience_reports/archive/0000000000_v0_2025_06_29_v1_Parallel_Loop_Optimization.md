# Parallel Loop Optimization

**Date/Version:** 2025-06-29 v1
**Title:** Parallel Loop Optimization

## Overview
Tasked with drawing a job and executing it, I received guidance to profile and parallelize loops. I focused on optimizing a scoring function within the codebase.

## Prompts
```
System: You are ChatGPT, a large language model trained by OpenAI.
Developer: always check the files in the repo ecosystem for your benefit...
User: draw job and perform task
```

## Steps Taken
1. Ran `python -m AGENTS.tools.dispense_job` which produced `parallelize_loops_job.md`.
2. Inspected loops in `speaktome/core/scorer.py` and identified the `ngram_diversity_score` function as a candidate for vectorization.
3. Implemented a vectorized approach using tensor operations to remove Python loops.
4. Executed `pytest -q` to ensure the test suite still passes.

## Observed Behaviour
All tests passed successfully, confirming the refactor did not break existing functionality. The new implementation relies solely on tensor operations and avoids Python-level iteration.

## Lessons Learned
Some scoring utilities still contained serial loops. Using hashing and `torch.unique` allows per-beam n-gram counting entirely on the GPU/CPU tensor backend.

## Next Steps
Evaluate other scoring functions and tree operations for similar vectorization opportunities. Investigate performance differences once optional dependencies like Torch become available.
