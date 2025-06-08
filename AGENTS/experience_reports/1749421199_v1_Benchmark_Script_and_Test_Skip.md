# Template User Experience Report

**Date/Version:** 1749421199 v1
**Title:** Benchmark Script and Test Skip

## Overview
Added a micro benchmark helper and ensured tests skip when torch is missing.

## Prompts
- "could you add if we need to add it the imports for measuring process time for benchmarking"

## Steps Taken
1. Created `testing/benchmark_tensor_ops.py` using `time.process_time`.
2. Updated `testing/AGENTS.md` to mention the new benchmark script.
3. Added a `numpy` extras group in `pyproject.toml`.
4. Modified `tests/test_ngram_diversity.py` to skip when PyTorch isn't installed.

## Observed Behaviour
- Benchmark script prints timing results for each available backend.
- `pytest` no longer errors on missing torch; test is skipped.

## Lessons Learned
Optional dependencies must be guarded in tests, and small utilities help gauge backend performance.

## Next Steps
Expand benchmarks with more operations and sizes.
