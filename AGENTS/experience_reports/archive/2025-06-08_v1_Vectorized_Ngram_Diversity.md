# Template User Experience Report

**Date/Version:** 2025-06-08 v1
**Title:** Vectorized N-gram Diversity

## Overview
Profiled the repository using cProfile and identified `ngram_diversity_score` in `scorer.py` as a loop-heavy hotspot. Implemented a vectorized version to reduce Python-level iteration.

## Prompts
- "draw job and perform task"
- "Identify computational hotspots and convert serial loops into parallel implementations where safe."

## Steps Taken
1. Installed optional dependencies `numpy` and `torch`.
2. Profiled `cpu_demo` and `scorer.ngram_diversity_score` using `cProfile`.
3. Replaced inner loops with tensor operations and added a new test `test_ngram_diversity_matches_slow`.

## Observed Behaviour
The updated function matches the original looped implementation. Profiling indicates fewer Python function calls for the same workload.

## Lessons Learned
Vectorizing hot loops with PyTorch `unfold` and `unique` can simplify code and improve performance without altering external behaviour.

## Next Steps
Consider further vectorization of other scoring utilities and explore full batching across beams.

## Aftermath
During review, another contributor proposed an alternative parallelised approach
to the same function. The team voted to adopt their implementation instead, so
this patch was reverted. The regression test and this report remain as evidence
of the experiment.
