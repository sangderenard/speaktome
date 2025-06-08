# Job Prioritization Addition

**Date/Version:** 2025-06-24 v2
**Title:** Zipf-based job dispenser implementation

## Overview
Implemented automatic ranking of job description files and a script to dispense a job using a Zipf distribution over the ranks.

## Prompts
```
create a prioritization metric for the job descriptions and then a script that will dispense a job not as a measure of the most important butt probabalistically chosen based on a zipf distribution of the rank order of the priority values which themselves should be a rank order. there is zero room for negotiation and any variance from this specific algorithm will be denied
```

## Steps Taken
1. Read repository guidelines under `AGENTS/`.
2. Added `prioritize_jobs.py` to compute ranking by word count.
3. Added `dispense_job.py` that samples according to Zipf weights.
4. Created new tests in `tests/test_job_prioritization.py`.
5. Ran `pytest -q` to verify all tests pass.

## Observed Behaviour
The scripts generate `job_priority.json` and select jobs deterministically when seeded. All tests succeeded.

## Lessons Learned
Using word count as a simple metric produces a clear rank order. Zipf sampling ensures less urgent tasks still surface occasionally.

## Next Steps
Future agents may refine the metric or expose the selection via CLI.
