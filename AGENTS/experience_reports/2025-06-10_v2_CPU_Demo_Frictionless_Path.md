# Template User Experience Report

**Date/Version:** 2025-06-10 v2
**Title:** CPU Demo Frictionless Path

## Overview
Validate that a new user can install only core dependencies and still run `run.sh` successfully, seeing CPU demo output.

## Prompts
"Keep troubleshooting and adjusting new user entry path until it just works. until you just are able to pull, run, and see stub dummy output from our no-torch sandbox. I need you to run in cycles doing this and fixing minor errors and lazy download, lazy import staging. A frictionless entry path is vital."

## Steps Taken
1. `bash setup_env.sh`  # minimal install
2. `bash run.sh -s "hello" -m 1`

## Observed Behaviour
- `run.sh` reported missing PyTorch and Transformers, then executed the CPU demo.
- Unrecognized arguments were ignored with a helpful message.
- Output showed three dummy sequences with scores.

## Lessons Learned
The CPU demo should gracefully accept unknown options so users can run the same command line regardless of installed extras. Updating `cpu_demo.py` to use `parse_known_args` solved this.

## Next Steps
Monitor for any remaining setup pitfalls. Document minimal installation clearly in the README.
