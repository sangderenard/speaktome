# Torch Soft Failure and Test Skip Logic

## Overview
Applied repository instructions to ensure failing Torch installations do not abort setup and updated tests to skip gracefully when Torch is unavailable.

## Prompt History
- "no attempt to install torch is in the logs please run the script again so I can see exactly what torch was attempted where and what the result was. Do you hide logs or did you not do it?"
- "log in and follow guidance to set up environment"
- "go into both setup_env.ps1 and setup_env.sh and wrap the torch section in soft failure because we don't actually care if torch gets installed, then make pytest tests aware of whether torch exists or not to blacklist tests requiring it entirely without any limited mode soft failure"

## Steps Taken
1. Reviewed setup scripts and confirmed Torch install commands already use safe failure.
2. Added graceful version check printing so missing Torch does not cause tracebacks.
3. Introduced a `requires_torch` pytest marker and collection hook to skip marked tests when Torch is absent.
4. Marked existing Torch-dependent tests with the new marker.
5. Recorded this report and validated guestbook.

## Observed Behaviour
- Setup scripts now display "Torch = missing" when Torch is not installed instead of a Python error.
- PyTest skips marked tests with a clear faculty skip message when running without Torch.

## Next Steps
Continue refining faculty-aware skips for other optional dependencies.
