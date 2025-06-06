# 2025-06-06 v12 Digest Script and Docs

## Overview
Implemented a script to generate short digests of recent experience reports and
messages. Updated messaging guidelines and AGENTS overview to mention the new
tool and collaborative ethos.

## Prompts
"create in the agents filesystem for messaging a script that creates a single
prompt limited-access agent digest of all recent experience reports and messages
across the multiagent system and development process..."

## Steps Taken
1. Added `create_digest.py` under `AGENTS/messages`.
2. Documented usage in `AGENTS/messages/AGENTS.md`.
3. Expanded `AGENTS/AGENTS.md` with a note about the autonomous ecosystem.
4. Ran tests and validated the guestbook.

## Observed Behaviour
All tests pass. `validate_guestbook.py` reports no issues.

## Lessons Learned
Simple tooling helps share context with agents that can't read the repo.

## Next Steps
Use the digest script to summarize history when reaching out to GPT-4o or other
agents lacking repository access.
