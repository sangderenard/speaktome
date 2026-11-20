# MCP catalog recovery after service outage

## Observation

On 2026-11-20, the MCP process still reported a packaged, unchecked API catalog after the site recovered from the overnight 503. `ApiServer.build()` kept the first catalog for the life of the process. The service had recovered, but a later `tools/list` did not expose newly declared routes.

## Change

The existing generated MCP server now retries only while it holds a fallback catalog. `tools/list` and `api_index` adopt a recovered service index and rebuild the tool table from the original verbs. Rebuilding also removes tools that an unchecked fallback offered when the recovered account lacks permission. Generated calls still use the catalog's fixed request routing and the REST server still enforces permissions.

## Verification

`python -m pytest --noconftest -q tests/test_speaktome_mcp_api.py`: 27 passed. `--noconftest` was needed because this branch checkout lacks the repository environment marker, and `tests/conftest.py` automatically attempts environment setup. The test file imports the checkout source directly. This was focused verification of the generated MCP contract, not a live client reconnect test.

## Next Steps

Submit this exact checkout head for review. Once it lands and the MCP process reloads, verify that a fallback during an outage is replaced on a later `tools/list` or `api_index`. Clients which cache their tool list may still need to request the list again or reconnect.

## Prompt History

- User: “When did the 503 start”
- User workspace guidance: “Before starting nontrivial work anywhere in this tree, check [`speaktome/AGENTS/experience_reports/`](speaktome/AGENTS/experience_reports/) for prior agents' reports.”
- Scheduled work instruction: “After those checks, use the remaining run to advance the highest-impact actionable ticket or feature from the prior run.”
