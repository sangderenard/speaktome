# Research MCP catalog handoff

**Date:** 2026-11-15
**Title:** Research and Development endpoints in the generated MCP catalog

## Overview
The Turing Research/Development review `bad965f3c5cc49df9cae9f3f20c13d79` received a blocking comment that its REST routes had no SpeakToMe catalog entries. This companion change registers the case, document, invariant, brief, promotion, vote, affirmation, and survey endpoints so signed-in agents with `view_site` can discover and call them through the generated MCP.

## Steps Taken
- Read the catalog and generated MCP tests through the SpeakToMe checkout API.
- Added the Research group and eleven route entries with the endpoint header and declared request fields.
- Added an MCP routing test for an explicit text document with a JSON null source reference.
- Ran `python -m pytest --noconftest -q tests/test_speaktome_mcp_api.py -k "research_document or legacy_field_schemas or every_offered_endpoint"`.

## Observed Behaviour
The targeted tests passed (3 passed). A full MCP test run found an existing expectation that omits the automatically added `X-Meowdoku-Entry: mcp` header. The catalog test suite also has existing failures for a review summary length and paths absent from the checked-out Turing source. These are outside this catalog handoff.

## Lessons Learned
The generated MCP exposes only cataloged REST routes. A required `source_ref` can still be JSON null when the source kind is text, so the tool schema needs an inline nullable type.

## Next Steps
Review this companion with the Turing Research/Development change. See `todo/1794795549_Research_MCP_Catalog_Review.stub.md`.

## Prompt History
> use the mcp for checking things out do not run git

> The new REST handlers lack entries in `speaktome_api/catalog.py`, so they will not become generated MCP tools for agents.
