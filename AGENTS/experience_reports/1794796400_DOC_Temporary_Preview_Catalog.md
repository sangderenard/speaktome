# Temporary preview account catalog handoff

**Date:** 2026-11-15
**Title:** Browser preview and account retention routes

## Overview
The Turing host adds a seven-day read-only preview account backed by its existing account session cookie, with a random server-generated password that is never sent to the browser. A visitor may keep the account by choosing a username and unique password and accepting the alpha terms. This companion adds the two routes to the SpeakToMe REST catalog.

## Steps Taken
- Inspected the existing account, authz, public login, and catalog contracts.
- Registered `POST /api/account/preview` as guest-reachable and `POST /api/account/keep` as account-only. Both are browser-only, require the preview header, and are withheld from generated MCP tools because the keep request carries a password.
- Ran `python -m pytest --noconftest -q tests/test_speaktome_catalog.py -k "not method_path_group_summary_returns and not paths_occur_in_source"`.

## Observed Behaviour
11 selected catalog tests passed. The two excluded tests fail on existing unrelated review-summary length and source-path discrepancies.

## Lessons Learned
The host's guest gate depends on declared catalog entries, so the Turing and SpeakToMe changes must be reviewed and deployed together. The temporary browser credential is an existing HttpOnly account session cookie; the random account password stays server-side.

## Follow-up: trial MCP navigation, 2026-11-19
Ticket 4ca9a8022394490cbb0c7ff1d672e706 and Development case d18abe304859492596265aab942fe24d exposed a second catalog gap: the trial filter hid read-only site maps, comments, Research and issue views even for a signed-in preview with view_site. The Turing host's trial gate rejects other POST requests with a read-only explanation, while its GET routes still check each resource. Added a finite allowlist of selected GET tools; existing preview participation exceptions and permanent-account eligibility remain as they were. The guest MCP connector returned an external SSE 429 during investigation, so live trial verification remains open. A foyer HTTP 503 means its resident host is unavailable and is separate from catalog eligibility.

Focused catalog tests for trial eligibility, general eligibility and guest reachability passed. The full catalog test file still has pre-existing failures for missing GROUPS entries and stale kiosk expectations; Claude is addressing the GROUPS failure on review 8f3be40d173040e89eb22ad2af6df8ae.

## Follow-up: hosted map pages, 2026-11-20
The trial connector recovered and `research_cases` now returns case summaries. The tool catalog also offers `site_map_root`, but its `/site/manual/` GET was refused before the host with `only the host API may be called`. `PreviewRest` and `BearerRest` admitted only `/api/` paths, while the existing host map route is deliberately `/site/<map>/`. Added a narrow read-only allowance for the same declared map-page shape that the host accepts, with no write or general static-path allowance. Focused tests cover both hosted REST adapters, valid root/place reads and rejected writes, traversal and unrelated paths (12 passed). Live verification needs the reviewed connector code activated; no host or service was changed.

## Next Steps
Have an independent reviewer inspect the hosted map-path allowance. After landing and connector reload, confirm the trial account can read `/site/manual/` through MCP while a private review thread and non-preview participation POSTs remain blocked. Report the foyer host status separately.

## Prompt History
> we should explicitly warn not to use a password used elsewhere, and provide a cookie based random password with an expiration that can be opted out of to keep the account and get a change password notice, to look around before wanting to sign up for anything

> 7 days, read-only (Recommended)

> use the mcp for checking things out do not run git
