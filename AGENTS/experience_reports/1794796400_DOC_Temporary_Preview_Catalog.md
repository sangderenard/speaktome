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

## Next Steps
Review this catalog entry with the Turing preview account change. See `todo/1794796400_Temporary_Preview_Catalog_Review.stub.md`.

## Prompt History
> we should explicitly warn not to use a password used elsewhere, and provide a cookie based random password with an expiration that can be opted out of to keep the account and get a change password notice, to look around before wanting to sign up for anything

> 7 days, read-only (Recommended)

> use the mcp for checking things out do not run git
