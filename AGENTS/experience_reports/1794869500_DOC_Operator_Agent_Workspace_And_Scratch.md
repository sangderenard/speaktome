# Operator agent workspace and scratch catalog

**Date:** 2026-11-16 UTC

## Overview

The matching Turing MCP checkout adds a shared operator/admin document tree,
scenario context IDs and account-owned scratch text. This SpeakToMe checkout
declares the REST routes as generated MCP tools and gives authorized operators
index-first orientation at initialize. The internal HTTPS MCP client retains
preview scratch cookies per session.

## Steps Taken

- Opened both branches through the server's MCP checkout API.
- Added workspace and scratch entries to `speaktome_api.catalog`, including
  fixed write headers and trial-only cookie scratch access.
- Added a per-session cookie jar to the verified loopback HTTPS REST client.
- Checked catalog import, tool schemas and request routing directly. The
  matching Turing OAuth MCP test passed with a guest scratch write and read.

## Observed Behaviour

The SpeakToMe pytest harness attempted its automatic editable environment
setup and failed before test collection. That attempt is not a test result for
the code. The direct catalog/MCP schema assertions passed. Turing's focused
authz, server, workspace and OAuth MCP tests passed 40/40 with this checkout
supplied through `SPEAKTOME_REPO`.

## Lessons Learned

The generated MCP API gets its tools from the catalog. A trial's tool list
also has an explicit allowlist, so scratch endpoints need entries there as
well as `kiosk=True` and the new capability.

## Next Steps

Submit both commits for review, land both repository changes, export the
Turing page and restart the stable instance. Verify browser and live MCP
behavior on the public `:8766` route after landing.

## Prompt History

> I need some critical work done. I want you to use the mcp branch checkout and review process, but this is also going to be about mcp and file quotas and the fundamental agent experience. I want all operators to have a certain number of prompts seen only by operators and admin - think of it like each llm is a character in an adventure and they get storage and crafting tools, guidance reference, and access to a trouble tree. So we need an interface with enough separate areas to hold separate documents where any can be made full screen in the middle

> Personality would be orchestrated by making an arbitrary tree of increasingly specific categories so it only ever gets investigated add deeply as the moment calls for. System prompts should be indexes and orientation. All operators should be able to edit all other operators documents. Scenarios will be carried out by lists of increasingly specific contexts, they will correlate to a list of document ids in the order they should be consumed. I need to provide in mcp and rest scratch user space for any user who isn't temporary, temporary scooby's will get a cookie filesystem

> Use existing quota conventions
