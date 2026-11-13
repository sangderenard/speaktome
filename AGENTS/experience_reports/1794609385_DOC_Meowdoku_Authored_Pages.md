# Meowdoku authored pages

**Date:** 2026-11-13 UTC

## Overview

Turing PR #1221 changes the Meowdoku root into a feed of account-authored
pages. Existing games receive direct entry pages. This report records the
cross-project boundary and the source-only result.

## Steps Taken

- Read workspace and Meowdoku instructions and prior reports.
- Traced Meowdoku's shared Accounts database, stable/development route gate,
  encrypted user-file store, account workspaces, and Turing site bundles.
- Added dev/release publication with owner preview, stable direct-read gating,
  explicit readable upload copies, and a trusted completed-bundle handoff.
- Ran focused publication, route, access, and workspace tests in an isolated
  Turing worktree; committed dc27d77d and opened PR #1221.

## Observed Behaviour

The existing Nodus SiteCompiler emits row tools, not Turing site page bundles.
The compiler handoff therefore registers a completed Turing bundle only after
it is present in an owned, measured account workspace. An encrypted private
upload cannot be made public by the server; the owner's browser must decrypt
and submit a separate readable copy. Stable reads hide dev pages, including
direct page and bundle asset URLs. Changing a registered bundle blocks release.

Focused publication and route tests passed (14); the final publication unit
suite passed (5). This does not establish a live browser or public deployment.
The isolated worktree did not have the Spectral checkout required by the full
web exporter, so only the game-page export helper ran.

## Lessons Learned

The shared Accounts DB is the publication authority across both Meowdoku
instances. A feed entry and every direct asset read need the same stage check.
The browser must explicitly create a public copy of any private file.

## Next Steps

Add a self-service job using the existing source divider, Turing site-bundle
builder, and workspace ledger. Coordinate the managed web export/restart and
verify the stable and development hosts in a browser before claiming release.

## Prompt History

> "when you say legacy do you mean like the main page meowdoku? because we need to dissolve that into individual pages, and track the pages by author like an instagram feed"

> "Only user-published pages"

> "Both"

> "Yes, make a separate public copy (Recommended)"

> "then they can mark them dev or release, and that determines if they show up in stable server at all"

> "Give page authors dev preview access (Recommended)"

> "can  you continue the work"
