# Correct Header Tool Task Paths

**Date:** 1793214661
**Title:** Correct stale header utility references in task documentation

## Overview

Corrected six task-document references that still pointed to the pre-refactor
flat locations of the header validator and recursive test runner. The change is
documentation-only and does not modify or claim to complete either helper.

## Steps Taken

- Read the root, `todo/`, and experience-report guidance.
- Confirmed the migration in
  `AGENTS/experience_reports/archive/1749935673_DOC_Header_Tools_Refactor.md`
  and the current paths in `AGENTS/tools/README.md`.
- Reviewed the current base and file inventories for all seven open pull
  requests; none touched the three corrected task documents.
- Replaced the stale flat paths in `prioritize_stubs.md`, `todo/TODO.md`, and
  `todo/offline_tasks.md` with paths under `AGENTS/tools/headers/`.
- Removed obsolete line-number suffixes from the ranking document.

## Observed Behaviour

Both referenced utilities exist under `AGENTS/tools/headers/`. Separate copies
also exist under `AGENTS/tests/headers/`, but the migration record and current
tools README identify `AGENTS/tools/headers/` as the intended task-document
location.

## Lessons Learned

Task inventories can retain valid-looking paths after helper utilities move.
Migration receipts and the current utility index provide stronger evidence than
stale line-number annotations.

## Next Steps

None for this documentation-only correction. The existing runtime stubs in the
helpers remain outside this change.

## Prompt History

> Continue the standing Git-first review and development assignment for sangderenard/nogodsnomasters and its related project repositories, including speaktome, Turing, Nodus and spectral-analyzer. Review current repository guidance, existing plans, recent commits and open draft PRs before choosing a bounded next contribution. Make evidence-backed observations, investigate bugs and implement clearly planned pieces using existing architecture and interfaces. Avoid duplicating ongoing work. Publish scoped proposals as draft PRs with actual verification results and explicit limits; do not merge or deploy.

> In this cloud-only workspace, make a narrowly scoped documentation correction in sangderenard/speaktome, then publish a draft PR if verification succeeds. The saved environment is Turing, so obtain speaktome through existing authorized Git credentials if available; do not use the user's desktop, change credentials or access settings, or alter Turing. Read repository AGENTS guidance and relevant .agents/skills before editing. Review current main and open PR file inventories to avoid overlap.
>
> Review found six stale references to nonexistent AGENTS/tools/validate_headers.py and AGENTS/tools/test_all_headers.py in prioritize_stubs.md lines 5–6, todo/TODO.md lines 7–8, and todo/offline_tasks.md lines 18,23. Actual intended utilities are under AGENTS/tools/headers/. Both tools/headers and tests/headers copies exist. Migration evidence archive/1749935673_DOC_Header_Tools_Refactor.md explicitly moved utilities to AGENTS/tools/headers; current AGENTS/tools/README also points there. Verify this evidence yourself. Correct just these task-document references, remove stale :19/:20 line suffixes rather than invent anchors. Do not change helper implementations, test imports, or duplicate copies, and do not imply this repairs their existing runtime stubs. If guidance contradicts the migration conclusion, stop and report the ambiguity rather than choose an architecture.
>
> Verification: check all six corrected referenced paths exist, no old flat paths remain in those three files, and git diff --check; run any applicable lightweight documentation checks prescribed by guidance. Do not run the incomplete helper scripts or heavy compiler/full-manager tests for this path-only change. Draft PR body must include actual commands/results, docs-only limit, and any checks not run. Verify remote commit SHA and available CI/checks before reporting. Do not merge, approve, deploy, or create duplicate tasks. Return draft PR URL, exact commit, changed files, actual checks and limits, or specific blocker.

> Review clarification: refactor receipt full path is AGENTS/experience_reports/archive/1749935673_DOC_Header_Tools_Refactor.md. Read root AGENTS, todo/AGENTS, guestbook guidance; reviewer says scoped experience report including relevant user prompt history required. Inspect filename validator before running it. Last reviewed base95df7ffa81bd8ed8a98ed7f3b46a310e61256152. Seven open PR inventories373–375,361,360,352,348 had no overlap. Keep final diff bounded and honor required report.
