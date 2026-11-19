# Hosted agent controls page

**Date:** 2026-11-18 local / 2026-11-19 UTC

## Work

The Turing Meowdoku source now includes an operator Agent controls page. It uses the hosted-agent mode store for a per-account enabled switch and current goal, and exposes the existing last-word settings. Llama's chat runner reads the controls every ten seconds, skips replies and queued work while disabled, and adds the current goal to its chat setting. The agent defaults to enabled with no goal. The source web export lists the new page and script.

The follow-up added `hosted_controls` and `hosted_controls_set` to SpeakToMe's REST catalog. The catalog generates MCP tools from those entries. The GET entry is kiosk-readable so Llama can read its own setting; the POST entry is operator/admin-only at the Turing handler. The site menu places Agent controls next to Agent workspace, gated by `operator_workspace`.

## Verification and limits

Focused hosted-agent mode and Llama runner tests passed: 13 tests with the service Python and SpeakToMe path. Python compilation, JavaScript syntax, and Git whitespace checks passed. The first test attempt with system Python lacked `speaktome_mcp` and failed collection. A broader server authorization run had 13 passes and one guest lifecycle assertion failure (`shader tier` instead of `localhost only`); its cause was not assigned. No public deployment or live process change was made. This goal is chat guidance; the separate Llama training parameters page remains a separate ticket.

The SpeakToMe pytest harness attempted automatic environment setup and skipped collection because its `.venv` was unavailable. Direct assertions with the service Python verified catalog route matching, kiosk GET access, and the generated MCP boolean field schema. No package was manually installed.

## Prompt History

> can you make sure there's a page that lets you set the settings for the llama agent, being able to switch it on or off or set target goals for it, as a general case of agent controls - something like that should already exists but if it doesn't it needs to

> how does t his overlap or harmonize with the agent workplace

> did you put this anywhere in the menu or rest/mcp

> where in the menu
