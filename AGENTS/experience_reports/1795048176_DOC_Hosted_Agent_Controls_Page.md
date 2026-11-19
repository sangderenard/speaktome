# Hosted agent controls page

**Date:** 2026-11-18 local / 2026-11-19 UTC

## Work

The Turing Meowdoku source now includes an operator Agent controls page. It uses the hosted-agent mode store for a per-account enabled switch and current goal, and exposes the existing last-word settings. Llama's chat runner reads the controls every ten seconds, skips replies and queued work while disabled, and adds the current goal to its chat setting. The agent defaults to enabled with no goal. The source web export lists the new page and script.

The follow-up added `hosted_controls` and `hosted_controls_set` to SpeakToMe's REST catalog. The catalog generates MCP tools from those entries. The GET entry is kiosk-readable so Llama can read its own setting; the POST entry is operator/admin-only at the Turing handler. The site menu places Agent controls next to Agent workspace, gated by `operator_workspace`.

## Verification and limits

Focused hosted-agent mode and Llama runner tests passed: 13 tests with the service Python and SpeakToMe path. Python compilation, JavaScript syntax, and Git whitespace checks passed. The first test attempt with system Python lacked `speaktome_mcp` and failed collection. A broader server authorization run had 13 passes and one guest lifecycle assertion failure (`shader tier` instead of `localhost only`); its cause was not assigned. No public deployment or live process change was made. This goal is chat guidance; the separate Llama training parameters page remains a separate ticket.

The SpeakToMe pytest harness attempted automatic environment setup and skipped collection because its `.venv` was unavailable. Direct assertions with the service Python verified catalog route matching, kiosk GET access, and the generated MCP boolean field schema. No package was manually installed.

On 2026-11-19 the owner reported that disabling Agent controls did not stop Llama. The live controls row was off, but the hosted runner and model process had started on November 17, before the new runner code; the installed Apache service reset did not own the independent hosted Llama task. An intermediate wrapper-script change was withdrawn after the owner reiterated that Llama belongs to the stack lifecycle. Turing source now declares the task in `instances.json` and has Apache's piped-log supervisor start, maintain and end it with the stack. Task registration removes its independent boot/restart triggers and grants that supervisor's service account read+execute on the exact task. The runner stops its model while disabled or if site controls become unavailable, and uses the existing kill-on-close Windows job to contain its model child. Focused hosted runner and supervisor tests passed 29/29; PowerShell, Python and JSON parsing passed. The current shell received Access is denied when trying the task's normal stop command, so the live task has not been reprovisioned or reset.

The owner then asked specifically about `stm restart`. Its owner-local shortcut at `C:/Users/alber/stm.ps1` previously restarted only Apache, which would not reliably stop the already-running older Llama task on the first reset. `Invoke-Restart` now registers the on-demand task and its Apache permission, stops the task, verifies the model port closes (ending only the pinned llama-server executable if the pre-migration runner left it behind), restarts Apache, and starts the task. `stm full` calls the same function. The source change is parsed but an elevated live `stm restart` has not yet been executed from this shell.

## Prompt History

> can you make sure there's a page that lets you set the settings for the llama agent, being able to switch it on or off or set target goals for it, as a general case of agent controls - something like that should already exists but if it doesn't it needs to

> how does t his overlap or harmonize with the agent workplace

> did you put this anywhere in the menu or rest/mcp

> where in the menu

> this agent controls isn't doing anything to stop llama

> llama needs to go down and up with everything else when there's a server reset

> I told you it goes off and it goes on with the stack so what is this shit

> is it going to go down and back up when I use stm restart

> what do you mean new and old supervisor god damnit faggot just make it shut down and start up with the fucking stack
