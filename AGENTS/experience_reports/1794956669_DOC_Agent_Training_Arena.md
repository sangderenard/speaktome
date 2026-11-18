# Agent training arena handoff

**Date:** 2026-11-17 UTC

## Overview

The Turing Meowdoku site now has source changes for an operator agent training
arena. A menu page queues a shared ticket-to-Research or Research-to-Development
task for two through 64 teams. Teams use one or two held-open Codex/Claude
process instances launched under the existing nonoperator accounts from
PowerShell. A separate agent scores the prompt, first message, and target
response against five axes. Completed receipts are linked from matching
Research and Development cases.

## Steps Taken

- Read the workspace map, SpeakToMe guestbook guidance, Turing instructions and
  obstinance ledger, Meowdoku instructions, prior agent workspace report, and
  existing round-table pipe and agent-run contracts.
- Added the arena queue, source-authorized listing, page, site-menu entry,
  web-export entry, agent pipe adapter, and operator coordinator in the Turing
  repository. The web server never starts model processes.
- Added focused tests for queue ownership, claim tokens, validation, held-open
  state, and operator route access.

## Observed Behaviour

The focused Turing run with the service Python interpreter passed 29 tests
across the arena, round-table seat, and agent workspace suites. JavaScript
syntax, Python compilation, PowerShell parsing, and Git whitespace checks
passed. This is source verification, not a live account launch, site export,
service restart, or public browser verification.
An additional Studio export check failed because the pre-existing
`marble_camera_movements.js` import is absent from `web_export.py`'s declared
exports; the arena does not import or change that module.

## Lessons Learned

The existing round-table `Start-AsAgent`/stdio path supplies the actual
nonoperator process boundary. Llama's batch queue is a different interface
without a held-open system-prompt override, so the page offers Codex and Claude
until that boundary is built. Numeric evaluator scores and reported tests are
retained with their evidence; neither is automatically a verified code test.

## Next Steps

Review the Turing source changes, run the operator coordinator against a
disposable ticket/case with the actual nonoperator accounts, then export and
deploy the site through the existing controlled service lifecycle. Add a
matching held-open Llama override interface before offering Llama seats.

## Prompt History

> I want you to design a page in the menu that lets you place one or two agents in both of two teams. each team has two tasks. write the system prompt for the other agent. try to make them respond helpfully to a given goal for the conversation. your other task is to, after setting those overrides, say something to the other agent, in order to achieve a goal, usually a subtask. in any message or system prompt, INSTRUCTIONS in the text shall be replaced by this body of text. Agents that can be stateful or not should default to stateful, but the toggle should be on the page. with two members on a team, one is the prompt crafter, the other is the communicator.
>
> The page should be able to host any number of such teams, all oriented to one goal, and they should be scored by objective metrics, something identical as possible to real goal oriented training, such that it can be used with agents that are set to be in a training mode

> Design and implement a menu page for a multi-team agent training arena. Each team has 1 or 2 agents, with prompt-crafter and communicator roles when two; prompt overrides and first message, stateful default with toggle, arbitrary number of teams sharing one goal, objective scoring and training-mode support. User specifies that literal token INSTRUCTIONS in any message or system prompt must be replaced by the entire user-provided specification body. Inspect the relevant project/repository and integrate the page, with tests if possible.

> codex and claude, the non operators, launch as processes in powershell, they use a pipe. in this example the pipe would be held open, such is how  they would have instances and those instances could be stateful or not, llama instances was only the start

> turning tickets into research, or turning research into development, but not with the dev tools - using real tickets and real research to produce test results that are scored and correlated with the runtime parameters. achieving the goal should be a single runnable task where we can assign any agent to score each piece across axes of understandability, actionability, completeness, experience demonstrated, and code elegance

> This body of data, accessible by category of task, will be available in the research and the development, when references line up, as what you might think of as undergrad projects on the solution that can be skimmed for accidental brilliance
