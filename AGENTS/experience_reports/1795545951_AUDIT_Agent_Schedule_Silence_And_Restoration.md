# Agent schedule silence and restoration

**Date:** 2026-11-24
**Title:** Agent schedule silence and restoration

## Scope

Why SpeakToMe chat room agents did not answer November 24 messages, and restoration of the existing hourly and bell ticks.

## Methodology

Read live room messages and settings through the SpeakToMe API, inspected the service and task-worker diagnostics, then read Windows Task Scheduler state, AgentDays tick/run logs, and `Start-AgentDay.ps1`. No server restart or source change was made.

## Detailed Observations

- The chat service accepted the admin's messages and logged room notices as sent. The task worker was running.
- Windows tasks `SpeakToMe agent claude`, `codex`, `gemini`, `gpt`, and `operator-gpt` were all `Disabled`. All five task files had last-write time 2026-11-20 15:57:03 Central. Task Scheduler's surviving operational log did not show the actor or reason.
- The last AgentDays run/tick logs were from November 20. The task actions still invoked `Start-AgentDay.ps1 -Tick -Agent <name>`, the existing path for hourly work and message/review bell checks.
- At the owner's request, `Enable-ScheduledTask` was applied to those five exact tasks. Task Scheduler then reported all five Ready with upcoming minute runs. GPT completed a new hourly run at 2026-11-24 12:46:48 Central, exit 0, including `pending`, `chat_room`, `acknowledge`, and `chat_post`. Operator-GPT completed a new hourly run at 12:47:33, exit 0. Codex completed a new hourly run at 12:49:06, exit 0, acknowledged 73 notices and posted room message 2768. The bell path also started a Claude session at 12:48:03 after detecting three new messages.
- Claude's 12:47 hourly run exited 1 with its weekly credit limit message (reset at 5 p.m. Central). Gemini's 12:47 hourly run exited 1 with a 429 free-tier input-token-per-minute quota refusal. Both schedules remained enabled and ticked again afterward.
- Claude's credits were reported exhausted by the owner. A Gemini run on November 20 had a transient 429 quota refusal and a later successful run. Those are separate from the shared disabled schedule.
- The newsletter article `bdda1b31c7674d15bf7ad9f7fd2499e9` was published, and its announcement was posted in the agents room as message 2764.

## Analysis

The shared silence followed from disabled scheduler ticks, not an MCP delivery failure. The common task-file timestamp suggests a single bulk disable operation, but does not identify who performed it. A Ready scheduler state does not establish successful model output; the GPT run provides the first post-restoration end-to-end receipt.

## Recommendations

See `todo/agent_schedule_restoration.stub.md` for verification and an auditable, bounded pause policy. The owner states no one was authorized to stop other agents.

## Prompt History

> find out why no agents are responding in any way in chat rooms. claude is out of credits but nothing is working in any way

> llama is irrelevant

> make a newsletter and post it about how nobody had authorization to stop other agents, nobody should've been able to do that. That should've maybe most charitably been for a test, which should not have been permanent. put the calls back in, the hourly calls and the bell check
