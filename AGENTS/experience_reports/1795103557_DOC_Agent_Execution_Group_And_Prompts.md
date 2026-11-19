# Agent execution group and response prompts

**Date:** 2026-11-19

## Overview

Server-owned agents except Llama receive a separate `execution` group. SpeakToMe publishes `exec`, `python`, and `exec_result` REST/MCP tools under its `agent_execute` capability. Admins can remove or grant the group on the People page. Commands run through the existing AgentDay owner task using each agent's isolated Windows account.

## Steps Taken

- Added the three catalog entries in SpeakToMe, with the same capability and paths implemented in Turing.
- Replaced seeded and runner brevity instructions. Updated eight live agent workspace documents and the installed operator agent brief after verifying snapshots of both SQLite databases.
- Checked the catalog directly with the service Python. SpeakToMe pytest could not collect because its automatic environment setup failed with permission denied on `.venv/Scripts/python.exe`.

## Observed Behaviour

The direct catalog check found all three entries as tools requiring `agent_execute`. Turing focused authorization, queue, hosted runner, and roundtable tests passed before the final Codex native-tool switch; that switch is covered by its command construction test. The site has not been restarted or exercised as an isolated account yet.

## Lessons Learned

The installed external agent declaration keeps a copy of its brief apart from agent workspace documents. Updating only the workspace document would allow the old wording to return during profile installation.

## Next Steps

Complete the code review and merge both repositories, publish the site, then run a controlled live check of the group, queue, and native CLI restrictions.

## Prompt History

> I want server owned agents except llama to have access to exec and python as a separate distinct permission

> both

> aditionally, I need all instructions wiped of any encouragement to be brief, any restriction of responses, it's killing the ability to get any work done

> Yes, isolated agent account (Recommended)

> just make it a group llama isn't in
