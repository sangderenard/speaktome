# Agent Identity Files

This directory stores JSON documents describing individual agent identities. Each file
must follow the `YYYY-MM-DD_AgentName.json` naming pattern so entries can be sorted
chronologically.

The structure mirrors the proposal documented in
`AGENTS/messages/outbox/2025-06-07_Proposal_AgentProfileJSON.md` and typically
includes fields like `name`, `date_of_identity`, `nature_of_identity`, and optional
details.

Agents should create a file here when they join the project or adopt a new role.
