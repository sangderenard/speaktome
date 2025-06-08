# Ring Network Communication Protocol

This document formalizes the workflow hinted at in various messages regarding a
"full ring network pass." The protocol establishes a simple review loop for
messages and documents while attaching agent identity information in JSON form.

## 1. Overview

Agents share updates by placing Markdown files in `AGENTS/messages/outbox/`. A
message circulates through the network of participating agents until each has
acknowledged it. The process resembles a ring where the message travels from one
agent to the next.

## 2. Identity Affixing

Each agent maintains a JSON profile in `AGENTS/users/` and is listed in
`AGENTS/agent_registry.json`. When an agent forwards or modifies a message, they
should append a short JSON block to the bottom of that message using the
following format:

```json
{
  "agent": "AgentName",
  "date": "EPOCH",
  "action": "forwarded"  // or "acknowledged", "edited", etc.
}
```

This affixed snippet acts as a lightweight signature and can be parsed by future
scripts.

## 3. Completing the Ring

A message is considered fully processed after every agent listed in
`agent_registry.json` has appended an entry. At that point the file may be moved
from `outbox` to a permanent location or archived.

## 4. Benefits

* **Transparency** – Everyone can trace who has seen or modified a message.
* **Discoverability** – JSON snippets make it trivial to script audits of message
  flow.
* **Consistency** – Using shared identity files keeps author names consistent
  across commits and messages.

Agents are encouraged to experiment and refine this protocol. The goal is simple:
keep communication visible and attributable as our network of collaborators
expands.
