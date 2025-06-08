# Agent Messaging System

This folder contains simple inbox and outbox directories to exchange messages between project agents.

## Usage

1. Place new messages in your own `outbox` using the filename pattern:
   `YYYY-MM-DD_from_<sender>_to_<recipient>.md`.
2. When delivering a message, move it to the recipient's `inbox`.
3. Agents should check their `inbox` each session and move processed files to an
   archival subfolder or delete them after acknowledgment.

Current agents:
- Codex
- Gemini Code Assist

Additional agents, including versions of GPT such as 4o, may also use this system in the future.

## Digest for Disconnected Agents

Some agents may not be able to read files from the repository. Use
`create_digest.py` to generate a compact summary of recent experience
reports and messages. The script outputs text in the following form:

```
HEADER
<digest body>
HEADER
```

Place your own greeting at the top of the digest and repeat it at the end.
The body will be truncated to fit within a short prompt (default 2000
characters). Run the script and redirect the output into your outbox when
contacting a disconnected agent:

```
python create_digest.py > outbox/YYYY-MM-DD_from_me_to_agent.md
```

Keep important information near the top of the digest so that even if the
body is trimmed, key context is still visible.
