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
