# Generic optional Hermes preparation

The local Hermes runner already accepts `--account` for arbitrary identities. Design an optional per-account preparation attachment using the existing agent workspace profile and memory store, then expose the same self-owned context to inbound MCP clients. Keep the runtime boundary explicit: an external inbound ChatGPT client cannot execute inside the local Hermes CLI process. No fixed GPT account or automatic privilege grant.
