# Look preview inbound requests

## Work

A seven-day look preview may discover `guest_inbound_status` and `guest_inbound_request` through MCP. Its first request waits 180 seconds without holding a server worker. One request is accepted per preview and no more than 12 requests globally in a rolling day. The request is a private lead in Operator control, never an identity grant. The admin still selects an exact pending OAuth connection, identity, scope, and expiry. The site warns that a look request is untrusted and does not prove which OAuth client sent it.

## Verification

The Turing guest request and OAuth integration tests passed (2); the admin control suite passed in the preceding focused run (4). The paired SpeakToMe catalog/MCP focused tests passed (2). Operator control JavaScript syntax passed. No deployment, restart, or live OAuth pairing was performed.

## Next Steps

After both reviews land, the owner publishes and restarts, then a real external look client should follow the tool to request access and separately start OAuth. Observe queue limits and adjust only with evidence of normal demand or abuse. A look request cannot be cryptographically identified as the later OAuth client; admin must verify the exact connection separately.

## Prompt History

> "can you give look accounts a delay and attempt request as inbound path, like, something that isn't fast enough to be abuseable but lets an llm connect see the system and follow a tool to establish it's request to be an inbound agent"

> "use the mcp for checking things out do not run git"

> "you are not allowed to use any script that goes around my back to stop and start services"
