# Guest MCP tunnel and session quota

## Work

The OpenAI Secure MCP Tunnel now connects the local SpeakToMe guest endpoint to the owner's ChatGPT workspace. ChatGPT created and connected a personal “SpeakToMe guest” plugin through the tunnel. Its first read-only chat request reached the MCP host but failed during initialize with HTTP 429: the five-new-session-per-IP hourly quota had been exhausted by connector discovery and validation. The Turing host review branch raises this cap to thirty per IP per hour; it retains the seven-day preview and public review rules. The matching documentation branch records the new limit.

The focused Turing OAuth/MCP test passed with the sibling SpeakToMe repository supplied through SPEAKTOME_REPO. It covers the thirtieth session succeeding and the thirty-first returning 429. The first test collection without SPEAKTOME_REPO failed because the sibling REST package was unavailable, then passed with the documented dependency.

## Next Steps

Have the host change reviewed and landed, restart the stable service through the existing control mechanism, then repeat a read-only ChatGPT guest tool call. Confirm the tunnel runtime's readiness after a machine restart; the native runtime manager currently keeps the client process healthy during this session.

## Prompt History

> We're going to pretty much let anyone use the mcp for now, as guest agents if that's a safe tier.

> Set up the tunnel

> Create restricted persistent key
