# Incoming OAuth MCP review handoff

**Date:** 2026-11-16 UTC
**Title:** First-party account OAuth and protected HTTP MCP source

## Scope

Coordinated branches: `speaktome:codex/incoming-oauth-mcp-20261115` and
`turing:codex/incoming-oauth-host-20261115`. SpeakToMe adds the bearer REST
adapter and protocol version support. Turing adds the account-backed OAuth
authorization server, token storage and the `/mcp` HTTP route. The two source
revisions must be deployed together; no live service was changed here.

## Evidence

An isolated loopback host test exercised client registration, explicit login
and consent, authorization code, PKCE refusal and acceptance, single-use code,
token challenge, MCP initialize, account-bound whoami, and Origin rejection.
It passed with `SPEAKTOME_REPO` pointing to the matching checkout. The stdio
suite exposed one stale assertion: the catalog already declares
`X-Meowdoku-Entry: mcp`, so the exact-request expectation was updated.
The public HTTPS endpoint was not probed or deployed.

## Next Steps

Review the paired branches, verify reverse-proxy routing and the configured
public origin, then run the OAuth flow through the public HTTPS endpoint before
marking it available to clients.

## Prompt History

> Urgent need:
>
> Yes. **Remote OAuth access is feasible**, and my earlier answer needs a distinction: SpeakToMe already uses OAuth to connect *out* to services such as Suno. It does **not yet** offer OAuth for clients connecting *in* to SpeakToMe’s MCP server. The current incoming server is local stdio.
>
> The path is to expose a **Streamable HTTP** MCP endpoint, for example `https://stichka.duckdns.org:8766/mcp`, and make it an OAuth protected resource. An authorization server would sign users in and issue tokens; the MCP endpoint would validate each token, map it to a SpeakToMe account, and apply that account’s existing permissions. MCP clients would discover the authorization server through published metadata.
>
> Use the speaktome mcp you have access to to checkout branches and submit them for review when ready

Clarification: the owner selected “SpeakToMe authorization server”.
