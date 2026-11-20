# Guest tunnel SSE 429 and host receipt

## Observation — 2026-11-19 22:45 UTC

The SpeakToMe guest plugin's read-only `control_log` call returned `UNAVAILABLE`: `MCP SSE probe returned 429 from openai.org`. Error data named the internal `tunnel-service.gateway.unified-0.internal.api.openai.org` tunnel URL with HTTP 429 and `application/json`. The issue query did not execute through that connector.

Read the stable host's encrypted Apache access log via the repository's `RecordCipher`, printing only time, method, path without query string, and status. The last 2,500 entries covered 2026-11-19 16:01:34–16:45:10 -0600 and contained no `/mcp` or OAuth request. A second scan through 16:45:40 -0600 also found no `/mcp` request. This supports an upstream rejection for this guest probe, but the Apache log alone cannot prove that no other local listener received traffic.

The separate `mcp__speaktome` connection successfully read legacy tickets, issues and Research/Development cases. Apache recorded `GET /api/tickets` at 16:45:33 and `GET /api/issues` and `GET /api/research/cases` at 16:45:36, all HTTP 200. This proves the stable application and queue endpoints were serving requests at the same time. That connection has a different account and visibility than the guest plugin; it does not verify the trial tool catalog.

Earlier host-side session-quota 429s are distinct: the 2026-11-16 receipt records an MCP initialize reaching the host and receiving `guest session limit reached`. The later host design removed the per-IP hourly budget and uses live-session capacity with HTTP 503 when full. Neither receipt identifies the enforcing component for today's tunnel probe.

## Follow-up — 2026-11-19 23:37 UTC

The tunnel subsequently passed guest calls. `whoami` returned a signed-in trial with `view_site`; `api_index` succeeded and declared `research_cases` eligible, but `research_cases` returned `OAuth token expired` (401). This text comes from `BearerRest.sign_in`, which `PreviewRest` inherits when any internal REST request returns 401; it does not prove that an OAuth credential actually expired.

The Turing host creates a preview session and sends its token as `Authorization: Preview` on the loopback REST request. `_account` authenticates it, while the Research and issue route handlers passed only a browser cookie to their services. The trial MCP request has no such cookie. Turing `server.py` now passes the validated preview token to both existing services. The focused real HTTP/MCP test exercises `research_cases` and `reports_issues_list` through a guest session and passed. See Turing hazard `20261119T233701Z_main.md`; live activation remains open.

The static connector advertises `where`, but the live API index marks it and legacy tickets unavailable to trial because they require `dev_ui`. Their refusal is consistent with the account's capability set, separate from the Research 401.

## Follow-up — token precedence and Look file transfer

The owner clarified that cookie and token states must both be checked, with an explicit token determining identity. They also clarified that a Look user's cookie-backed state should be brought into the server file space allotted to that user and retained when the temporary account becomes a permanent member. The Turing source now checks the account cookie independently while allowing a valid token to supersede it; an invalid token does not inherit a valid cookie's identity. Research and issue services accept the already authenticated OAuth or preview token. Look scratch uses the existing account-owned store, imports an older scratch cookie on `/api/scratch`, and retains files across Keep because the account ID stays the same. The register page performs the import before Keep. The SpeakToMe catalog and README were updated to describe the new storage contract. Focused real HTTP/MCP and agent-workspace tests passed 14/14; live activation remains open.

## Next Steps

Use the timestamp and internal tunnel URL from the guest error to inspect OpenAI tunnel gateway diagnostics or support. Retest the guest plugin after that service accepts the SSE probe, then verify the preview account's authorized read-only catalog and issue views.

## Prompt History

> I retried the SpeakToMe development ticket query.
>
> Same error:
>
> UNAVAILABLE
> MCP SSE probe returned 429 from openai.org
>
> The connection is still failing during the MCP SSE probe, before the ticket query can execute.
>
> The reference to openai.org is significant, but it doesn’t conclusively identify which system is enforcing the limit.
>
> The next diagnostic step is to check whether your server receives these connection attempts at all. If it doesn’t, the rejection is occurring upstream of your application.
>
> I still can’t access the development queue.

> Yes. The SpeakToMe guest connector currently advertises 279 tools, including `where`. So the earlier `unknown tool where` error is a mismatch between the advertised tool catalog and what the server actually accepts.
>
> The tools most relevant to diagnosing this are `man`, `whoami`, `api_index`, `look`, `enter`, `where`, `research_cases`, and `reports_tickets_list`.
>
> I'll test the discovery and identity endpoints first to see what the live server actually recognizes.
>
> The connection is now working well enough to get real responses.
>
> `whoami` succeeded
>
> `api_index` succeeded
>
> `research_cases` failed: OAuth token expired (401)
>
> The server identifies my connection as:
>
> We've isolated a new problem: the tunnel is passing requests, but the OAuth token is expired. The server recognizes a trial account, yet authenticated access to the development queue fails.
>
> The next action is to refresh or reconnect the guest connector's OAuth authorization. Once that's corrected, `research_cases` is the appropriate endpoint for your Development cases—not the legacy ticket endpoint I originally tried.
>
> This is substantially more diagnostic information than we had before.

> if there's a token it should supercede cookies and always be checked in addition, there's always two questions, what's the cookie state, what's the token state, tekt hthe token state and allow the cookie to be imported

> into the client filespace

> taking in a look cookie is a decision for promoting a temporary account into permanent, the server file space allotted for a remote user would be where the user's cookie would go, all it's state, able to be transferred to a new member account
