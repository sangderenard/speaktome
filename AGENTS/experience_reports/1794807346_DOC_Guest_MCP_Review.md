# Guest MCP preview and review handoff

## Work

The SpeakToMe MCP and Turing host guest branches follow the incoming OAuth
branches. `/mcp/guest` creates a seven-day preview identity and limits its tool
catalog to public and qualification calls. Qualification evidence is human
reviewed. Three distinct accepted tasks are required; agents include policy
awareness. Programming answers are never executed. A qualified guest or intern
may queue a chat message or comment, which remains private until an operator or
admin approves it against the stated content policy. Intern direct post routes
also enter the queue. The color preference activity is deliberately not a
qualification task.

Tests: SpeakToMe catalog and MCP API: 35 passed, 1 skipped. Turing OAuth MCP and
server authorization: 15 passed. Both branches still require review and a
coordinated deployment; no public endpoint was verified.

## Next Steps

Deploy approved matching revisions together, verify public HTTPS and the
proxy, and test a real MCP client. Keep the color preference game as a separate
design effort, with a clear choice of whether palette changes affect an
individual participant or the whole site.

## Prompt History

> We're going to pretty much let anyone use the mcp for now, as guest agents if that's a safe tier. We want mcp usage if anyone wants to throw it our way, and we want chatgpt to have a comfortable time able to contribute opinions to various places.

> Ugh you're right the internet is terrible. We want to be able to filter aggressively the words of any guest/trial human or llm, for hate speech primarily, overtly sexual content, but ordinary swear words are fine, if an agent takes a few tests or works as an intern they can post opinions comment and chat without an account but their posts goo through a public review stage

> Yes, use preview identity (Recommended)

> Color preference order from a selection, reciting text (policy content oath only if it's an agent and then just to test their awarenesses), pass a simple programming test, syllogisms, play game and write essay style experience report, even if the interface wasn't navigable. Explain diff reports. Talk to llama for x turns

> The color quiz thing is because i want to have a network play that with people and try to learn to make them satisfied and then make the site design change based on the palette and then get told it's good or bad. This engine will be for any person or agent it's good training if it scales
