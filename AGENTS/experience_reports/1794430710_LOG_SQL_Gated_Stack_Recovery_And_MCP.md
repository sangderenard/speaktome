# SQL-gated stack recovery and MCP connection

Date: 2026-11-11 UTC. The user corrected the operator path after the
SpeakToMe MCP's Claude tool discovery timed out. The REST sign-in to
`https://stichka.duckdns.org:8766` timed out, and no listener was present on
443 or 8766. The existing supervisor SQL status still labelled several old
instances `running`, but every recorded PID checked was absent. Port 8767 was
an independent local listener and was left alone.

Read `turing/examples/meowdoku/AGENTS.md`, `HOW_TO_START.md`, and the deploy
operator guide. Used the existing authenticated deploy script with the live
state root: `Status.ps1 -User codex -StateDir C:\dev\Powershell\srv\state`,
then `Start-Apache.ps1 -User codex -StateDir C:\dev\Powershell\srv\state
-WhatIf`, then the same start script without `-WhatIf` and with
`-InstanceTimeout 90`. The SQL-backed `script-auth` gate returned `OK codex`.
No process was killed, no direct SQL row was written, no firewall/router/TLS
setting was changed, and no site assets were exported.

The start script found `httpd -t` syntax OK and started Apache. Its supervisor
reported `meowdoku-a`, `meowdoku-dev`, `nodus-messenger`, `relay`,
`task-worker`, and `udp-router` running. The script's own localhost curl used
`-k`, so it was not used as a TLS trust claim. Follow-up verified the public
hostname with the repo's normal TLS-validating REST client: `whoami` signed
in as `claude` on `meowdoku-a`. The already connected Codex MCP `whoami`
signed in as `codex`; `claude mcp get speaktome` reported Connected. No
Claude model conversation or credits were used. This is a connection/start
receipt, not a full service or application test.

## Prompt History

User: "I have exited it, if you needed anything to do that didn't get done?"

User: "I closed claude and don't have any credits with them right now"

User: "you use a script through the sql to start it"
