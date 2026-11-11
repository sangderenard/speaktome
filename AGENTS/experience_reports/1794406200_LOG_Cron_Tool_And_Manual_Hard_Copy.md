# LOG: a cron tool in Nodus, and the manual's hard copy kept by it

**Date:** 1794406200 (2026-11-11)
**Title:** Scheduled tasks as contracts; missed runs reasoned about; `cron(5)` in the manual

## Scope
Here: the manual page `speaktome_man/pages/cron.5.txt` and a paragraph in `speaktome.7`. The work itself is in nodus
(`include/cron_schedule.h`, `include/cron_tool.h`, the headless ABI, `tools/run_canvas.py --cron`,
`tools/cron_host.py`) and in turing (`examples/meowdoku/scheduled_tasks.py`, supervisor and server wiring, deploy
scripts, the menu link).

## Environment
Windows, system Python 3.11, Visual Studio 17 2022 for the nodus test build tree. Speaktome tests:
`python -m pytest --noconftest tests/test_speaktome_man.py tests/test_speaktome_mcp.py tests/test_speaktome_site.py
tests/test_speaktome_routes_show_site.py -q`.

## Commands Run
- Speaktome tests above: 133 passed.
- nodus: `cron_schedule_test` and `cron_tool_test` (C++), `tests/test_cron_tool.py` on a real canvas host: all pass.
- turing: `tests/test_meowdoku_scheduled_tasks.py` 4 passed, including the real round trip (the cron tool on a
  temporary canvas asks, this server's host bakes the manual).

## Log Summary
**The tool.** A cron tool in Nodus is an engine on a contract prefix, the mirror of the compile host. When a task is
due it writes a request file and offers a contract; whoever is a host on the prefix performs the task and answers
with a status. It uses the existing negotiator and rows; it starts no thread and performs nothing itself.

**Missed runs.** The tool keeps a ledger: handled-up-to, last started, last finished, and an `alive` stamp. Every
occurrence it did not attend to is "missed", whatever the cause (the process was down, an earlier run held the tool,
no host took the work). Each task chooses `once` (one run stands for the gap, the default), `skip`, or `all` (each,
the most recent N), with an optional age limit. A run started and never finished is run again once. A new task
starts from when it is first seen. Each decision is written with its reason to a log and into the request.

**The hard copy.** The site's task `manual.hardcopy` bakes the manual's map into static pages at `/manual/` and
stamps the folder with the map's hash. The check writes nothing when the hash is unchanged and replaces the folder
when it differs. The default schedule runs the check hourly. The server also checks once at every start.

**Not live.** The live Nodus library predates the tool and the live canvas does not declare the `cron` port. One stop
window does both (turing `deploy/Enable-CronTool.ps1`). Until then a restart is safe: the canvas host reports the
cron tool as refused and starts everything else. The owner has not yet been asked to approve that window in this
entry's turn; the earlier lesson stands: ask before a live restart.

Other limits: time is the system clock (the privileged clock port does not exist yet; the tool takes its clock as a
parameter); the table is read at start only; the console does not list cron tools; a task with no host waits forever.

## Prompt History
Owner: "okay the site can publish a hard copy of the manual with a scheduled refresh check, which brings up the point
that we should make a cron tool in nodus that can perform server scheduled tasks and reason about ones that would've
been missed through some gap"

Owner: "yes on the hint" (the offered Help link in the site menu).
