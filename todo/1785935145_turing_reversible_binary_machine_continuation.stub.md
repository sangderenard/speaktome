# Turing reversible binary-machine continuation

Resume from
`AGENTS/experience_reports/1785935145_DOC_Reversible_Binary_Machine_Continuation_Handoff.md`.

Immediate work:

1. Diagnose why real `cmd.exe` pipeline execution creates/closes its reversible
   pipe but does not resolve the registered `/c/work/hello-card.exe`, although
   the existing interactive command proof does.
2. Prove a registered child executor receives pipeline stdin and publishes
   stdout back through inherited virtual pipe handles.
3. Cold-reopen, reverse, and replay that successful pipeline tape.
4. Continue process lifecycle, console-screen, and exception/unwind capability
   families while preserving fail-closed behavior and the authenticated
   per-instruction journal.

Current anchors:

- ordinary command tape:
  `C:\dev\Powershell\turing\build\cmd-echo-pipe-tier-20260805.segmented-tape`
- pipeline/error-recovery tape:
  `C:\dev\Powershell\turing\build\cmd-pipe-card-v2-20260805.segmented-tape`
- broad relevant regression selection: 253 passing tests
- current system `cmd.exe`: 286 imports, 153 matched, 133 unsupported

Never add ambient host exec, filesystem, registry, pipe, or device fallback.
