1. Add `schema_version` and a single `connect()` recording it in accounts.db paths.
2. Add `--state-dir` (default unchanged) so every store derives from it, not `web_dir.parent`; test on port 8767 only.
3. Cut over to `C:/dev/Powershell/srv/state` via Stop-Apache/Start-Apache after approval; move secrets and logs with it.
4. Make `instances.json` args state-relative so leased worktrees share one instance definition.
5. Nodus trust grants, then the `wss://<host>/ws/contract/<prefix>` bridge.
