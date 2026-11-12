# Radio source and service continuation

The native system-output Radio table is in the isolated Nodus Free-runtime
branch. Its capture/fftfree and UDP rows run together after one headless step.
See `AGENTS/experience_reports/1794518848_DOC_Nodus_Radio_Free_Table.md`.

Remaining work from the owner's earlier Radio request:

- Add a native or sanctioned Python media-file playback source row and keep it
  in the same Radio table when selected.
- Expose linear as well as the existing logarithmic band mapping through the
  real fftfree analyzer contract.
- Register Radio as an AST/MCP service with a subscription operation that
  returns its UDP endpoint and packet contract; add recording later.
- Qualify host/device change recovery and longer-running latency and drop
  behavior before calling it a hard real-time service.
