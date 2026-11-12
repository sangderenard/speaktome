# Development service loans

See `AGENTS/experience_reports/1794509459_DOC_Development_Service_Loans_And_Scheduler_Throughput.md`.

Reconcile the existing SQL workspace lease and branch-review ownership semantics, then implement one account-owned checkout that borrows one bounded Development service through the supervisor and Nodus contract port. Keep custom branch source/build/state separate from Stable. Add stable-headroom load proof before allowing a Stable loan. The Nodus FREE process implementation is a separate scheduler task; restore the fastest scheduled path first.
