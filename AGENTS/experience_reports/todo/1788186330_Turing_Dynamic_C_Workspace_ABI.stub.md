# Turing dynamic C workspace ABI

Design a sized, caller-owned workspace contract for compiler-owned temporaries
whose element count is known only at runtime. Thread size/alignment metadata
through linked call frames and reuse the existing runtime extent vector.
Preserve the current explicit C emission shortfall until capacity and lifetime
can be proven; never fall back to a one-element buffer or process-global
scratch.

Source report:
`AGENTS/experience_reports/1788186330_DOC_Turing_C_Backend_Parity_Audit_Repairs.md`
