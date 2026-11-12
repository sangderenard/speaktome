# Existing divider request contract: source mode and unresolved runtime binding

The original ray-host upload is planned but not compiled. Its wasm request
is outside the current Nodus row driver's c/llvm/fortran backends, and its
real host callable signatures are outside the bridge's all-AbstractTensor
column adapter. Existing GenericObject and typed byte/pointer FIFOs must be
reused, not replaced or mistaken for missing transport.

An explicitly authorized source-only mode now exists in isolated Turing
`codex/divider-unstable-permission-20261111` and Nodus `fix/unstable-source-only`
worktrees. It reuses Site intake/ownership, the existing queue, native Nodus
FIFO host, divider and sanctioned lowerer. It does not adopt or create a
generic executable binding. Explicit inference admission is separately
unchecked under the same default-off unstable permission.

Real original-source trials reached SSA lowering. The lexical partition now
retains the real `dict[str,int]` closure declaration and its column contracts
along the actual capture edge. Turing checkpoint4010f365 reaches final
concordance: string glyph-span queries conflict with scalar string-token map
keys. Compatible dynamic key storage, equality and insertion remain unresolved.
The attempted SSA shares arena storage by reference, but there is no working
native alias of the original Python dictionary in this source-only route.
Preserve actual declared evidence. Do not infer runtime ABI from numeric columns, invent call
values/JSON encodings for Python objects, change the immutable live upload, or
bypass the Site compiler ownership path. Native closure publication and the
automatic GenericObject body_reference binder remain separate unimplemented
boundaries. No GPU link or rendering is authorized by this note.

Evidence and current tests:
`AGENTS/experience_reports/1794440502_AUDIT_Divider_Request_Boundary.md`.
