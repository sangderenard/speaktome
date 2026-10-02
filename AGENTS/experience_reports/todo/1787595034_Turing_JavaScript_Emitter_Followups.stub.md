# TODO Stub

**Date:** 1787595034
**Title:** Turing JavaScript SSA emitter follow-ups

See `AGENTS/experience_reports/1787595034_DOC_Turing_Class_First_JavaScript_SSA_Emitter.md`.

- [x] Accept richer `ClassSchema` facts when present and prove that its SSA
      projection agrees before emitting inheritance, defaults, static methods,
      types, or fuller signatures.
- [ ] Emit the planned inheritance, defaults, static methods, and types where
      each destination can express them faithfully.
- [ ] Attach WGSL and Wasm/assembly deployment regions to the dependency-free
      JavaScript host while keeping classes and resident state host-owned.
- [ ] Define exact i64/BigInt and JavaScript-number precision policies before
      advertising those vocabularies as complete.
- [ ] Extend Python-oracle parity beyond the completed elementwise catalogue:
      reductions and tensor construction/fill first, then shape/index work.
- [ ] Register the JavaScript destination in target selection and bundle
      publishing after those surfaces can accept a complete `IRModule`.
