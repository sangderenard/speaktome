# Canonical glyph-token interner continuation

- Expand Turing `C10: Strings as values` into the existing cross-system
  architecture: Turing identity/provenance, Nodus KPath token paths, and
  Spectral Analyzer glyph manufacture.
- Specify and version the canonical reversible mapping from Unicode scalar
  token sequences to one aggregate text token.
- Keep semantic text identity separate from shaping/layout identity and
  physical render-asset identity.
- Reconcile Turing's FNV whole-word token and minted string-glyph vocabulary
  without introducing a parallel interner.
- Canonicalize Nodus atlas nodes/edges so collision-checked token-path dedup can
  actually reuse equal sequences, then define persistence/cross-process rules.
- Add a token-reference boundary to Spectral Analyzer's existing
  `ExtrudedTokenAsset`/`RenderAssetCatalog` system while retaining reversible
  external text metadata.
- Align compiler-training examples across character sequence, aggregate token,
  compiler topology, layout path, and optional rendered asset. Treat token IDs
  as categorical identities and assign versioned dense embedding rows.
- Test round trips, cross-backend identity, collision handling, cross-repository
  lookup, and the Windows `spawn` fuzzer path before changing existing string
  behavior.

Authoritative audit:
`../1792849246_AUDIT_Turing_Nodus_Spectral_Glyph_Token_Architecture.md`
