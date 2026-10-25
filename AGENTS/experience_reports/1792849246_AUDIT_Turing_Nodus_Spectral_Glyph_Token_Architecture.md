# Turing, Nodus, and Spectral Glyph-Token Architecture Audit

**Date:** 2026-10-24
**Title:** Character-token sequences, aggregate token identity, and glyph manufacturing

## Scope

This audit expands the unscoped `C10: Strings as values` row recorded by
Turing PR #890. It searches all of Turing, Nodus, and the relevant
Spectral Analyzer glyph-manufacturing code for the intended non-string
representation:

1. decode an external character sequence into canonical character identities;
2. represent the ordered sequence with integers;
3. intern that sequence and mint one aggregate integer token;
4. perform equality, lookup, composition, training, and runtime transport over
   integers and table edges;
5. recover character/glyph/render data only through explicit library lookups.

The audit also retains the later Windows test procedure enabled by Turing PR
#906. No compiler, Nodus, or Spectral Analyzer source was changed and no tests
were run during this audit.

## Methodology

- Fast-forwarded the independent `C:/dev/Powershell/turing` repository from
  `32de07ff` to `ab7d7185` with `git pull --ff-only`. Existing untracked
  `.claude/` and `out/` content was preserved.
- Read the workspace, Turing, compiler, tensor, hazards-log, guestbook, and
  testing guidance before inspecting source.
- Read PR #906 and its merged commit `3f2e7550`, including the Windows
  multiprocessing change and hazard receipt.
- Read PR #890 and its consolidated plan change. The current plan retains the
  row `C10: Strings as values` as open and unscoped.
- Searched tracked source and documentation across Turing, Nodus, and Spectral
  Analyzer. Generated artifacts, caches, build trees, and nested agent
  worktrees were excluded from conclusions.
- Inspected the concrete Turing string table, glyph-sequence materializer,
  compiler-training token store, Nodus KPath atlas/shaper/token pipeline, and
  Spectral Analyzer render-asset catalog, character atlas, token composer, and
  render-object library.
- Compared claims in documentation with the code paths that actually allocate
  and consume identities.

## Detailed Observations

### 1. Turing already has two halves of the intended representation

`src/compiler/string_table.py` defines a universal signed-i64 word identity as
FNV-1a over UTF-8 bytes. `StringTable` records token-to-text reverse lookup.
`src/compiler/ir_string_interning.py` rewrites string constants to
`string_token`, makes constant concatenation produce a token, and tags equality
as integer-token comparison. `src/compiler/ir_string_ops.py` explicitly keeps
bytes in string views and says token-to-bytes is a display-only table lookup.

Separately, `src/compiler/glyph_sequence_format.py` represents authored
spellings as `list[int]`, one `string_token(character)` per character.
`src/compiler/resident_sequence_materialization.py` constructs a minted
`STRING_GLYPH_VOCABULARY` whose rows map a whole-word token to sequence length
and `(ordinal, character-token)` entries. The glyph vocabulary therefore
already expresses:

```text
whole-word token -> ordered sequence of character tokens
```

This is strong evidence for the intended architecture. It is not yet the
canonical minting rule for the whole-word token: the whole-word token is still
computed directly from the original bytes, while the character-token sequence
is a derived rendering/provenance table.

The distinction is deliberate today. The regression
`test_dynamic_scalar_string_token_is_not_coerced_to_glyph_sequence` prevents a
dynamic scalar token from being silently reclassified as a resident glyph
sequence.

### 2. Turing's exact identity encoder is stronger than its hot string token

`src/common/tensors/abstract_nn/token_encoder.py` already provides an injective,
reversible base-257 integer encoding of canonical JSON bytes. The compiler
training database stores lossless token sequences and one integer token ID per
position. Reduced graphs retain exact `ssa_identity_tokens` and dense-slot
boundaries.

This is suitable as an exact proof or vocabulary identity, but its integers are
unbounded and should not be confused with a fixed-width runtime handle. The
current FNV-1a i64 string token is compact, but it is a hash. `StringTable.intern`
uses `setdefault` and does not reject a distinct string that reaches the same
64-bit token. Therefore the current i64 is a useful lookup key, not a
mathematically injective identity over all finite strings.

### 3. Nodus owns the structural glyph/token-path substrate

Nodus KPath decodes UTF-8/UTF-16 into Unicode scalar `CodepointSequence`s,
shapes them into HarfBuzz clusters and glyph IDs, extracts FreeType outlines,
and records codepoint/glyph nodes plus layout edges. Its documentation states
the core law directly: a token is a sequence of atlas edges. `TokenId` is a
strong integer type; token paths retain the edge span, and postings retain
token/offset/span provenance.

`AtlasBuilder::add_token_dedup` hashes an edge list but verifies candidate edge
lists before reuse, which is the correct collision-safe interning shape.
However, the production `build_token_from_sequence` path calls `add_token`, not
`add_token_dedup`. It also creates new codepoint nodes, glyph nodes, and edge
IDs on each build. Thus identical text built twice does not yet necessarily
reuse one token. The dedup helper is present, but the canonicalization required
to make equal character/glyph sequences share equal edge sequences is not yet
installed in this path.

Nodus also states that no general string-interning utility exists yet, and a
separate sparse-graph builder locally interns attribute-key strings to u32
indices. This confirms that KPath is the correct substrate but has not yet been
promoted to a repository-wide string authority.

### 4. Spectral Analyzer owns the glyph manufacturer and physical library

`camera_software/render_assets.py` defines `ExtrudedTokenAsset`, stable
content-addressed asset keys, `CharacterAtlas`, a persistent
`RenderAssetCatalog`, and bake requests for individual atlas glyphs and whole
token sequences. `LIGHT_FIELD_ASSET_ARCHITECTURE.md` states the lookup order:
exact sequence first, then character-atlas fallback. Requests are
content-addressed and deduplicated.

`camera_software/sprite_compositor.py::CachedTokenStringComposer` implements a
three-level materialization policy:

```text
exact full string -> exact non-whitespace segment -> individual character
```

`camera_software/render_object_library.py` supplies the same preference for
3-D scene objects and retains missing token/character requests explicitly.
This is a real glyph-manufacturing system, not a presentation mock. It stores
font source identity, extrusion/material recipes, physical light-field
conditions, ray-traced evidence, convergence state, and reusable whole-token
or glyph products.

The current identity payload still contains the raw Python string. It does not
consume Turing character tokens or a Nodus token path. Its SHA-256 key is
portable and content-addressed, but it is an asset identity over text plus
render context, not the shared text token itself.

The relevant glyph-library files were last changed in tracked history on
2026-07-22 (`c4cb29fe`). The repository's most recent tracked change on
2026-09-14 concerns shader/material compilation, and the newer untracked
electrical-network files do not add a competing glyph/string authority.

### 5. The three systems form a natural ownership chain

Observed responsibilities are:

```text
Turing
  authored/source identity, compiler provenance, integer runtime operations,
  reversible training records

Nodus KPath
  codepoint -> cluster -> glyph -> layout-edge path, aggregate TokenId,
  postings and backend-independent path/tape transport

Spectral Analyzer
  font/extrusion/material/light-field manufacture, persistent rendered asset
  evidence, exact-token lookup and character fallback
```

The missing item is a shared identity contract across those authorities. A new
parallel string runtime would duplicate systems that already exist.

## Analysis

### Canonical mathematical contract

Let `C` be the set of Unicode scalar values and `I` the integer identity
space. Let `a : C -> I` intern one character and let a source string
`s = (c_0, ..., c_{n-1})` have canonical atom sequence

```text
S(s) = (a(c_0), ..., a(c_{n-1})) in I*.
```

The text vocabulary owns a collision-safe interner

```text
m_text : I* <-> I_text
```

with a retained inverse table `expand_text(m_text(S)) = S`. Within the
resident vocabulary, the required law is

```text
m_text(S_1) = m_text(S_2)  iff  S_1 = S_2.
```

A fixed-width hash may index candidate rows, but it cannot be the proof of
identity. Equality must either compare the retained sequence on hash collision
or use a minted dense ID whose unique sequence row is the authority.

Ordinary text operations then become integer/path operations:

```text
equal(x, y)       := x == y
length(x)         := |expand_text(x)|
character(x, i)   := expand_text(x)[i]
concat(x, y)      := m_text(expand_text(x) ++ expand_text(y))
slice(x, i, j)    := m_text(expand_text(x)[i:j])
```

The hot representation is an integer token or a span into integer sequence
storage. Raw UTF bytes exist at ingestion, external I/O, or diagnostic/display
boundaries.

Text identity must remain separate from shaping and physical manufacture.
Given shaping context `q` (script, language, OpenType features, font digest),
Nodus owns

```text
shape_q : I* -> E*
m_layout : (I_text, q, E*) <-> I_layout,
```

where `E*` is the retained KPath edge sequence. Spectral Analyzer then owns an
asset lookup/mint over the layout identity and physical context `r`:

```text
m_asset : (I_layout, r) -> I_asset.
```

This separation prevents a font, ligature, material, or lighting change from
changing semantic text equality while still making manufactured artifacts
content-addressed and reproducible.

### LLM/NN training consequence

The integer identity is categorical, not ordinal. A network must never learn
from the numerical magnitude of `TokenId`. Persist the canonical token and map
it through a versioned dense vocabulary row before embedding:

```text
v : I_text -> {0, ..., V-1}
embedding(t) = W[v(t)].
```

The compiler-training store should retain aligned, lossless views:

```text
(character-token sequence,
 aggregate text token,
 compiler structural-token sequence,
 Nodus layout token/path,
 optional Spectral asset token,
 provenance and verification receipts).
```

That supports several honest training objectives without making a model an
identity authority:

- next character-token or next structural-token prediction;
- sequence-to-aggregate-token retrieval;
- aggregate-token-to-sequence reconstruction;
- source/SSA/machine-topology alignment;
- layout/render retrieval conditioned on explicit context;
- contrastive learning where equal canonical sequences are positives and
  distinct retained sequences are hard negatives.

Every learned output remains provisional until the deterministic interner,
compiler, or renderer verifies it. An untrained model cannot mint canonical
identity.

## Recommendations

1. Expand PR #890's C10 row into a cross-system contract, not a compiler-only
   string feature. Name the existing authorities explicitly: Turing
   `StringTable`/glyph provenance, Nodus KPath atlas, and Spectral Analyzer
   render-asset library.
2. Make the character-token sequence the canonical text record. Mint the
   aggregate text token from that exact sequence with a retained inverse and
   collision proof.
3. Preserve separate identities for semantic text, shaped layout, and physical
   render assets. Do not put font or material context into text equality.
4. Reconcile Turing's whole-word FNV token with its existing minted
   `STRING_GLYPH_VOCABULARY`. Do not add a second ad hoc interner.
5. In Nodus, canonicalize/reuse codepoint, glyph, and layout edges before using
   collision-checked token-path deduplication. Persist the resulting token
   vocabulary if cross-process stability is required.
6. Let Spectral Analyzer accept canonical text/layout token references plus
   explicit rendering context while retaining source text only as reversible
   metadata and external UI data.
7. Align the compiler-training lexicon with the same canonical identities.
   Keep dense embedding-row assignment versioned and separate from identity.
8. Require cross-repository round trips and collision tests before replacing
   current string paths. No broad rewrite should begin from this audit alone.

The implementation continuation is recorded in
`todo/1792849246_turing_nodus_spectral_glyph_token_interner.stub.md`.

## Retained test procedure from PR #906

When implementation work resumes, testing begins by reading the newest Turing
hazard receipt with `python tools/hazards_log.py`. The current full suite is not
a sanctioned baseline.

PR #906 made `tools/differential_fuzz.py` usable on Windows by selecting a
module-level `spawn` child there and retaining `fork` elsewhere. The rerun tool
is main-guarded. The later battery is:

1. focused token/glyph/interner unit tests and cross-repository round trips;
2. the native semantics corpus on every shared backend using
   `TYPED_CHECK_BACKENDS=c,llvm,fortran`;
3. the relevant differential-fuzz tier, with zero `WRONG` and zero `TYPE`;
   PR #890 calls for `FUZZ_EXT=5`, but current `main` documents only tiers
   1-4, so confirm that tier 5 has landed before invoking it;
4. the four-part regression gate against its committed baselines, with every
   part log ending in an actual pytest summary;
5. a new per-PR hazards entry created by `python tools/hazards_log.py new`.

The cloud Linux gate remains authoritative. Windows concerns are material when
the change touches Windows build paths or vendoring. Existing W1/W2 Windows
hazards remain separate from this architecture.

## Prompt History

> pull turing look at pr 906 content for testing the system and providing rich feedback to an ongoing project that will resume in a few days. save that process in your memory so that when we are done with this next mentioned task, you will be able to perform the tests requested.look at pr 890. This is vital and a critical moment to express it in very pure engineering and math with forward thought for llm/nn training - expand on 890's task by including searches in nodus and across all of turing not just the compiler, check some of the more recently modified files in spectral analyzer maybe? look for evidence in source and documents that has to do with the design intention to not have traditional string programming, but instead to intern every character sequence into it's token sequence, then minting a new token for the whole, so that all actual string activity is integer activity with memory lookups.

> look into the systems that manage glyph libraries and strings, I'm almost certain nodus has that, and some stuff in spectral analyzer is specifically a glyph manufacturer, that's how we manage the strings

> You will not do a better job than the systems already here

> Never "find a better way." If the task names a mechanism, use that mechanism as it is spelled.
