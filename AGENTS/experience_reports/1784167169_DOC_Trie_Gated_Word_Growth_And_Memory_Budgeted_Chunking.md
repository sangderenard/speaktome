# Documentation Report

**Date:** 1784167169
**Title:** Trie-gated word growth (both directions) and memory-budgeted expand-batch chunking

## Overview

Direct continuation of the same session as
`1784162289_DOC_FluxGraph_GPU_Efficiency_Pass_And_Choice_Policy_Vectorization.md`
(read that first for the empty_cache()/choice_policy history). The user ran
the demo live (`--ticks 20 --budget 3 --branch 5 --alpha .5 --beta 10 --poetic
--visualize`) and hit three real problems at once: a `torch.OutOfMemoryError`
crash, ticks taking minutes instead of seconds, and completely non-dictionary
output like `'raAbAbAbAbAbAbAballah ulenceately'` despite `--auto-dictionary`
being on. This session diagnosed and fixed all three.

## Root causes found (not what I initially guessed)

1. **The OOM was in `_expand_batch`'s own round-0 chunked loop**
   (`flux_graph.py`, the shared forward+backward batch built once per
   tick), not inside word growth's per-beam loop as I first assumed from
   the traceback's surface shape. `expand_batch_chunk_size` (2048) is a
   *flat* row-count cap that never accounts for `row_len` growing over a
   run -- `2048 rows x ~20-30 growing row_len x 50257 vocab` in float32 is
   already multiple GB, and neither `empty_cache()` nor anything about
   word growth touches that number.
2. **The garbage output** (`AbAbAbAballah`) was a real, reproducible bug:
   backward word growth's per-token dictionary filter is *flat and
   stateless* -- it only ever asks "is this one token's own decoded text a
   dictionary word" (yes for "Ab" -> "ab", yes for "allah", apparently in
   the curated wordlist), never "does the *accumulating* span still head
   toward a real word." Nothing stopped the same short, individually-valid
   fragment from being prepended over and over. Confirmed by decoding the
   actual token ids from the user's log (4826="Ab", 6242="AB",
   31840="allah") before writing a line of new code.
3. **The slowness** was `_grow_backward_word` rescoring the *entire*
   ~24,431-candidate dictionary-filtered pool, unbatched across beams, on
   *every single subword step* -- a pre-existing, previously-documented
   scope-limit that `--poetic`'s wider `poetic_shortlist_k` (20 vs
   `branch_factor`'s 5) made much worse by feeding more parallel beams
   into that same expensive per-step rescan.

## What changed

**`speaktome/core/word_trie.py`**
- `WordTrie` gained `root_node()`, `is_end(node)`, `walk_from(node, s)` --
  incremental walking from a saved state, not always from root -- and a
  `reverse=True` constructor mode that indexes words backwards
  (`word[::-1]`), needed because backward growth discovers a word from its
  end toward its start (each token gets *prepended*).
- New `TrieGate` class: given a fixed vocabulary pool (decoded once) and a
  trie node, returns exactly the pool candidates that validly continue
  from there, cached by node identity so the (CPU-only, no model call)
  decode-and-walk cost is paid once per unique trie node ever visited
  across a whole session, not once per revisit. Reverse-aware: when the
  underlying trie is reversed, candidate text gets reversed before
  walking too (prepending in normal-text space is *appending* in
  reversed-representation space) -- caught this as a real bug via a
  from-scratch differential test before it ever touched the real word-
  growth code (first attempt produced `[([1], -0.1)]` instead of growing
  "running" at all).

**`speaktome/core/flux_graph.py`**
- `_grow_forward_word`: boundary detection (the `starts_new_word` check
  against the model's own top-k over the full vocab -- free, since the
  step's forward pass already produces that distribution) is unchanged.
  What changed is *continuation* candidate selection: instead of taking
  the model's top-k and discarding whichever don't extend a valid trie
  prefix (which could starve a beam outright if none of the top picks fit
  the dictionary), the trie's own children at the beam's current state
  are looked up first (`TrieGate`, no extra model call -- just a gather
  on logits already in hand) and `choice_policy` only ranks among
  already-guaranteed-valid candidates. `word_trie=None` falls back to the
  exact original behavior (a real test calls this function directly with
  `word_trie=None`, bypassing the outer gate -- had to preserve that
  path).
- `_grow_backward_word`: now uses a *second*, reversed `WordTrie`
  (`FluxGraphConfig.backward_word_trie`, new field) -- growth candidates
  are that trie's children from the beam's state, scored via one
  `score_candidates` call over just that narrowed set instead of the
  whole ~24k pool every step. `backward_word_trie=None` falls back to the
  original score-everything-then-topk behavior exactly.
- New `FluxGraphConfig.max_expand_elements` (default `None` = unchanged
  behavior) and `FluxGraph._plan_expand_chunks`: a greedy chunk planner
  that bounds `rows x row_len x vocab_size` directly instead of just row
  count, so the effective row cap shrinks as a chunk's own row_len grows.
  Applied to `_expand_batch`'s round-0 chunking (the actual OOM site) and
  to `_grow_backward_word`'s no-trie fallback scoring call. A single row
  that alone exceeds the budget still gets served in its own one-row
  chunk rather than dropped.

**`speaktome/demo_flux_graph.py`**
- Builds a second, reversed `WordTrie` alongside the existing forward one
  whenever word growth is enabled (both the `--dictionary-file` and
  `--auto-dictionary` paths), passed through as
  `FluxGraphConfig.backward_word_trie`.
- New `--max-expand-elements` flag, default `400_000_000` (~1.6GB/call in
  float32) -- opt-out via `--max-expand-elements 0`, matching this file's
  existing `0`-disables convention (e.g. `--no-repeat-ngram-size`).
  `FluxGraphConfig`'s own dataclass default stays `None` (no behavior
  change for library callers who don't opt in); the demo defaults to the
  safer behavior since that's what actually crashed.

## Steps taken / verification discipline

Given two reverted mistakes earlier in this same session (removing
`empty_cache()` based on one lucky profile, then a bucketed-padding
attempt that made memory *worse*), verification here was deliberately
layered, cheapest-and-most-certain first:

1. Every new WordTrie/TrieGate primitive got a standalone manual sanity
   check (`python -c "..."`) before being wired into anything real.
2. The reversed-trie candidate-text bug (see above) was caught this way,
   before it ever reached `_grow_backward_word`.
3. Full existing test suite (91 tests, all direction) re-run after *every*
   function rewrite, not just at the end.
4. New permanent tests added: `WordTrie.walk_from`/`is_end`/`reverse`,
   `TrieGate` (including the reverse-candidate-text case explicitly),
   `_plan_expand_chunks` (flat-cap-preserved, budget-shrinks-with-row-len,
   oversized-single-row-still-served), and a `_grow_backward_word`
   regression test that reproduces the exact "Ab"-repetition failure mode
   with a dummy model engineered to prefer repeating a short
   dictionary-valid fragment forever -- this is the test that would have
   caught the original bug and now guards against it recurring.
   99 -> 102 tests total (all passing).
5. Only then, real GPT-2 verification, in increasing order of how much it
   resembled the failing scenario:
   - A quick 3-tick sanity run: output changed from garbage to real
     coherent English ("the quick brown foxes that chewed on us").
   - The *exact* `--seed-rng 42` scenario that OOM'd earlier this session
     (documented in the prior report): all 5 ticks completed, no crash.
     Tick timing dropped from 462.4s/288.4s (crashed on tick 3 before) to
     16.1s/2.6s/2.5s/2.8s/2.9s -- roughly 100x faster on later ticks, not
     just "no longer crashes."
   - A scenario close to the user's actual original command (`--budget 3
     --branch 5 --alpha .5 --beta 10 --poetic --auto-dictionary`, minus
     `--visualize` since that's pygame UI, not relevant to verification):
     8 ticks, 3-45s each, no crash, all real dictionary words in the
     output (some awkward mid-word splits like "oklaho"+"ma" as separate
     path segments -- likely `max_subword_steps` truncating a longer word
     before it finishes; a real but much smaller residual issue than the
     repetition garbage, and the semantic incoherence itself is expected
     given how exploratory `alpha=0.5, beta=10` deliberately are).

## Lessons learned

- **A traceback's call stack is ground truth; my first mental model of
  where a problem lives is not.** I initially attributed the OOM to word
  growth's per-beam loop before actually reading the traceback closely
  enough to see it was `_expand_batch`'s own round-0 chunking -- a
  different code path with a different fix.
- **Reproduce the exact failing input in isolation before trusting a
  fix.** Decoding the specific token ids from the user's own log
  (4826/6242/31840) up front turned "the dictionary filter isn't working"
  into a precise, falsifiable claim ("each token passes a flat per-token
  check individually") before any code was written -- and directly
  motivated the actual fix (stateful trie walk) rather than a guess.
- **A from-scratch differential/sanity check catches bugs a unit test
  suite -- built around the *old*, buggy design -- structurally can't.**
  The reversed-candidate-text bug in `TrieGate` would have passed every
  existing test (none of them exercised `backward_word_trie` at all,
  since it didn't exist before this session) and only showed up because a
  manual script was run against the new code before wiring it in.
- **A flat row-count cap and a memory-safe cap are different things once
  any other dimension of the batch can grow.** `expand_batch_chunk_size`
  was never wrong on its own terms (it does cap row count); it just
  doesn't know that `row_len` growing over a run changes what "2048 rows"
  costs. The fix isn't a bigger constant, it's bounding the product
  directly.

## Next Steps

- The `max_subword_steps` truncation potentially splitting long real
  words (e.g. "oklahoma" -> "oklaho" + "ma") across two separate graph
  nodes is a real, smaller residual issue observed in verification,
  distinct from the repetition-garbage bug this session fixed. Not
  investigated further here -- flagging for whoever picks this up next.
- The stale `.venv`/`tests/conftest.py` pytest gate noted in the prior
  report is still unresolved; this session continued running tests via
  direct function-call invocation against system Python 3.11.
- `max_expand_elements`'s default (400,000,000, ~1.6GB/call) was chosen
  to leave real headroom on the 12GB GPU this was tested on, not derived
  from a principled per-GPU calculation -- worth revisiting if this runs
  on meaningfully different hardware.

## Prompt History

- "why does the process take much longer now (don't try to change
  anything we need to talk about things)? why is it going out of memory?
  why does this traversal seem to not contain any dictionary limited
  content, what is the object taking up all the memory" (followed by a
  pasted real demo run + traceback)
- "what do you mean 'one backward pool scan per node' and what are you
  talking about using beam terminology for our graph?"
- "okay so instead of searching like it should search it's just a, what,
  topk in either direction? with no filter for needing to be forming a
  dictionary word? all those searches for a whole word need to have the
  vocabulary gating making a faster and faster search that can't result
  in non-words"
- "yes, proceed, but don't let the oom thing get totally out of hand,
  there needs to be a real system control keeping that in check
  elegantly, I don't know exactly how that's getting overloaded again"
