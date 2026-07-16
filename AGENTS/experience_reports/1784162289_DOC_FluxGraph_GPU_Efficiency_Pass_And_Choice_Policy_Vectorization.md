# Documentation Report

**Date:** 1784162289
**Title:** FluxGraph GPU efficiency pass (mostly a correction of my own mistakes) and choice_policy.py sampling vectorization

## Overview

Continuation of the GPU efficiency pass stubbed in `implicit_backpath.py`
above `ImplicitBackpathScorer._score_batch` (see prior report
`1784144799_DOC_FluxGraph_Word_Level_And_Physics_Extensions.md`). The stub's
own instruction was "profile a real GPT-2 run first ... before optimizing
blind" -- followed that, and it saved me from landing a change that looked
like a clean win but actually caused OOM. Two of the three real changes made
here were attempts that had to be reverted; the third (choice_policy.py) is
the one that actually landed. Recording the failed attempts in detail
because the *reasoning about why they failed* is the valuable part for
whoever picks up memory-footprint work on this pipeline next.

## Steps Taken

1. Fixed the stale `.venv` blocker: it pointed at a Python 3.10 install no
   longer on this machine. System Python 3.11 already had every needed
   package (torch, pytest, nltk, wordfreq, transformers) installed directly
   to site-packages by a previous session -- ran tests by importing test
   modules directly (`python -c "import tests.test_x; tests.test_x.test_foo()"`)
   rather than through pytest, since `tests/conftest.py` hard-gates on
   `sys.executable` living under `ROOT/.venv` with a marker file, which no
   longer matches. Did not attempt to repair the venv/poety machinery
   itself -- out of scope, and the repo's poetry config wants Python
   >=3.12, which also isn't installed. This is a real, unresolved gap for
   whoever wants proper `pytest` invocation back; recorded here rather than
   silently patched around.
2. Profiled a real GPT-2 run (`python -m cProfile -m speaktome.demo_flux_graph
   "the quick brown fox" --ticks 3 --budget 2 --branch 3 --auto-dictionary`)
   before changing anything. Found two real costs, not the ones the stub's
   TODO list guessed at:
   - `torch.cuda.empty_cache()`, called after every chunk in
     `_score_batch`/`_expand_batch`/`_grow_forward_word`: 9.1s of pure
     synchronization overhead across 26 calls in that one profiled run.
   - `choice_policy.py`'s `AlphaBetaPolicy._weighted_sample_without_replacement`:
     16.4s of pure-Python self-time (plus another ~5.4s in the sibling
     `mixed_weights` listcomp) across just 137 calls -- a full-vocab
     (~50,257) `pow()` + Python-level sort on *every* `choose()` call even
     though only `k` (e.g. 3) candidates ever survive. More raw CPU time
     than any GPU work in the same window. Outside the stub's named files.
3. **Attempt 1 (reverted):** removed the `empty_cache()` calls based on
   that first profile. Re-profiled with `--seed-rng 42` (a different,
   more backward-heavy growth pattern) to check robustness -- this run
   took 865s instead of 100s and crashed with a real
   `torch.OutOfMemoryError` on tick 3, after ticks 1-2 took 462s and 339s
   each. Reverted immediately, restored `empty_cache()`, added a comment
   (later found to be wrong -- see Attempt 3) claiming it was "confirmed
   load-bearing."
4. **Attempt 2 (reverted, worse than the crash it was meant to fix):**
   read the `_score_batch`/`_expand_batch` allocation shapes and
   confirmed row lengths change nearly every tick as context grows --
   textbook CUDA allocator fragmentation. Implemented `padded_length()`:
   round every padded sequence length up to a fixed 32-token bucket, so
   consecutive ticks mostly request the identical tensor shape instead of
   a new one each time. All 91 tests passed (padding is masked/unread, so
   this is provably a pure "how" change). Re-ran the exact `--seed-rng 42`
   scenario that had OOM'd to verify the fix -- it OOM'd *immediately*,
   on the very first call (`spawn_first_children`, never even reached
   tick 1), trying to allocate 12.27 GiB. Root cause: rounding a *short*
   initial sequence (row_len=6, before any ticks have grown context) up
   to a 32-token bucket is a >5x size inflation, and that gets multiplied
   by `expand_batch_chunk_size` (2048) and `vocab_size` (50,257) --
   turning a small request into one bigger than the whole 12GB GPU. Fixed
   granularity bucketing is fundamentally the wrong tool when the same
   dimension gets multiplied by a huge constant (vocab) at large batch
   size -- a fine graduation for short sequences becomes catastrophic
   relative overhead. Reverted immediately.
5. **The actual correction, found by testing the counterfactual I should
   have tested before Attempt 1:** ran the *original, fully unmodified*
   code (empty_cache present) against the exact same `--seed-rng 42`
   scenario that Attempt 1's revert was "protecting." It hit the
   identical `torch.OutOfMemoryError` at tick 3, with near-identical
   per-tick timing (463.8s/288.4s vs Attempt 1's 462.4s/339.6s).
   `empty_cache()` was never preventing this crash -- it was going to
   happen either way, because `expand_batch_chunk_size` (2048) candidates
   x `vocab_size` (50,257) x a growing context length, in float32, can
   exceed a 12GB GPU's capacity regardless of allocator hygiene. This is
   an absolute-memory-footprint problem, not a fragmentation problem, and
   no amount of `empty_cache()`/bucketing changes what "how it's
   computed" without changing chunk size. Redid the `empty_cache()`
   removal (safe now, confirmed no downside) with an honest comment
   explaining what was actually learned, instead of the disproven
   "load-bearing" claim from step 3.
6. **The change that actually landed:** vectorized
   `choice_policy.py`'s `AlphaBetaPolicy` (`mixed_weights` and
   `_weighted_sample_without_replacement`) with plain numpy -- CPU-only,
   zero GPU memory involvement, so none of the above risk applies.
   Verified bit-for-bit equivalence against the original Python-loop
   implementation over 250+ seeded trials (realistic ~50k-vocab weights,
   small vocabularies, all-positive/some-zero/some-negative/all-zero
   weight distributions, k=1 through k=n) in a scratch differential-test
   script before touching the real file -- matched exactly in every case.
   Key subtlety: `self._rng.random()` must be drawn *only* for
   `weight > 0` candidates, in ascending index order, exactly like the
   original's `if w <= 0.0` branch -- drawing for every index
   unconditionally (the natural first instinct) changes the RNG stream
   whenever any candidate is masked out, silently breaking
   seed-reproducibility. Used `np.power(2.718281828459045, x)`, not
   `np.exp(x)` -- verified these are not bit-identical (differ by 1 ulp),
   and `np.power` matches the original per-element `pow()` exactly.
   `np.argsort(-keys, kind="stable")` mirrors Python's
   `list.sort(reverse=True)` tie-breaking for equal keys (both keep
   ascending-index order for ties), which matters when several `w <= 0`
   candidates tie at `key=-inf`. Benchmarked: ~2.9x on the sampling step
   alone, ~31x on the `mixed_weights` computation, at realistic
   50,257-vocab scale.

## Observed Behaviour

All 91 tests across the same suite the previous session verified
(`test_flux_graph`, `test_implicit_backpath`, `test_choice_policy`,
`test_writing_token_filter`, `test_token_filters`, `test_word_trie`,
`test_word_boundary`, `test_word_sources`, `test_poetic_attractor`,
`test_graph_layout`) pass at every checkpoint in this session, including
after each revert and after the final landed changes. (2 of those tests
need pytest's `tmp_path` fixture and can't run via bare function calls in
this venv-less setup -- same gap as note 1 above, not a regression.)

## Lessons Learned

- **Test the counterfactual before attributing causation.** Attempt 1's
  entire false start happened because I saw "I made one change, then a
  crash" and concluded the change caused the crash, without first
  checking whether the *original* code also fails under the same harder
  conditions. It did. A one-directional A/B test (before vs. after a
  change) is not enough when you haven't also confirmed "before" was
  actually safe under the condition that broke "after."
- **A fix that "looks like a win" on one profiling run needs a harder
  adversarial test before you trust it**, especially for GPU memory
  behavior, where a single lucky (or unlucky) random seed changes the
  regime entirely. The stub said "profile before optimizing blind" --
  the missed half of that advice was "profile *several* scenarios,
  including a deliberately harder one," not just one.
- **Fixed-granularity padding is the wrong tool when the padded
  dimension gets multiplied by something huge (vocab_size here).**
  Bucketing to reduce allocator churn is a legitimate technique in
  general, but its overhead is *relative to the padded dimension itself*
  -- rounding a short sequence up to a fixed bucket can be a >5x blowup
  precisely when short sequences are common (e.g. right after `seed()`,
  before any ticks have grown context), and that blowup gets multiplied
  by every other dimension of the batch. If bucketing is revisited, it
  needs proportional/multiplicative growth (e.g. next power of 2) or a
  bucket size chosen relative to the *smallest* expected sequence, not a
  flat constant picked without checking that arithmetic.
- **Fragmentation and absolute footprint are different failure modes,
  and it's easy to misdiagnose one as the other.** The actual ceiling
  here is `expand_batch_chunk_size (2048) x vocab_size (50,257) x
  row_len`, in float32, against a 12GB GPU -- a capacity question, not
  an allocator-hygiene question. No purely-internal "how it's computed"
  change fixes a capacity ceiling; only reducing what gets computed
  (smaller chunks) would, and that's a real behavior/performance
  tradeoff, not a free efficiency win.
- **CPU-only optimizations (choice_policy.py) carry none of the GPU
  memory risk that GPU-tensor-shape changes do**, and are much cheaper to
  verify exhaustively (a scratch differential test against hundreds of
  seeded trials takes seconds to run). When a stub's efficiency work
  spans both GPU and CPU-bound code, the CPU-bound part is the
  lower-risk place to actually land changes same-session; GPU-memory
  changes deserve more adversarial testing before landing, not less.
- **RNG-stream preservation across a vectorization is a real, easy-to-miss
  correctness hazard**, not a formality: the original code's `if w <= 0.0`
  branch means the number and order of `random.Random.random()` calls
  depends on the *content* of the weights, not just their count. A
  vectorized rewrite that draws unconditionally for every index looks
  correct (same formula) but silently changes what a given seed produces
  the moment any weight is masked out.

## Next Steps

The GPU-side memory ceiling (`expand_batch_chunk_size`/`max_batch_size` x
`vocab_size` x context length exceeding available GPU memory on long runs
or heavy backward-word-growth scenarios) is real and unresolved -- it is
not something either of my reverted attempts fixed, and it predates this
session. If it's worth addressing, the actual lever is reducing
`expand_batch_chunk_size`/`max_batch_size` (fewer candidates scored per
single forward call, trading off more, smaller calls) -- a genuine
behavior/performance tradeoff that changes runtime characteristics, not a
"how it's computed" efficiency-only change, so it should be raised with
the user rather than done unilaterally. The stale `.venv`/`tests/conftest.py`
gate (item 1 above) is also unresolved: system Python 3.11 works fine for
everything actually needed, but real `pytest` (not direct function-call
invocation) requires either fixing `.venv` to point at an installed
interpreter or relaxing `conftest.py`'s `_venv_marker_ok()` gate --
neither attempted here, out of scope for a GPU-efficiency task.

The `_grow_forward_word`/`_grow_backward_word` per-beam batching
scope-limit noted in the original stub (word growth's follow-up rounds
aren't batched across beams the way round-0 is) is still open and
untouched by this session.

## Prompt History

- "read agent info, you are encoraged to explore and understand the repo
  before working"
- "what was the last experience report, where as the previous agent left
  you"
- "start on the known work, make a judgement call on the pytest, we're
  just not using that python version anymore, this is a really old repo"
- (mid-session, via AskUserQuestion after the empty_cache/choice_policy
  findings were surfaced) "Both" -- do the persistent-buffer/root-cause
  fix in the stub's named files first, then also tackle
  choice_policy.py's sampling bottleneck.
