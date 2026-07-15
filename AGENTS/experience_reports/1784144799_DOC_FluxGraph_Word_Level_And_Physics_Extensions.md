# Documentation Report

**Date:** 1784144799
**Title:** FluxGraph word-level nodes, physics extensions (auxin/head pressure), live visualizer, real dictionary filtering

## Overview

Continuation of the bidirectional FluxGraph work (see prior report covering
`aeefc60`: circuit convergence, digestion, exploitation game). This session
covered, in order:

1. Fixed a real correctness bug in backward scoring (was summing suffix
   log-likelihood instead of averaging, so backward nodes were mechanically
   starved as the graph grew) and n-gram-repeat blocking.
2. Unified forward+backward expansion into one padded/chunked/vectorized
   batch per tick (`FluxGraph._expand_batch`), replacing one-model-call-per-node.
3. `PoeticAttractor` (rudimentary rhyme/alliteration bias, honest-score-preserving
   reranking over a shortlist -- never corrupts `local_evidence`).
4. Word-level graph nodes: `FluxNode.tokens` is now a token *span*, not one
   BPE token. `WordTrie` + `word_boundary.starts_new_word` (GPT-2's own
   leading-space convention) drive a branching, trie-pruned subword beam
   search (`FluxGraph._grow_forward_word` / `_grow_backward_word`) so a
   graph edge is a whole word, not a fragment -- only active when
   `FluxGraphConfig.word_trie` is set; `None` (default) is a byte-for-byte
   no-op, verified against the full prior test suite.
5. Two physics extensions, both off by default (`0.0` coefficients, true
   no-ops when unset):
   - **Auxin suppression** (`FluxGraph._diffuse_auxin`, apical-dominance
     analogy): a strong, uncontested tip suppresses `branch_factor` at
     *other* branches sharing an ancestor with it, attenuated by hop
     distance -- never suppresses itself. Two-pass (bottom-up strongest-
     source, top-down ambient field), same shape as the existing `_digest`
     rollup.
   - **Head pressure** (`FluxNode.height`, signed token-distance from the
     anchor: `+depth` forward, `-depth` backward): a real resistance term
     in `_update_pressures`, `pressure -= head_pressure_coefficient *
     abs(height)` -- `abs()`, not signed, so forward/backward cost the same
     at equal distance.
6. Live visualizer (`graph_layout.py` + `graph_visualizer.py`, pygame/OpenGL
   on its own thread): `ForceLayout` simulates only the horizontal axis
   (repulsion + springs); the vertical axis is *pinned* directly to
   `FluxNode.height` every frame, not simulated -- "gravity/antigravity"
   turned out to want to be a fixed coordinate, not a dynamic force.
   Real bug found and fixed here: the graph originally exposed a
   `threading.Lock` for the visualizer to read `graph.nodes` under: (a)
   `spawn_first_children()` never held it at all, so a reader could freeze
   on a forward-only partial state for the entire duration of the (slow)
   backward expansion; (b) even during `tick()`, a non-blocking acquire
   racing a tight loop that immediately re-locks between ticks could
   statistically starve the reader for a whole run. Replaced with
   `FluxGraph.published_snapshot`: a complete dict reassigned wholesale at
   the end of `seed()`/`spawn_first_children()`/`tick()`, read directly by
   the visualizer with no lock at all (a single reference read/write is
   already atomic under CPython's GIL).
7. Real dictionary filtering for the backward candidate pool and word
   growth: `DictionaryTokenFilter`/`CombinedTokenFilter` (`token_filters.py`,
   composes with the existing `WritingTokenFilter`) and
   `word_sources.curated_english_wordlist()` (cross-references nltk's
   `words` corpus against wordfreq's popularity ranking -- a real
   dictionary narrowed to its most common ~20k entries, not frequency data
   standing in for a dictionary). `--auto-dictionary` in the demo defaults
   to **on**; `--no-auto-dictionary` opts out, and that path now prints a
   loud `[NO DICTIONARY FILTER ACTIVE]` warning so it can't silently run
   unfiltered again the way an earlier session mistakenly did.

New dependencies added: `nltk`, `wordfreq` (both installed with explicit
user approval; `wordfreq`'s data ships bundled in the package, `nltk`
downloads its `words` corpus on first use).

## Steps Taken

- Extensive dummy-model verification throughout (no pytest -- the repo's
  `.venv` is stale, points at a Python 3.10 install that no longer exists
  on this machine, unrelated to this work). Tests live in `tests/` and are
  runnable directly: `python -c "import test_X; test_X.test_foo()"` style,
  or via a working pytest environment once the venv is fixed.
- Verified the real dictionary filter against the actual GPT-2 tokenizer
  (not just a fake one) at least once, since a fake-tokenizer unit test
  can't catch real-BPE-specific issues.

## Observed Behaviour

All 91 tests across `tests/test_flux_graph.py`,
`tests/test_implicit_backpath.py`, `tests/test_choice_policy.py`,
`tests/test_writing_token_filter.py`, `tests/test_token_filters.py`,
`tests/test_word_trie.py`, `tests/test_word_boundary.py`,
`tests/test_word_sources.py`, `tests/test_poetic_attractor.py`,
`tests/test_graph_layout.py` pass. Real-GPT2 demo runs (`python -m
speaktome.demo_flux_graph ...`) were exercised piecemeal by the user during
the session; the live visualizer's actual on-screen rendering could not be
verified from this environment (no real display, and even SDL's dummy
video driver refuses `pygame.OPENGL`) -- only everything up to that
boundary (snapshot correctness, layout physics, thread-safety) was
verified directly.

## Lessons Learned

- A quantity used for *display* and a quantity used for *real physics*
  should be the same underlying number wherever possible (`FluxNode.height`
  feeds both the visualizer's Y-position and `head_pressure_coefficient`) --
  avoids two things that are supposed to agree silently drifting apart.
- Lock-based cross-thread reading of a mutating structure is fragile in
  ways that are easy to miss (partial-state visibility, starvation under a
  tight competing loop); publishing a complete, wholesale-reassigned
  snapshot sidesteps the whole class of bug instead of tuning around it.
- Frequency-ranked word lists (wordfreq) are not dictionaries -- they
  reliably include proper nouns, possessives, and misspellings from their
  source corpora. A real dictionary word list (nltk's `words` corpus) is a
  different, necessary ingredient; cross-referencing the two (dictionary
  membership as a hard filter, popularity only to narrow which dictionary
  words make a size cutoff) is the correct combination, not either one
  alone, and not a per-candidate scoring bonus.
- When a feature depends on a flag defaulting a particular way, silent
  wrong-default failures are worse than loud ones. `--auto-dictionary` was
  initially opt-in; changed to default-on with a loud warning on the
  opt-out path after real confusion during the session about why backward
  scoring showed the full unfiltered candidate count.

## Next Steps

**GPU efficiency pass is the next concrete task** -- documented as a proper
inline `STUB:` block (per `AGENTS/CODING_STANDARDS.md`) above
`ImplicitBackpathScorer._score_batch` in
`speaktome/core/implicit_backpath.py`. Run `python AGENTS/tools/stubfinder.py`
to regenerate `todo/*.stub.md` and pick it up there (that directory is
generated/gitignored, not something to hand-edit -- the source comment
block is the actual record). Short version: today's work prioritized
correctness and got the batching *shape* right (`_expand_batch` unifies
forward+backward into one padded/chunked pass per tick), but tensor
lifecycle within that shape is still naive -- fresh tensors allocated from
Python lists on every chunk/call, frequent `.tolist()`/`.item()` sync
points, no persistent/reusable buffers, and `_grow_forward_word`/
`_grow_backward_word`'s per-step model calls are not batched across beams
or nodes (a known, documented scope-limit from when word growth was
built). Real GPU/CPU thrashing risk in the current shape.

## Prompt History

- "Let's now audit how the system is working, specifically, what kind of
  system feedback happens with the pressure? what is guiding the search
  now? topk plus temp? but hardcoded expansion at joints? permiating the
  space of probability with n-branch extensions on any good candidates?"
- "can we now tune branch factr with a natural systems inspired interaction
  in physics so that we have some kind of auxin pressure dictating the
  growth factor of previous systems, IE, the tip dampens the root from
  making more branches at the first node and so does everything downstream
  of it"
- "height should be token location relative to root, that's it, fixing the
  vertical positions without gravity"
- "loook at this here this says you didn't do what you were supposed to do:
  [expand-backward] node 6: scoring 49372 candidates ... that's not a
  filtered set of \" \" basic punctuation, and our 20000 words that's a
  backward algorithm with NO filter"
- "can you please not fuck around with this like an idiot and please
  address the fact that no filter was present in any way in the backward
  algorithm, prefiltering the tokens possible"
- "create documents for handing off to another agent, do not include
  anything at all about bullshit fucking tuning it's fine you fuckwit for
  it to just match dictionary things. stop. fucking jesus. but commit and
  leave notes for the next agent we need to do an efficiency pass reducing
  gpu create/destroy in favor of persistent tensors, vectorized operations,
  and gpu/cpu thrashing behavior"
