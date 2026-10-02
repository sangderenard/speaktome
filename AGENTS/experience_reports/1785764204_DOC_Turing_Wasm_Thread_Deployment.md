# Turing Wasm Thread Deployment

**Date:** 2026-08-03
**Title:** Parallel Control tags dispatched as tiled Web Worker deployments

## Activity

The adjacent `turing` repository now projects lexical `ParallelDeployment`
blocks and compatible durable deployment-table regions into a browser-facing
Wasm thread-plan ABI. The HTML execution shell divides each independent lane
into eight-element-aligned tiles, bounds simultaneous workers using the
browser's hardware-concurrency report, executes ordinary compiled Wasm region
kernels in Blob Web Workers, and commits their disjoint outputs only after an
awaited barrier Join.

The implementation deliberately uses copied worker-local memories instead of
`SharedArrayBuffer`, so the generated pages do not require COOP/COEP headers
and remain usable on GitHub Pages. Unsupported control, colliding output
slots, unavailable Workers, or worker failures retain/replay the translated
serial Wasm coordinator.

The first fluid-page bake exposed that the published thread plan was not on
the normal URL-backed module execution path. The corrected runner now assigns
each element tile a complete dependency-preserving vertical schedule and
crosses one Join barrier per tick. It does not copy intermediate fields at
each of the 69 topology waves. A bounded pool of persistent workers caches
worker-local memories and module instances. The main thread fetches and
compiles each immutable Wasm card once, then structured-clones the compiled
modules into the pool; only external inputs and public final outputs cross the
worker boundary per tick.

Whole-program source records also gained configured specialization. A
`TURING_PAGE["constants"]` map replaces named public parameters with Python
literals before ProcessGraph construction and topology reduction. The source
record remains complete, while the AOT record identifies itself as
`configured`, removes specialized parameters from its runtime tensor ABI, and
records the constant map in the immutable bundle manifest. The fluid ecology
now specializes `dt` to `0.025` through that mechanism.

## Verification

The focused coordinator and HTML suites passed 44 tests. A headless Chrome
test instantiated a real compiled Wasm multiplication kernel inside the Blob
worker and verified that `[1, 2, 3]` joined as `[2, 4, 6]`. The surrounding
Wasm binary, class module, and site bundle suites passed 63 tests, and Node's
parser accepted the emitted runtime JavaScript.

The fresh fluid bundle is `v1-297bc12f93c86250`. Its recorded topology has
946 source operations, 155 reduced regions, 69 dependency waves, 50 parallel
waves, maximum operation width 5, and 78 vertically fused regions. Headless
Chrome produced matching non-black threaded and forced-serial frames. The
sampled threaded tick used seven workers and took 173.7 ms; the serial
coordinator took 409.8 ms, a 2.36x speedup and 57.6% lower execution time.
The isolated end-to-end Python-to-WebGL browser regression passed, followed
by 93 focused compiler/shell tests.

### Correction: collective reductions cannot use whole-program element tiles

Visual inspection exposed a semantic error in that benchmark. The fluid
program contains whole-invocation `sum` reductions. Running the complete
graph separately in each element tile changed those into per-tile sums and
created multiple independent entity-state consensuses. The apparent 2.36x
speedup therefore compared different programs and is retracted.

The compiler now marks fused programs and regions with an extent effect.
`pointwise` graphs remain eligible for complete vertical element tiles;
`collective` graphs retain the whole-extent Wasm coordinator until a genuine
partial-reduction Join is implemented. This is enforced from compiler
metadata rather than JavaScript operator-name inference.

The configured AOT boundary also distinguishes immutable constants from
mutable parameters. Site contracts derive mutable parameters from
`state_feedback`, reject static/mutable overlap, and pass the same guard into
`compile_ast_aot`. The fluid page supplies full 256x176 tensor literals for
its static coordinate and rest-surface parameters, scalar literals for `dt`
and fixed audio inputs, and leaves every feedback channel mutable.

## Next Steps

No required follow-up remains for this deployment pass.

## Prompt History

> "now in web assembly, interpret the parallel tags as a multi thread deploy and join please, using some mechanism to add apropriate amounts of tiles at once"

> "can you do a fresh generation of the fluidic sim and see if the threads improve performance on parallizeable operation strings, and do make sure the topology of parallelization vertically fuses"

> "oh also institute constants maps for full program record, that redefine any of the parameters to constants before reduction, is like, instead of numeric vs full we can have configured"

> "I'm confused by your comments, I want you to proceed, but I don't understand what you mean partial reduction executor, makes me nervous the website is like, not using binaries and we are just kinda goofing around still not running a true assembly binary or something"

> "mark feedback parameters please in contracts so they are impossible to make static"

> "there is supposed to be an existing means of marking parameters as static for a compile, you need to look harder, and make it if it's not there"

> "I mean when you compile you put in a dictionary and every entry you put in is baked in as a static value you provided"
