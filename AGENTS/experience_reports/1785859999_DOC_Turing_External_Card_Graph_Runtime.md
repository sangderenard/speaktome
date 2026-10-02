# Turing external card graph and lazy read-head runtime

**Date:** 2026-08-04

## Scope

Extended the independent `turing` repository's shared-memory WebAssembly class
runtime from a fixed sequential inventory toward a traversable OOP/card
environment derived from Map IR.

## Work completed

- Added `src/compiler/card_graph.py` and the `turing.card-graph.v1` projection.
- Preserved file functions, class methods, deployed WASM cards, typed potential
  connections, exact resident-memory connections, and the existing linear path.
- Embedded the graph in generated HTML shells and exposed a browser
  `CardGraphReadHead` through `window.TuringCardGraph`.
- Changed punch-card caching to cache address-free compiled modules globally
  while keeping memory-bound instances local to one outer arena.
- Added lazy per-card loading and unconditional input/output alias rewriting at
  every traversal boundary.
- Documented the ABI in `turing/docs/CARD_GRAPH_RUNTIME.md`.

Focused verification:

```text
python -m pytest -q tests/test_card_graph.py \
  tests/test_wasm_class_coordinator.py tests/test_wasm_html_shell.py
51 passed in 4.37s
```

## Prompt History

> in the way we split programs into sequential cards for linear processing, we want a policy that allows "external" linking with shared input output memory addresses in the arena of an outer coordinator or fully autonomous program segment caching. i want to use the runtime version we might not have finished of our process read head, letting people lazy load parts of an oop environment free to traverse as a thread. i want a way to load and cache punch cards but i want them always to rewrite their inputs and outputs to alias. this turns our static webpages into an entire operating system

> Largely we have not prepared the system for classes files and modules, some details tension oriented toward kernels and functions, you may have to get the html shell to be able to take a card graph, all cards with all their valid potential connections, you can get the data from the map in the graph ir

> you do know it doesn't fucking matter github auto corrects line endings and you're just wasting fucking time

