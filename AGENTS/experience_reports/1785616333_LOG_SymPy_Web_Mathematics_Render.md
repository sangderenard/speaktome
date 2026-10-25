# SymPy Web Mathematics Render Log

**Date:** 1785616333
**Title:** Existing reduced-ProcessGraph SymPy target rendered in the WASM shell

## Command

From `C:\dev\Powershell\turing`:

```text
python -m pytest tests/test_wasm_html_shell.py tests/test_sympy_math_renderer.py tests/test_wasm_class_modules.py tests/test_wasm_class_coordinator.py tests/test_wasm_binary.py tests/test_machine_targets_wasm.py tests/test_process_graph_shell.py -q
python build_homepage.py
```

## Log

```text
101 passed, 1 warning in 7.44s

8/8 backends emitted this program
wrote C:\dev\Powershell\turing\index.html (5855 KB)
wrote 3 coordinated class variants and one contiguous module

SymPy process model:
3096 live symbolic nodes
3252 relations, including 160 constraints
3 public output equations
lazy JSON/MathML inventory
```

The integration selects `process_graph_to_sympy_relations` after the same
normalization used by the language backends. It adds presentation MathML and
semantic depiction metadata only; it does not add or alter symbolic translation
rules. The human-facing page classifies the program as a numeric map, Boolean
predicate, state transition, or implicit relation. It does not render the SSA
inventory as thousands of equations. The exact 3,252-clause SymPy model is not
loaded until the visitor explicitly clicks its download button. SymPy's named
`&InvisibleTimes;` entity is resolved to Unicode before XML safety parsing; all
3,252 generated fragments passed XML parsing.

The process graph is now one unique-ID `div` per node in native CSS grid
schedule coordinates. There is no graph canvas, graph shader, edge renderer,
CSS transition, keyframe, or CSS animation. Each completed WebAssembly timing
tick directly writes the targeted div's opacity, scale, and blur custom
properties. Phosphor deposits are normalized against rolling p95 microseconds
per node, integrated with exponential decay, and reported alongside raw live
median/p95 timing statistics. Both staged method cards and the contiguous
whole-program execution feed this same profile display.

## Prompt History

> stop, yyou are trying to reinvent things, another agent concurrent to your work already made the path to full sympy symbolic expression

> no, you aren't getting it again, I'm telling you you are doing it, but all the tools to do it are done. all you have to do is pick sympy as a language to reduce to and catch that and direct it to put the process graph of the reduced version into sympy using ssa to sympy or whatever, you're not being asked to make the translation, you'r ebeing told to use it to make it randered

> remove the entire graph canvas, do not have any shaders made, put divs on the page and use css with unique per div id

> css should not be performing animations at all, the assembly merely needs to set the css values and let normal document rendering read them each tick

> we need to figure out how to normalize the intensity of the program. possibly by just normalizing the intensity of the photphor effects XD
