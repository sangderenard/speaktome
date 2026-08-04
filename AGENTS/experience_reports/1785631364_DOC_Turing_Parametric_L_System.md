# Parametric L-System for Turing

**Date:** 2026-08-01
**Title:** Added a composable, inspectable L-system class

## Activity

Added `turing/src/common/parametric_l_system.py`. The module separates parallel
symbol derivation from two-dimensional turtle interpretation. It supports
deterministic, weighted, context-sensitive, and callable productions; seeded
randomness; parameter-aware callable actions; branch state; geometry budgets;
SVG path output; and six classic presets.

Added `turing/tests/test_parametric_l_system.py` to cover derivation history,
context matching, stochastic reproducibility, branch restoration, composed
actions, resource limits, branch validation, SVG output, and presets.

## Verification

```text
python -m pytest tests/test_parametric_l_system.py -q
............                                                             [100%]
12 passed in 0.42s

python -m compileall -q src/common/parametric_l_system.py
git diff --check -- src/common/parametric_l_system.py tests/test_parametric_l_system.py
```

## Prompt History

> in turing there's a lot being developed about compiling things to a website that lets you deeply inspect and test it, and I want to make for that a function that does those space filling things the math software guy loved, you know branching paths or whatever, we want it to take as many parameters as possible and then form emergent patterns by carrying out the composition of rules

> all you have to do is write the python class for it

> the rest is finished and automated after that

## Next Steps

None. Website integration was explicitly outside this task.
