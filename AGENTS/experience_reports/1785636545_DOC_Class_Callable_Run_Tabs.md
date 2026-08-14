# Class Callable Run Tabs

**Date:** 2026-08-01
**Title:** Generated class-grouped callable run systems for source inspection pages

## Activity

Traced the parametric L-system inspection bundle from AST ingestion through
ProcessGraph reduction, class navigation, shell emission, gallery discovery,
and browser rendering.

The first data-loss boundary was in `build_source_inspection_page`: it built
the class-navigation table before topological reduction populated the shared
function table. Method names survived, but every `function_reference` was
therefore `null`. Reduction now runs before navigation is assembled; the 20
explicit methods in `TurtleTrace` and `ParametricLSystem` retain graph-backed
references.

Source inspection pages now enumerate callables in the requested order:

1. classes in source order, with methods grouped under each owner;
2. module-level functions after the class groups.

Every callable gets its own signature inputs, Run control, lowering status,
and nested generated inspection page. The shell uses two levels of hidden
`div` tabs: outer owner tabs select a class or module-functions group, and
inner tabs select one method/function run system. The large raw map IR is also
hidden behind a diagnostic tab so it does not extend the initial page.

Published the final L-system bundle at:

```text
site/programs/parametric-l-system/versions/v1-9f5e7fe19eb449bc/
```

It contains 20 callable systems and 42 inventoried artifacts. The loopback Go
server reports it as the latest gallery version.

## Verification

```text
python -m pytest tests\test_site_bundle.py tests\test_wasm_html_shell.py tests\test_parametric_l_system.py -q
52 passed, 1 warning in 6.55s
```

Browser accessibility inspection confirmed the outer class tabs and populated
method references. The remaining execution gap is explicit: structural
inspection bundles do not yet emit executable backend artifacts for arbitrary
Python object/string methods, so their Run controls remain disabled with that
reason rather than silently failing.

## Prompt History

> can you supervise the process of making a page from the l system parametric class and see where we're losing data that's preventing it from running

> well, okay, when a class is being made into a page, or is in a page, it should have an inputs for every class method, making them inspectable, able to have pages generated for those methods too

> like a run system for every method groups by class should happen, then the ones outside classes

> you decided to ignore that I said each method needs a run section in the shell

> use divs to make hidden tabs so the methods in a source being examined don't stretch the page vertically forever, they get packed into tabs

> stack the classes and all that content into even higher level tabs so the classes don't run down the page forever

## Next Steps

None recorded as a guestbook task. Runtime artifact emission remains an
architectural boundary described above, not an unowned Speaktome todo.
