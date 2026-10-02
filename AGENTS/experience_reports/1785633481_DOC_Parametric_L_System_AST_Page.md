# Parametric L-System AST Page

**Date:** 2026-08-01
**Title:** Generated the space-filler inspection page from its source AST

## Activity

Corrected an earlier misunderstanding: the requested page was not a bespoke
visual interface. It was the existing structural AST inspection shell applied
to `turing/src/common/parametric_l_system.py`.

Added `build_source_inspection_bundle` to the generic site-bundle emitter, so
file/class/method AST inspections use the same content-versioned manifest and
gallery layout as numeric compilations. `build_site_page.py` now automatically
selects structural inspection when a source has no public top-level numeric
entrypoint.

Generated and published:

```text
site/programs/parametric-l-system/versions/v1-850c95df6235bf73/
  bundle.json
  index.html
  source/python_source/parametric_l_system.py
```

The generated page includes the module ProcessGraph, `ParametricLSystem` class
map, instance fields, method graph identities, class-navigation LUT, semantic
methods, and original source. No page-specific visual implementation was added.

## Verification

```text
python -m pytest tests/test_parametric_l_system.py tests/test_site_bundle.py tests/test_wasm_html_shell.py -q
49 passed, 1 warning in 4.36s

go test .
ok github.com/sangderenard/nogodsnomasters (cached)
```

## Prompt History

> can you or did you already make a page for the space filler

> no, what? why would there be presets. I'm not asking you to hand write a page I'm asking you to generate it off the ast

> proceed

## Next Steps

None.
