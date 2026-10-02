# Gallery Publish Root

**Date:** 2026-08-01
**Title:** Fixed gallery generators to publish outside the Turing repository

## Activity

Traced the duplicated generated-site trees at the workspace root and in
`turing/`. The copies were byte-identical except for one legacy demo. No HTML
or generated artifact was changed during this task.

Changed the Turing-side generator entry points so their default destination is
the parent `nogodsnomasters` workspace root. Gallery bundle APIs now reject the
Turing repository itself as a publish root, preventing future page generation
from silently splitting output between the two repositories. Updated the
rebuild documentation and added command-default regression coverage.

## Verification

```text
python -m pytest tests\test_site_bundle.py tests\test_site_build_commands.py -q
10 passed, 1 warning in 4.75s
```

The warning is the existing Python 3.11 `cffi` use of deprecated `imp`.

## Prompt History

> content for the gallery of program pages generated is scattered between turing and the root, it's supposed to generate into the root, not turing

> you had better not have edited any html, you had better have fixed the generator for pages

> how do I start the go server

## Next Steps

None.
