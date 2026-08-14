# File Scope And Gallery Versions

**Date:** 2026-08-01
**Title:** Added file-scope inspection, runnable outer functions, and collapsed gallery versions

## Activity

Changed source-inspection shells so the module/file is a first-class owner.
The selected outer tab is now `file scope`; its first nested tab is the first
module-level function, while a sibling `symbols` tab lists imports, constants,
type aliases, annotated bindings, and other assignments around the classes.
Class owners follow as peer tabs.

Published module-level functions now retain the URL of their complete Python
source and receive an enabled Run control. The loopback Go server exposes a
local-only `/api/run` endpoint which validates that URL against a published
bundle, invokes only a module-level function through `run_site_callable.py`,
and returns typed JSON. Byte-plane tuples are rendered as RGB canvases by the
inspection shell. Class methods remain structurally separate because they
require an object instance.

The gallery shell now groups the flat bundle inventory by program slug. One
card is shown per program, the server's `latest` item is selected by default,
and a native version dropdown changes the card link and artifact details.

Regenerated the workspace homepage and published the L-system bundle at:

```text
site/programs/parametric-l-system/versions/v1-c1d5419ae92c1262/
```

It contains 21 callable systems. The restarted Go server reports that version
as newest among six L-system versions.

## Verification

```text
go test ./...
ok github.com/sangderenard/nogodsnomasters

python -m pytest tests\test_wasm_html_shell.py tests\test_site_bundle.py tests\test_parametric_l_system.py -q
53 passed, 1 warning in 4.23s
```

An end-to-end loopback request invoked the published `render` function with a
64 by 64 domain and received three RGB byte planes of 4,096 bytes each.

## Prompt History

> generate the page again please, and when the gallery items are listed, can you make it collapse items to their newest version and put a dropdown on the version so they can be selected from, but always default to newest

> now, I had you put the method outside the class, now we need to talk about code page level or I don't know what we'll call it, file level, outer functions - you know the way we have all those classes on tabs - we also need the open root level methods listed there, even though they're not classes, as selectable and thus runnable, nothing runnable is there hen we open the page so, like, let's get the file scope figured out now please, we can just give it another shell but we already do multiple classes, this just means putting global constants, symbolics, methods, the things around the classes

## Next Steps

None recorded as a guestbook task.
