# Nogodsnomasters Local Compiler-Page Publisher

**Date:** 2026-08-01
**Title:** Added a loopback Go server and versioned Turing page bundles

## Activity

Added a standard Turing program-bundle emitter under
`turing/src/compiler/site_bundle.py` and its `build_site_page.py` command.
Each compiled Python source is content-versioned beneath
`site/programs/<slug>/versions/<version>/`, with a manifest, generated page,
original and emitted sources, external WASM, optional SymPy model, compiler
log, and a hash/size inventory.

Added the root `server.go`, which serves the published homepage, accepts only
loopback Python-generation requests, invokes the Turing builder without a
shell, and infers its gallery live by walking valid bundle manifests. The HTML
shell now contains the localhost server control, trusted Python upload form,
gallery, and bundle-aware resource resolution. The real homepage and versioned
homepage artifacts were regenerated and synchronized to the root publication.

## Verification

```text
go test .
ok github.com/sangderenard/nogodsnomasters

go vet .

python -m pytest tests/test_site_bundle.py tests/test_wasm_html_shell.py -q
36 passed, 1 warning in 3.53s

python build_homepage.py
9/9 backends emitted this program
wrote index.html and 3 coordinated class variants plus one contiguous module
```

An isolated live server check uploaded `docs/site_bundle_example.py` through
`POST /api/generate`, discovered the resulting 13-artifact bundle through
`GET /api/gallery`, fetched its generated HTML (HTTP 200), and fetched its
external WASM as `application/wasm` (HTTP 200). The temporary server and bundle
tree were removed afterward.

## Prompt History

> can you give the nogodnomasters a little go server that hosts the main page locally and also unlocks that page's ability to take python files through the interface and issue a backend command to generate it's page, and with a gallery that uses the folder tree of the site data to infer what has been prepared to make available to show, make sure when sources are processed like this, there is a standard way they are packed in folders with all their associated material and versioning, and that the htmlshell has a server address box defaulting to localhost, and looks for resources in the right place, and the site emission will no longer only be for the home page, but will put anything generated where it belongs

## Next Steps

None required for the local workflow.
