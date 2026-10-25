# Generated Page Resource Fallback

**Date:** 1785670248
**Title:** Generated page resource fallback

## Command

From `C:\dev\Powershell\turing`:

```text
py -3.11 -m pytest tests\test_wasm_html_shell.py tests\test_site_bundle.py -q
```

## Log

```text
41 passed, 1 warning in 4.02s
```

The generated HTML resolver was changed to produce an ordered list of resource
locations. It recognizes the repository root from a nested `/site/` path for
both `file:` and hosted pages, retains the declared resource route, adds the
configured loopback Go server, and retries resource downloads across candidates.

## Prompt History

> "manually opening the file, using the go server, or on github, in all cases the link to the bundle is different. I need the generated pages to be able to search a list of locations, because I'm tired of the page not working in certain contexts"

The workspace `AGENTS.md` directed agents to read the SpeakToMe behavior guides,
whose guestbook instructions require this report.
