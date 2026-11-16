# MCP HTTPS review continuation

## Work

The live Turing and SpeakToMe checkouts held `codex/fix-mcp-internal-https`, but the review gate had no accepted review for either HTTPS change. The updater skipped both because they were off their default branches. I copied only the HTTPS repair into fresh centrally managed checkouts based on current remote defaults: Turing passes its local CA to the OAuth MCP bridge, and SpeakToMe's internal REST client requires loopback HTTPS with certificate verification. The original feature branch and its separate guest session capacity commit remain untouched.

## Verification

- Turing `tests/test_meowdoku_oauth_mcp.py` passed (1 test) with `SPEAKTOME_REPO` set to the matching SpeakToMe checkout. It exercises a TLS listener and guest MCP calls.
- The direct SpeakToMe pytest invocation did not run its tests: its `conftest.py` attempted automatic environment setup, which failed while installing the editable package, then skipped collection. This is an environment limit, not a test result for the code.

## Next Steps

Submit both review commits and have the review gate land them. Return the live checkouts to their default branches, run `Update-LiveCheckouts.ps1 -ExportWeb`, restart Apache, and verify the public MCP endpoint.

## Prompt History

> use the mcp for checking things out do not run git

> it was my impression this was fixed, but there is actually an administrator script I've lost sight of ive been just restarting apache but there's a script that also puts everything on main

> the script doesn't try to ff/merge? so I guess you'll need to be merging your work there
