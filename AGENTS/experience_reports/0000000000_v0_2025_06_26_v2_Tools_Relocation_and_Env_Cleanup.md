# Tools Relocation and Env Cleanup

## Prompt History
```
clean the environmental setup script so there isn't any meta conversation about the repo reset, make it helpful and professional. in the dev setup use head and tail to present each document as a preview block not an entire document. make a script for producing these kinds of top and bottom summaries and put it in tools. look for python that is not in speaktome, agents/tools, or the archive. find any references to them in agent documents or tools or scripts, make a thorough relocation to tools, consolidating scripts and correcting references
```

## Steps Taken
1. Implemented `AGENTS/tools/preview_doc.py` for head/tail previews.
2. Updated `setup_env.sh` to remove purge commentary and add a concise summary.
3. Modified `setup_env_dev.sh` to use `preview_doc.py` and new tool paths.
4. Moved various utility scripts into `AGENTS/tools` and adjusted imports and audit lists.
5. Ran the test suite via `python testing/test_hub.py`.

## Observed Behaviour
Scripts run smoothly and tests continue to pass after the relocation.

## Next Steps
Monitor for any broken references in remaining documentation.
