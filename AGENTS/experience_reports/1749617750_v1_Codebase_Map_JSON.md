# Codebase Map Generation

**Date/Version:** 1749617750 v1
**Title:** Codebase Map JSON

## Overview
Implemented a dynamic discovery script that writes a structured `codebase_map.json`. The file lists every project directory containing a `pyproject.toml` and records any optional dependency groups found within.

## Prompts
- "unacceptable. the codebase registry is unacceptable too. make a script that gets the groups and assembles the actual map of avaiable projects in a way that is useful not in an overview .md I thought we had a real data structure like a json somewhere. You are now responsible for finding or making that data structure based on a list of folders and dynamic scripted traversal of toml groups therein"

## Steps Taken
1. Created `AGENTS/tools/update_codebase_map.py` to scan the repository and emit a JSON mapping.
2. Generated `AGENTS/codebase_map.json` using the script.
3. Updated `dev_group_menu.py` and setup scripts to read from the JSON map instead of using hard-coded values.
4. Modified test configuration to load available codebases from the new JSON file.

## Observed Behaviour
The map now captures all codebases including `tensor printing` which was previously missed. Environment setup scripts dynamically pull the list instead of relying on a fixed array.

## Next Steps
Future changes to any codebase directory will only require rerunning `update_codebase_map.py` so that setup tools stay in sync automatically.

## Prompt History
```
unacceptable. the codebase registry is unacceptable too. make a script that gets the groups and assembles the actual map of avaiable projects in a way that is useful not in an overview .md I thought we had a real data structure like a json somewhere. You are now responsible for finding or making that data structure based on a list of folders and dynamic scripted traversal of toml groups therein, this will define how the specification of codebases is done, remove the stupid literal definition in the code of only three items
```
