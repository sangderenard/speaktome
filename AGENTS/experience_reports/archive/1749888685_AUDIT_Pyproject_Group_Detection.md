# Template Audit Report

**Date:** 1749888685
**Title:** Pyproject-Based Header Recognition Review

## Scope
Investigate existing utilities that read `pyproject.toml` to determine codebase names and optional dependency groups for header management.

## Methodology
- Searched the repository for scripts referencing `pyproject` and optional groups.
- Inspected the `AGENTS/tools` directory for relevant utilities.
- Reviewed existing documentation and previous experience reports.

## Detailed Observations
- `AGENTS/tools/update_codebase_map.py` builds a JSON map of codebases and dependency groups extracted from each `pyproject.toml` file. The script discovers project directories and parses the optional dependencies using `tomllib`【F:AGENTS/tools/update_codebase_map.py†L1-L66】.
- `AGENTS/tools/dev_group_menu.py` provides an interactive menu to select codebases and groups. It loads the map from `update_codebase_map.py` and can install selected groups automatically【F:AGENTS/tools/dev_group_menu.py†L1-L122】【F:AGENTS/tools/dev_group_menu.py†L160-L274】.
- `AGENTS/tools/auto_env_setup.py` parses the root `pyproject.toml` to list available groups and runs the setup scripts accordingly【F:AGENTS/tools/auto_env_setup.py†L30-L76】【F:AGENTS/tools/auto_env_setup.py†L77-L115】.
- A minimal header parser exists in `AGENTS/tools/dynamic_header_recognition.py`, but its functions `parse_header` and `compare_trees` are stubs with TODO comments【F:AGENTS/tools/dynamic_header_recognition.py†L1-L75】.

## Analysis
The repository already includes tooling to map codebases and optional groups from `pyproject.toml` files. These utilities feed into interactive setup (`dev_group_menu.py`) and automated environment initialization (`auto_env_setup.py`). However, the dynamic header recognition script does not yet integrate with this mapping to resolve missing groups when header imports fail. Implementing this connection would allow each script to identify its codebase and required groups at runtime based on the generated map.

## Recommendations
1. Extend `dynamic_header_recognition.py` to parse `pyproject.toml` and correlate missing imports with their respective groups.
2. Expose a simple API from `update_codebase_map.py` so header utilities can query available groups programmatically.
3. Document a workflow for scripts to use this information during header import failures to suggest the correct group installation.

## Prompt History
- System: "You are ChatGPT, a large language model trained by OpenAI."
- User: "okay, the task for us right now, then, is to look around for scripts we might have already created... please generate an audit experience report without implementing any changes in the commit other than that report"
