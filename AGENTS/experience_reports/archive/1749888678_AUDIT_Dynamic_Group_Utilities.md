# Audit Report: Existing Dynamic Group and Header Recognition Utilities

**Date:** 1749888678
**Title:** Dynamic Codebase/Group Discovery Search

## Scope
Audit of existing utilities for automatically deriving codebase and optional dependency groups from `pyproject.toml` files, and review of dynamic header recognition scripts.

## Methodology
- Searched the repository for terms related to dynamic header recognition and `pyproject`.
- Inspected modules under `AGENTS/tools` for functions that parse `pyproject.toml` files.
- Reviewed previous documentation and proposals.
- Examined test files referencing these utilities.

## Detailed Observations
- `AGENTS/tools/dev_group_menu.py` implements `build_codebase_groups` which walks registered codebases and extracts optional dependency groups from discovered `pyproject.toml` files using `extract_group_packages`.
- The same module offers an interactive menu to select codebases and groups for installation.
- Proposal `AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md` outlines the design for automatic discovery and provides example code.
- `AGENTS/tools/dynamic_header_recognition.py` defines `HeaderNode` and stub functions such as `parse_header`; tests under `AGENTS/tests/headers` duplicate this skeleton.

## Analysis
Dynamic group detection is already partially implemented through `dev_group_menu.py` which reads optional dependency groups from each codebase's `pyproject.toml`. Documentation in the proposal matches the current implementation. However, the header recognition script remains a minimal stub, indicating future expansion is needed to tie header failures to missing groups automatically.

## Recommendations
- Complete the parser in `dynamic_header_recognition.py` so scripts can map missing imports to dependency groups.
- Connect `build_codebase_groups` with the header utilities once parsing is available.
- Continue referencing the proposal to maintain alignment with the intended design.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago ... please generate an audit experience report without implementing any changes in the commit other than that report"
- "always check the files in the repo ecosystem for your benefit. ... Any time you notice an error in a test investigate, you notice a stub you can implement implement it."
