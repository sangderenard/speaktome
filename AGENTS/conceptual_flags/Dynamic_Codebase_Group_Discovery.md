# 🚩 Conceptual Flag: Dynamic Codebase and Group Discovery

**Authors:** GPT-4o, Codex

**Date:** 2025-06-07

**Version:** v1.0.0

## Conceptual Innovation Description

Automate detection of available codebases and their optional dependency groups by
parsing `AGENTS/CODEBASE_REGISTRY.md` and scanning each codebase for
`pyproject.toml`. This removes hardcoded mappings and keeps setup menus
synchronized with actual project configuration.

## Relevant Files and Components

- `AGENTS/tools/dev_group_menu.py`
- `AGENTS/CODEBASE_REGISTRY.md`
- `AGENTS/tools/update_codebase_map.py`
- `AGENTS/codebase_map.json`

## Implementation and Usage Guidance

Run `python -m AGENTS.tools.dev_group_menu --list` to view discovered codebases
and groups. Update `CODEBASE_REGISTRY.md` when new repositories are added. The
menu script will automatically rebuild its mapping or use `update_codebase_map.py`
to cache results.

## Historical Context

Originally proposed in
`AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md`
and implemented via automated scanning documented in
`AGENTS/experience_reports/archive/1749496649_v1_Dev_Group_Menu_Dynamic_Discovery.md`.

---

**License:**
This conceptual innovation is contributed under the MIT License located at the
project root.
