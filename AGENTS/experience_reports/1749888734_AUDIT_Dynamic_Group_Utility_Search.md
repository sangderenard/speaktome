# Audit Report: Dynamic Group Utility Search

**Date:** 1749888734
**Title:** Search for dynamic group selection utilities

## Scope
Investigate existing scripts that parse `pyproject.toml` files to determine optional dependency groups or codebase membership. The goal is to assess the current state of automation that might aid header-based environment setup.

## Methodology
- Searched the repository for references to group detection or `pyproject` parsing using `grep`.
- Reviewed the relevant utilities under `AGENTS/tools`.
- Inspected historical discussions in `AGENTS/messages/outbox` for past design notes.

## Detailed Observations
- `dev_group_menu.py` includes functions that read optional dependency groups from a project's `pyproject.toml` and build a mapping of codebases to groups. Key lines show the extraction process:
  ```python
  def extract_group_packages(toml_path: Path) -> dict[str, list[str]]:
      """Return optional dependency mapping from ``pyproject.toml``."""
      try:
          data = tomllib.loads(toml_path.read_text())
      except Exception:
          return {}
      return data.get("project", {}).get("optional-dependencies", {})
  ```
  ```python
  def build_codebase_groups() -> tuple[dict[str, dict[str, list[str]]], dict[str, Path]]:
      """Return mapping of codebase name to group->packages and name->path."""
      mapping: dict[str, dict[str, list[str]]] = {}
      paths: dict[str, Path] = {}
      for cb_path in discover_codebases(REGISTRY):
          groups: dict[str, list[str]] = {}
          for toml_file in cb_path.rglob("pyproject.toml"):
              groups = extract_group_packages(toml_file)
              if groups:
                  break
          mapping[cb_path.name] = groups
          paths[cb_path.name] = cb_path
      return mapping, paths
  ```
  【F:AGENTS/tools/dev_group_menu.py†L128-L149】
- `update_codebase_map.py` generates a JSON map capturing each codebase path and its optional groups. Relevant snippet:
  ```python
  def discover_codebases(root: Path) -> list[Path]:
      """Return directories containing a top-level `pyproject.toml`."""
      returnvals = [p.parent for p in root.glob('*/pyproject.toml')]
      # Always include AGENTS/tools as a path:
      special_path = root / 'AGENTS' / 'tools'
      if special_path.is_dir():
          returnvals.append(special_path)
      return returnvals
  ```
  ```python
  def extract_groups(toml_path: Path) -> dict[str, list[str]]:
      """Read optional dependency mapping from `pyproject.toml`."""
      try:
          data = tomllib.loads(toml_path.read_text())
      except Exception:
          return {}
      groups = data.get('project', {}).get('optional-dependencies')
      if groups:
          return groups
  ```
  【F:AGENTS/tools/update_codebase_map.py†L26-L45】
- `dynamic_header_recognition.py` defines a `HeaderNode` tree and stub parser for future header analysis:
  ```python
  HEADER_START_REGEX = re.compile(r"^# --- BEGIN HEADER ---$", re.MULTILINE)
  HEADER_END_REGEX = re.compile(r"^# --- END HEADER ---$", re.MULTILINE)
  ...
  class HeaderNode:
      """Node in a tree representation of the header."""
      def __init__(self, name: str, content: str = "", children: Optional[list["HeaderNode"]] = None) -> None:
          self.name = name
          self.content = content
          self.children: list[HeaderNode] = children or []
  ```
  【F:AGENTS/tools/dynamic_header_recognition.py†L58-L73】
- Historical design notes in `AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md` describe the plan for dynamic codebase and group discovery.
  【F:AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md†L1-L12】

## Analysis
These utilities collectively support automated discovery of optional dependency groups. `dev_group_menu.py` already reads `pyproject.toml` files to populate a mapping used during environment setup. `update_codebase_map.py` can export this mapping to JSON, enabling other tools to locate codebases and their groups without hardcoding paths. The dynamic header parser is still a stub but could ultimately analyze headers to determine required groups by examining import blocks.

## Recommendations
- Integrate `update_codebase_map.py` with header validation tools so scripts can automatically determine which codebase they belong to.
- Expand `dynamic_header_recognition.py` to parse import statements and match them against the codebase map, enabling group hints for missing dependencies.
- Document usage of these utilities in `AGENTS/headers.md` to encourage consistent adoption across new scripts.

## Prompt History
```
okay, the task for us right now, then, is to look around for scripts we might have already created a while ago, there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material, it doesn't get pushed and breaks repos. anyhow I think at some point we were prepping a suite of utilities to integrate with other header utilities and a great aspiration in that is what would fix the present moment, a script capable of automatically recognizing from the toml of the project root in the monorepo which codebase and what groups from it are necessary for the present header error. It was to be an integral part of the header template that each script be empowered to find out what codebase it's in and what groups it needs for what level of function, so we need probably a way to standardize wrapping extra functionality from problematic or hardware specific optional imports. This is a complex issue so please generate an audit experience report without implementing any changes in the commit other than that report
```
