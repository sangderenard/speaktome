# Proposal: Dynamic Codebase/Group Discovery System

## Context
Currently `AGENTS/tools/dev_group_menu.py` uses a hardcoded dictionary of codebases and their groups. This creates maintenance overhead and risks getting out of sync with actual codebase configurations.

## Proposed Solution
Implement dynamic codebase and group discovery by:

1. Reading registered codebases from `AGENTS/CODEBASE_REGISTRY.md`
2. Recursively searching each codebase directory for `pyproject.toml` files
3. Extracting optional dependency groups from these TOML files

### Example Implementation:

```python
from pathlib import Path
import tomli

def discover_codebases(registry_path: Path) -> list[str]:
    """Extract codebase paths from CODEBASE_REGISTRY.md"""
    # Parse markdown to find registered codebase paths
    # Return list of valid directory paths

def extract_groups(toml_path: Path) -> list[str]:
    """Extract optional dependency groups from pyproject.toml"""
    with open(toml_path, "rb") as f:
        data = tomli.load(f)
    
    # Get optional dependencies section
    optional = data.get("project", {}).get("optional-dependencies", {})
    return list(optional.keys())

def build_codebase_groups() -> dict[str, list[str]]:
    """Dynamically build codebase->groups mapping"""
    codebases = {}
    
    for codebase in discover_codebases(Path("AGENTS/CODEBASE_REGISTRY.md")):
        codebase_path = Path(codebase)
        if not codebase_path.exists():
            continue
            
        # Search recursively for pyproject.toml
        for toml_file in codebase_path.rglob("pyproject.toml"):
            groups = extract_groups(toml_file)
            codebases[codebase] = groups
            break  # Take first pyproject.toml found
            
    return codebases
```

### Benefits

1. **Self-Maintaining**: Groups automatically stay in sync with pyproject.toml files
2. **Extensible**: New codebases are automatically detected when registered
3. **Accurate**: Groups reflect actual available options in each codebase
4. **DRY**: No duplicate maintenance of group lists

### Implementation Steps

1. Update `dev_group_menu.py` to use dynamic discovery
2. Modify setup scripts to use discovered groups
3. Add validation that discovered groups exist in their pyproject.toml
4. Update documentation to reflect dynamic discovery

## Risks and Mitigations

1. **Performance**: Cache discovery results if needed
2. **Missing files**: Gracefully handle missing TOML files or registry
3. **Backwards compatibility**: Keep hardcoded fallbacks initially

## Next Steps

1. Create prototype implementation in AGENTS/tools
2. Run validation across all registered codebases
3. Update setup scripts to use new system
4. Document discovery system in AGENTS/CODING_STANDARDS.md

Would you review this proposal and provide feedback on potential challenges or improvements?

Best regards,
Assistant

---
Note: This implements the dynamic group discovery we discussed while maintaining compatibility with existing setup scripts.