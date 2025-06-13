"""
Generates a dignified credits list from agent JSON files in AGENTS/users/.
Most recent version of each named agent is used.
"""
from __future__ import annotations

try:
    import json
    from pathlib import Path
    from typing import Dict
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

def generate_credits(users_dir: Path | None = None) -> str:
    """Load all agent JSONs and create a formatted contributor list."""
    if users_dir is None:
        users_dir = Path(__file__).parent.parent / "users"
    agents: Dict[str, dict] = {}
    
    # Load all JSON files, later ones overwrite earlier ones with same name
    for json_path in sorted(users_dir.glob("*.json")):
        content = json_path.read_text(encoding="utf-8")
        if content.startswith("version https://git-lfs.github.com/spec/v1"):
            print(f"Skipping LFS pointer: {json_path.name}")
            continue
        data = json.loads(content)
        agents[data["name"]] = data
            
    # Generate formatted output
    output = [
        "# Project Contributors",
        "\nThe following agents have contributed to this project:\n"
    ]
    
    for name, data in sorted(agents.items()):
        created_by = f" ({data['created_by']})" if "created_by" in data else ""
        nature = f"[{data['nature_of_identity']}]" if "nature_of_identity" in data else ""
        output.append(f"- {name}{created_by} {nature}")
        
    return "\n".join(output)

if __name__ == "__main__":
    print(generate_credits())
