"""
Generates a dignified credits list from agent JSON files in AGENTS/users/.
Most recent version of each named agent is used.
"""
import json
import glob
from pathlib import Path
# --- END HEADER ---

def generate_credits():
    """Load all agent JSONs and create a formatted contributor list."""
    users_dir = Path(__file__).parent.parent / "users"
    agents = {}
    
    # Load all JSON files, later ones overwrite earlier ones with same name
    for json_path in sorted(users_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
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
