import json
import os
from datetime import datetime
# --- END HEADER ---

USERS_DIR = os.path.join(os.path.dirname(__file__), "users")

def sanitize_filename(name):
    return name.replace(" ", "_").replace("/", "_")

def prompt(field, default=None, required=False):
    while True:
        val = input(f"{field}{' ['+default+']' if default else ''}: ").strip()
        if not val and default is not None:
            return default
        if val or not required:
            return val
        print("This field is required.")

def main():
    print("=== Agent Registration ===")
    name = prompt("Agent name", required=True)
    date_of_identity = prompt("Date of identity (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
    nature_of_identity = prompt("Nature of identity (human/llm/script/hybrid/system_utility)", required=True)
    entry_point = prompt("Entry point (code, script, or description)")
    description = prompt("Description")
    created_by = prompt("Created by (if different from name)")
    tags = prompt("Tags (comma-separated)").split(",") if prompt("Add tags? (y/n)", default="n").lower() == "y" else []
    notes = prompt("Notes")
    
    profile = {
        "name": name,
        "date_of_identity": date_of_identity,
        "nature_of_identity": nature_of_identity,
    }
    if entry_point: profile["entry_point"] = entry_point
    if description: profile["description"] = description
    if created_by: profile["created_by"] = created_by
    if tags: profile["tags"] = [t.strip() for t in tags if t.strip()]
    if notes: profile["notes"] = notes

    filename = f"{date_of_identity}_{sanitize_filename(name)}.json"
    filepath = os.path.join(USERS_DIR, filename)
    os.makedirs(USERS_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"Agent profile saved to {filepath}")

if __name__ == "__main__":
    main()
