#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
CODEBASE="$(basename "$SCRIPT_DIR")"
MAP="$REPO_ROOT/AGENTS/codebase_map.json"
get_groups() {
python - "$CODEBASE" "$MAP" <<'PY'
import json, sys
name=sys.argv[1]; path=sys.argv[2]
try:
    data=json.load(open(path))
    groups=data.get(name, {}).get("groups", {})
except Exception:
    groups={}
notorch=[g for g,pkgs in groups.items() if not any("torch" in p for p in pkgs)]
print(','.join(notorch))
print(','.join(groups))
PY
}
read -r NOTORCH_GROUPS ALL_GROUPS <<< "$(get_groups)"
MODE="notorch"
for arg in "$@"; do
  case $arg in
    -full|-torch) MODE="full" ;;
    -minimal) MODE="minimal" ;;
    -notorch) MODE="notorch" ;;
  esac
done
case $MODE in
  full)
    GROUPS="$ALL_GROUPS"; NOTORCH="" ;;
  minimal)
    GROUPS=""; NOTORCH="-notorch" ;;
  *)
    GROUPS="$NOTORCH_GROUPS"; NOTORCH="-notorch" ;;
esac
ARGS=(-headless -codebases="$CODEBASE")
[ -n "$GROUPS" ] && ARGS+=(-groups="$CODEBASE:$GROUPS")
cd "$REPO_ROOT"
bash "$REPO_ROOT/setup_env_dev.sh" $NOTORCH "${ARGS[@]}"
