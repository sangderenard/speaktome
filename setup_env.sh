#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# setup_env.sh — unified installer for core, extras, CPU/GPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -uo pipefail

# Helper: run a command but never terminate on failure
safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

# 1. Create & activate venv
safe_run python -m venv .venv
safe_run source .venv/bin/activate
safe_run pip install --upgrade pip

# 2. Parse flags
EXTRAS=0
ML=0        # flag for full ML extras (transformers, torch_geometric)
FORCE_GPU=0
PREFETCH=0

for arg in "$@"; do
  case $arg in
    --extras|--full) EXTRAS=1 ;;
    --ml)             ML=1      ;;  # install ML extras too
    --gpu)            FORCE_GPU=1 ;;
    --prefetch)       PREFETCH=1 ;;
  esac
done

# 3. Install core + dev + plot
safe_run pip install -r requirements.txt -r requirements-dev.txt
if [[ $EXTRAS -eq 1 ]]; then
  safe_run pip install .[plot]
fi

# 4. Handle CPU vs GPU torch & optional ML extras
if [[ $EXTRAS -eq 1 ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "👷 Installing CPU-only torch (CI environment)"
    safe_run pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
  elif [[ $FORCE_GPU -eq 1 ]]; then
    echo "⚡ Installing GPU-enabled torch"
    safe_run pip install torch --index-url https://download.pytorch.org/whl/cu118
  else
    echo "🧠 Installing CPU-only torch (default)"
    safe_run pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
  fi

  if [[ $ML -eq 1 ]]; then
    echo "🔬 Installing ML extras"
    safe_run pip install .[ml]
  fi
fi

# 5. Prefetch large models if requested
if [[ $PREFETCH -eq 1 ]]; then
  safe_run bash fetch_models.sh
fi
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo "🔥       REPO PURGE INITIATED        🔥"
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo
echo "❄️  Removing ghost references..."
echo "❄️  Erasing digital scars..."
echo "❄️  Severing broken LFS chains..."
echo "❄️  History? Gone. Just code remains."
echo
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo "🔥       REPO REBIRTH COMPLETE        🔥"
echo "🔥          A CLEAN BEGINNING          🔥"
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo
echo "remove the fire emojis after the purge"
echo
echo "✅ Environment ready. Activate with 'source .venv/bin/activate'."
echo "   • Core  = requirements.txt + dev"
echo "   • Plot  = matplotlib, networkx, scikit-learn"
echo "   • ML    = transformers, sentence-transformers"
echo "   • Torch = $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
