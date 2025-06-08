#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# setup_env.sh — unified installer for core, extras, CPU/GPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# 1. Create & activate venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

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
pip install -r requirements.txt -r requirements-dev.txt
if [[ $EXTRAS -eq 1 ]]; then
  pip install .[plot]
fi

# 4. Handle CPU vs GPU torch & optional ML extras
if [[ $EXTRAS -eq 1 ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "👷 Installing CPU-only torch (CI environment)"
    pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
  elif [[ $FORCE_GPU -eq 1 ]]; then
    echo "⚡ Installing GPU-enabled torch"
    pip install torch --index-url https://download.pytorch.org/whl/cu118
  else
    echo "🧠 Installing CPU-only torch (default)"
    pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
  fi

  if [[ $ML -eq 1 ]]; then
    echo "🔬 Installing ML extras"
    pip install .[ml]
  fi
fi

# 5. Prefetch large models if requested
if [[ $PREFETCH -eq 1 ]]; then
  bash fetch_models.sh
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
