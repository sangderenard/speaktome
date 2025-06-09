#!/usr/bin/env bash
# ------------------------------------------------------------------------
# setup_env.sh -- no Unicode, all pip/python in venv unless --no-venv
# ------------------------------------------------------------------------

set -uo pipefail

USE_VENV=1
for arg in "$@"; do
  case $arg in
    --no-venv) USE_VENV=0 ;;
  esac
done

# Helper: run a command but never terminate on failure
safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

if [ $USE_VENV -eq 1 ]; then
  safe_run python -m venv .venv
  source .venv/bin/activate
  VENV_PYTHON="./.venv/bin/python"
  VENV_PIP="./.venv/bin/pip"
else
  VENV_PYTHON="python"
  VENV_PIP="pip"
fi

safe_run $VENV_PYTHON -m pip install --upgrade pip
safe_run $VENV_PIP install -r requirements.txt -r requirements-dev.txt

NOEXTRAS=0
ML=0        # flag for full ML extras (transformers, torch_geometric)
FORCE_GPU=0
PREFETCH=0
CODEBASES="speaktome,AGENTS/tools,time_sync"

for arg in "$@"; do
  case $arg in
    --noextras)       NOEXTRAS=1 ;;
    --ml)             ML=1      ;;  # install ML extras too
    --gpu)            FORCE_GPU=1 ;;
    --prefetch)       PREFETCH=1 ;;
    --codebases=*)    CODEBASES="${arg#*=}" ;;
  esac
done

# Always include the time_sync codebase
case ",${CODEBASES}," in
  *",time_sync,"*) ;;
  *) CODEBASES="${CODEBASES},time_sync" ;;
esac

if [ $NOEXTRAS -eq 0 ]; then
  safe_run $VENV_PIP install .[plot]

  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
      echo "Installing latest stable CPU-only torch (CI environment)"
      safe_run $VENV_PIP install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
  elif [ $FORCE_GPU -eq 1 ]; then
      echo "Installing GPU-enabled torch"
      safe_run $VENV_PIP install torch -f https://download.pytorch.org/whl/cu118
  else
      echo "Installing latest stable CPU-only torch (default)"
      safe_run $VENV_PIP install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
  fi

  if [ $ML -eq 1 ]; then
    echo "Installing ML extras"
    safe_run $VENV_PIP install .[ml]
  fi
fi

# Install codebases in editable mode so local changes apply immediately
IFS=',' read -ra CB <<< "$CODEBASES"
for cb in "${CB[@]}"; do
  [ "$cb" = "." ] && continue
  if [ -f "$cb/pyproject.toml" ] || [ -f "$cb/setup.py" ]; then
    safe_run $VENV_PIP install -e "$cb"
  fi
done

# Prefetch large models if requested
if [ $PREFETCH -eq 1 ]; then
  safe_run bash fetch_models.sh
fi

safe_run $VENV_PYTHON AGENTS/tools/ensure_pyproject_deps.py

echo "Environment setup complete."
echo "[OK] Environment ready. Activate with 'source .venv/bin/activate'."
echo "   * Core  = requirements.txt + dev"
echo "   * Plot  = matplotlib, networkx, scikit-learn"
echo "   * ML    = transformers, sentence-transformers"
echo "   * Torch = $(python -c 'import torch; print(torch.__version__, torch.version.cuda if torch.cuda.is_available() else "N/A")')"
