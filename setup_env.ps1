# Windows PowerShell environment setup script for SpeakToMe
# Mirrors setup_env.sh functionality

param(
    [switch]$extras,
    [switch]$ml,
    [switch]$gpu,
    [switch]$prefetch
)

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# 1. Create & activate venv
Safe-Run { python -m venv .venv }
Safe-Run { & .venv\Scripts\Activate.ps1 }
Safe-Run { pip install --upgrade pip }

# 2. Install core + dev requirements
Safe-Run { pip install -r requirements.txt -r requirements-dev.txt }
if ($extras) {
    Safe-Run { pip install .[plot] }
}

# Install package in editable mode so changes are picked up automatically
Safe-Run { pip install -e . }

# 3. Handle CPU vs GPU torch & optional ML extras
if ($extras) {
    if ($env:GITHUB_ACTIONS -eq 'true') {
        Write-Host '👷 Installing CPU-only torch (CI environment)'
        Safe-Run { pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu }
    } elseif ($gpu) {
        Write-Host '⚡ Installing GPU-enabled torch'
        Safe-Run { pip install torch --index-url https://download.pytorch.org/whl/cu118 }
    } else {
        Write-Host '🧠 Installing CPU-only torch (default)'
        Safe-Run { pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu }
    }

    if ($ml) {
        Write-Host '🔬 Installing ML extras'
        Safe-Run { pip install .[ml] }
    }
}

# 4. Prefetch large models if requested
if ($prefetch) {
    Safe-Run { powershell -ExecutionPolicy Bypass -File fetch_models.ps1 }
}

# Ensure optional dependencies from pyproject.toml are installed
Safe-Run { python AGENTS/tools/ensure_pyproject_deps.py }

Write-Host "Environment setup complete. Activate with '.venv\\Scripts\\Activate.ps1'"
