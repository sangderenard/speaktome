# Windows PowerShell environment setup script for SpeakToMe
# No Unicode. All pip/python commands run inside venv unless -NoVenv is used.

param(
    [switch]$extras,
    [switch]$ml,
    [switch]$gpu,
    [switch]$prefetch,
    [switch]$NoVenv,
    [string[]]$Codebases = @("speaktome", "AGENTS/tools", "time_sync")
)

# Always include the time_sync codebase
if (-not ($Codebases -contains 'time_sync')) {
    $Codebases += 'time_sync'
}

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

if (-not $NoVenv) {
    # 1. Create venv
    Safe-Run { python -m venv .venv }
    if ($IsWindows) {
        # Windows layout
        Safe-Run { & .venv\Scripts\Activate.ps1 }
        $venvPython = ".venv\Scripts\python.exe"
        $venvPip = ".venv\Scripts\pip.exe"
    }
    else {
        # POSIX layout
        $venvPython = "./.venv/bin/python"
        $venvPip = "./.venv/bin/pip"
    }
} else {
    $venvPython = "python"
    $venvPip = "pip"
}

# 2. Install core + dev requirements
Safe-Run { & $venvPython -m pip install --upgrade pip }
Safe-Run { & $venvPip install -r requirements.txt -r requirements-dev.txt }
if ($extras) {
    Safe-Run { & $venvPip install .[plot] }

    # 3. Handle CPU vs GPU torch & optional ML extras
    if ($env:GITHUB_ACTIONS -eq 'true') {
        Write-Host 'Installing CPU-only torch (CI environment)'
        Safe-Run { & $venvPip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu }
    } elseif ($gpu) {
        Write-Host 'Installing GPU-enabled torch'
        Safe-Run { & $venvPip install torch --index-url https://download.pytorch.org/whl/cu118 }
    } else {
        Write-Host 'Installing CPU-only torch (default)'
        Safe-Run { & $venvPip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu }
    }

    if ($ml) {
        Write-Host 'Installing ML extras'
        Safe-Run { & $venvPip install .[ml] }
    }
}

# Install codebases in editable mode so changes are picked up automatically
foreach ($cb in $Codebases) {
    if ($cb -ne "." -and ((Test-Path "$cb\pyproject.toml") -or (Test-Path "$cb\setup.py"))) {
        Safe-Run { & $venvPip install -e $cb }
    }
}

# 4. Prefetch large models if requested
if ($prefetch) {
    Safe-Run { powershell -ExecutionPolicy Bypass -File fetch_models.ps1 }
}

# Ensure optional dependencies from pyproject.toml are installed
Safe-Run { & $venvPython AGENTS/tools/ensure_pyproject_deps.py }

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate.ps1' on Windows or 'source .venv/bin/activate' on Unix-like systems"
