# Windows PowerShell environment setup script for SpeakToMe
# No Unicode. All pip/python commands run inside venv unless -NoVenv is used.

param(
    [switch]$NoExtras,
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
    
    # Always use Windows paths in PowerShell
    $venvDir = Join-Path $PSScriptRoot '.venv'
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    $venvPip = Join-Path $venvDir 'Scripts\pip.exe'
    
    # Activate the virtual environment
    $activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'
    if (Test-Path $activateScript) {
        . $activateScript
    } else {
        Write-Host "Error: Virtual environment activation script not found at $activateScript"
        return
    }
} else {
    $venvPython = "python"
    $venvPip = "pip"
}

# 2. Install core + dev requirements
Safe-Run { & $venvPython -m pip install --upgrade pip }

# Always install torch first for GPU safety
if ($env:GITHUB_ACTIONS -eq 'true') {
    Write-Host 'Installing CPU-only torch (CI environment)'
    Safe-Run { & $venvPip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html }
} elseif ($gpu) {
    Write-Host 'Installing GPU-enabled torch'
    Safe-Run { & $venvPip install torch -f https://download.pytorch.org/whl/cu118 }
} else {
    Write-Host 'Installing CPU-only torch (default)'
    Safe-Run { & $venvPip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html }
}

# If not called from a dev script, launch the dev menu for all codebase/group installs
$calledByDev = $false
foreach ($arg in $args) {
    if ($arg -eq '--from-dev') { $calledByDev = $true }
}

if (-not $calledByDev) {
    Write-Host "Launching codebase/group selection tool for editable installs..."
    $env:PIP_CMD = $venvPip
    & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install
    Remove-Item Env:PIP_CMD
}

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate.ps1' on Windows or 'source .venv/bin/activate' on Unix-like systems"
