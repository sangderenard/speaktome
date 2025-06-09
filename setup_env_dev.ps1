# PowerShell developer environment setup script for SpeakToMe

$ErrorActionPreference = 'Stop'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# Run the regular setup script with forwarded arguments
Safe-Run { powershell -ExecutionPolicy Bypass -File setup_env.ps1 @args }

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$venvPip = Join-Path $PSScriptRoot '.venv\Scripts\pip.exe'

if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment Python not found at $venvPython"
    return
}
if (-not (Test-Path $venvPip)) {
    Write-Host "Error: Virtual environment Pip not found at $venvPip. Ensure setup_env.ps1 created the .venv correctly."
    return
}


function Install-Speaktome-Extras {
    $speaktomeDir = Join-Path $PSScriptRoot "speaktome"
    if (-not (Test-Path $speaktomeDir -PathType Container)) {
        Write-Host "Warning: SpeakToMe directory not found at '$speaktomeDir' or it is not a directory. Skipping extras installation."
        return
    }
    Push-Location $speaktomeDir

    # Ensure pip is up to date and editable install is present
    Write-Host "Attempting to upgrade pip..."
    try {
        & $venvPython -m pip install --upgrade pip
    } catch {
        Write-Host "Warning: Failed to upgrade pip. Error: $($_.Exception.Message)"
    }

    Write-Host "Attempting to install SpeakToMe in editable mode..."
    try {
        & $venvPip install -e .
    } catch {
        Write-Host "Warning: Failed to install SpeakToMe in editable mode. Error: $($_.Exception.Message)"
    }

    $optionalGroups = @("plot", "ml", "dev")
    foreach ($group in $optionalGroups) {
        Write-Host "Attempting to install optional group: $group"
        try {
            & $venvPip "install" ".[$group]"
        } catch {
            Write-Host "Warning: Failed to install optional group: $group. Error: $($_.Exception.Message)"
        }
    }

    $backendGroups = @("numpy", "jax", "ctensor")
    foreach ($group in $backendGroups) {
        Write-Host "Attempting to install backend group: $group"
        try {
            & $venvPip "install" ".[$group]"
        } catch {
            Write-Host "Warning: Failed to install backend group: $group. Error: $($_.Exception.Message)"
        }
    }

    Pop-Location
}

Install-Speaktome-Extras


Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }
Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }
Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }
