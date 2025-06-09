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

Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }
Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }
Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }

function Install-Speaktome-Extras {
    $speaktomeDir = Join-Path $PSScriptRoot "speaktome"
    Push-Location $speaktomeDir

    # Ensure pip is up to date and editable install is present
    & $venvPip install --upgrade pip
    & $venvPip install -e .

    $optionalGroups = @("plot", "ml", "dev")
    foreach ($group in $optionalGroups) {
        Write-Host "Attempting to install optional group: $group"
        try {
            & $venvPip "install" ".[$group]"
        } catch {
            Write-Host "Warning: Failed to install optional group: $group"
        }
    }

    $backendGroups = @("numpy", "jax", "ctensor")
    foreach ($group in $backendGroups) {
        Write-Host "Attempting to install backend group: $group"
        try {
            & $venvPip "install" ".[$group]"
        } catch {
            Write-Host "Warning: Failed to install backend group: $group"
        }
    }

    Pop-Location
}

Install-Speaktome-Extras
