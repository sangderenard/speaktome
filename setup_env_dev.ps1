# PowerShell developer environment setup script for SpeakToMe

$ErrorActionPreference = 'Stop'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# Run the regular setup script with forwarded arguments
$scriptRoot = $PSScriptRoot
Safe-Run { & "$scriptRoot\setup_env.ps1" @args }

$venvDir = Join-Path $scriptRoot '.venv'
if ($IsWindows) {
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    $venvPip = Join-Path $venvDir 'Scripts\pip.exe'
} else {
    $venvPython = Join-Path $venvDir 'bin/python'
    $venvPip = Join-Path $venvDir 'bin/pip'
}

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
        $response = $null
        Write-Host "Install optional group '$group'? [y/N] (auto-skip in 3s): " -NoNewline
        for ($i=3; $i -gt 0; $i--) {
            if ($Host.UI.RawUI.KeyAvailable) {
                $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
                $response = $key.Character
                break
            }
            Start-Sleep -Seconds 1
        }
        if (-not $response) { $response = 'N' }
        if ($response -match '^[Yy]$') {
            Write-Host "Attempting to install optional group: $group"
            try {
                & $venvPip "install" ".[$group]"
            } catch {
                Write-Host "Warning: Failed to install optional group: $group. Error: $($_.Exception.Message)"
            }
        } else {
            Write-Host "Skipping optional group: $group"
        }
    }

    $backendGroups = @("numpy", "jax", "ctensor")
    foreach ($group in $backendGroups) {
        $response = $null
        Write-Host "Install backend group '$group'? [y/N] (auto-skip in 3s): " -NoNewline
        for ($i=3; $i -gt 0; $i--) {
            if ($Host.UI.RawUI.KeyAvailable) {
                $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
                $response = $key.Character
                break
            }
            Start-Sleep -Seconds 1
        }
        if (-not $response) { $response = 'N' }
        if ($response -match '^[Yy]$') {
            Write-Host "Attempting to install backend group: $group"
            try {
                & $venvPip "install" ".[$group]"
            } catch {
                Write-Host "Warning: Failed to install backend group: $group. Error: $($_.Exception.Message)"
            }
        } else {
            Write-Host "Skipping backend group: $group"
        }
    }

    Pop-Location
}

Install-Speaktome-Extras

# Optionally run document dump with user confirmation and countdown
Write-Host ""
$docdump = $null
Write-Host "Run document dump (headers, stubs, docs)? [Y/n] (auto-yes in 10s): "
for ($i=10; $i -gt 0; $i--) {
    if ($Host.UI.RawUI.KeyAvailable) {
        $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        $docdump = $key.Character
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $docdump) { $docdump = "Y" }
if ($docdump -match "^[Yy]$") {
    Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }
    Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }
    Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }
    Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }
} else {
    Write-Host "Document dump skipped."
}
