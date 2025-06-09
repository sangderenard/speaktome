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
$useVenv = $true
foreach ($arg in $args) {
    if ($arg -eq '--no-venv' -or $arg -eq '-no-venv') { $useVenv = $false }
}
Safe-Run { & "$scriptRoot\setup_env.ps1" @args --from-dev }

# Update the venv path handling section:
if ($useVenv) {
    $venvDir = Join-Path $scriptRoot '.venv'
    # Always use Windows paths since we're in PowerShell
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    $venvPip = Join-Path $venvDir 'Scripts\pip.exe'
} else {
    $venvPython = 'python'
    $venvPip = 'pip'
}

# Update the activation section:
if ($useVenv) {
    if (-not (Test-Path $venvPython)) {
        Write-Host "Error: Virtual environment Python not found at $venvPython"
        return
    }
    if (-not (Test-Path $venvPip)) {
        Write-Host "Error: Virtual environment Pip not found at $venvPip"
        return
    }

    # Always use Windows path for activation in PowerShell
    $activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'
    if (Test-Path $activateScript) {
        . $activateScript
    } else {
        Write-Host "Error: Virtual environment activation script not found at $activateScript"
        return
    }
}


# Update the Install-Speaktome-Extras function to use proper paths:
function Install-Speaktome-Extras {
    $speaktomeDir = Join-Path $scriptRoot "speaktome"
    if (-not (Test-Path $speaktomeDir -PathType Container)) {
        Write-Host "Warning: SpeakToMe directory not found at '$speaktomeDir'. Skipping extras installation."
        return
    }
    Push-Location $speaktomeDir

    Write-Host "Attempting to upgrade pip..."
    try {
        & $venvPython -m pip install --upgrade pip
    } catch {
        Write-Host "Warning: Failed to upgrade pip. Error: $($_.Exception.Message)"
    }

    Write-Host "Launching codebase/group selection tool..."
    $env:PIP_CMD = $venvPip
    & $venvPython (Join-Path $scriptRoot "AGENTS\tools\dev_group_menu.py") --install
    Remove-Item Env:PIP_CMD

    Pop-Location
}

Install-Speaktome-Extras

# Interactive document menu with inactivity timeout
function Read-InputWithTimeout([int]$seconds) {
    $task = [System.Threading.Tasks.Task[string]]::Factory.StartNew({ [Console]::ReadLine() })
    if ($task.Wait([TimeSpan]::FromSeconds($seconds))) { return $task.Result } else { return $null }
}

function Show-Menu {
    param([int]$Timeout = 5)
    while ($true) {
        Write-Host "`nDeveloper info menu (timeout $Timeout s):"
        Write-Host " 1) Dump headers"
        Write-Host " 2) Stub finder"
        Write-Host " 3) List contributors"
        Write-Host " 4) Preview AGENT_CONSTITUTION.md"
        Write-Host " 5) Preview AGENTS.md"
        Write-Host " 6) Preview LICENSE"
        Write-Host " 7) Preview CODING_STANDARDS.md"
        Write-Host " 8) Preview CONTRIBUTING.md"
        Write-Host " 9) Preview PROJECT_OVERVIEW.md"
        Write-Host "10) Launch dev group menu"
        Write-Host " q) Quit"
        Write-Host -NoNewline "Select option: "
        $choice = Read-InputWithTimeout $Timeout
        if (-not $choice) { Write-Host "No input in $Timeout seconds. Exiting."; break }
        switch ($choice) {
            '1' { Write-Host "Running dump_headers"; Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }; $Timeout = 60 }
            '2' { Write-Host "Running stubfinder"; Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }; $Timeout = 60 }
            '3' { Write-Host "Running list_contributors"; Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }; $Timeout = 60 }
            '4' { Write-Host "Preview AGENT_CONSTITUTION.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }; $Timeout = 60 }
            '5' { Write-Host "Preview AGENTS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }; $Timeout = 60 }
            '6' { Write-Host "Preview LICENSE"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }; $Timeout = 60 }
            '7' { Write-Host "Preview CODING_STANDARDS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }; $Timeout = 60 }
            '8' { Write-Host "Preview CONTRIBUTING.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }; $Timeout = 60 }
            '9' { Write-Host "Preview PROJECT_OVERVIEW.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }; $Timeout = 60 }
            '10' { Write-Host "Launching dev group menu"; Safe-Run { & $venvPython AGENTS/tools/dev_group_menu.py }; $Timeout = 60 }
            'q' { Write-Host "Exiting."; break }
            'Q' { Write-Host "Exiting."; break }
            default { Write-Host "Unknown choice: $choice" }
        }
        Write-Host ""
    }
}

Show-Menu
Write-Host "For advanced codebase/group selection, run: python AGENTS/tools/dev_group_menu.py"
