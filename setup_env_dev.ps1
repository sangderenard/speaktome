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

# Interactive document menu with inactivity timeout
function Read-InputWithTimeout([int]$seconds) {
    $task = [System.Threading.Tasks.Task[string]]::Factory.StartNew({ [Console]::ReadLine() })
    if ($task.Wait([TimeSpan]::FromSeconds($seconds))) { return $task.Result } else { return $null }
}

function Show-Menu {
    param([int]$Timeout = 60)
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
        Write-Host " q) Quit"
        Write-Host -NoNewline "Select option: "
        $choice = Read-InputWithTimeout $Timeout
        if (-not $choice) { Write-Host "No input in $Timeout seconds. Exiting."; break }
        switch ($choice) {
            '1' { Write-Host "Running dump_headers"; Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown } }
            '2' { Write-Host "Running stubfinder"; Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome } }
            '3' { Write-Host "Running list_contributors"; Safe-Run { & $venvPython AGENTS/tools/list_contributors.py } }
            '4' { Write-Host "Preview AGENT_CONSTITUTION.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md } }
            '5' { Write-Host "Preview AGENTS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md } }
            '6' { Write-Host "Preview LICENSE"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE } }
            '7' { Write-Host "Preview CODING_STANDARDS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md } }
            '8' { Write-Host "Preview CONTRIBUTING.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md } }
            '9' { Write-Host "Preview PROJECT_OVERVIEW.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md } }
            'q' { Write-Host "Exiting."; break }
            'Q' { Write-Host "Exiting."; break }
            default { Write-Host "Unknown choice: $choice" }
        }
        Write-Host ""
    }
}

Show-Menu
