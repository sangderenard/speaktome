# PowerShell developer environment setup script for SpeakToMe

$ErrorActionPreference = 'Stop'
$env_box_file = Join-Path $PSScriptRoot "ENV_SETUP_BOX.md"
$env:ENV_SETUP_BOX = "`n$(Get-Content $env_box_file -Raw)`n"

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# Run the regular setup script with forwarded arguments
$scriptRoot = $PSScriptRoot
$activeFile = $env:SPEAKTOME_ACTIVE_FILE
if (-not $activeFile) { $activeFile = Join-Path ([System.IO.Path]::GetTempPath()) 'speaktome_active.json' }
$env:SPEAKTOME_ACTIVE_FILE = $activeFile
$menuArgs = @()
$useVenv = $true
$torchChoice = ''
foreach ($arg in $args) {
    $arg_lc = $arg.ToLower()
    if ($arg_lc -eq '-no-venv') {
        $useVenv = $false
    }
    elseif ($arg_lc -like '-codebases=*' -or $arg_lc -like '-cb=*') {
        $cbVal = $arg.Split('=')[1]
        if ($cbVal -and $cbVal.Trim().Length -gt 0) {
            $menuArgs += '-codebases'
            $menuArgs += $cbVal
        }
    }
    elseif ($arg_lc -like '-groups=*' -or $arg_lc -like '-grp=*') {
        $grpVal = $arg.Split('=')[1]
        if ($grpVal -and $grpVal.Trim().Length -gt 0) {
            $menuArgs += '-groups'
            $menuArgs += $grpVal
        }
    }
    elseif ($arg_lc -eq '-torch') { $torchChoice = 'cpu' }
    elseif ($arg_lc -eq '-gpu' -or $arg_lc -eq '-gpu-torch') { $torchChoice = 'gpu' }
}
$poetryArgs = '--without cpu-torch --without gpu-torch'
if ($torchChoice -eq 'cpu') { $poetryArgs = '--with cpu-torch' }
elseif ($torchChoice -eq 'gpu') { $poetryArgs = '--with gpu-torch' }
$env:SPEAKTOME_POETRY_ARGS = $poetryArgs

$filtered = @()
foreach ($arg in $args) {
    $arg_lc = $arg.ToLower()
    if ($arg_lc -ne '-torch' -and $arg_lc -ne '-gpu' -and $arg_lc -ne '-gpu-torch') {
        $filtered += $arg
    }
}
Safe-Run { & "$scriptRoot\setup_env.ps1" @filtered -from-dev }
Remove-Item Env:SPEAKTOME_POETRY_ARGS

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

    # Install dev requirements via Poetry
    Safe-Run { poetry install --with dev }
}

function Read-KeyWithTimeout([int]$seconds) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        if ([System.Console]::KeyAvailable) {
            $key = [System.Console]::ReadKey($true)
            if ($key.Key -eq 'Enter') { continue }
            return $key.KeyChar
        }
        Start-Sleep -Milliseconds 100
    }
    return $null
}

function Show-Menu {
    param([int]$Timeout = 5)
    while ($true) {
        Write-Host "`nDeveloper info menu (timeout $Timeout s):"
        Write-Host " 0) Dump headers"
        Write-Host " 1) Stub finder"
        Write-Host " 2) List contributors"
        Write-Host " 3) Preview AGENT_CONSTITUTION.md"
        Write-Host " 4) Preview AGENTS.md"
        Write-Host " 5) Preview LICENSE"
        Write-Host " 6) Preview CODING_STANDARDS.md"
        Write-Host " 7) Preview CONTRIBUTING.md"
        Write-Host " 8) Preview PROJECT_OVERVIEW.md"
        Write-Host " 9) Launch dev group menu"
        Write-Host " q) Quit"
        Write-Host -NoNewline "Select option: "
        $choice = Read-KeyWithTimeout $Timeout
        if (-not $choice) { Write-Host "No input in $Timeout seconds. Exiting."; break }
        switch ($choice.ToString().ToLower()) {
            '0' { Write-Host "Running dump_headers"; Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }; $Timeout = 10 }
            '1' { Write-Host "Running stubfinder"; Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }; $Timeout = 10 }
            '2' { Write-Host "Running list_contributors"; Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }; $Timeout = 10 }
            '3' { Write-Host "Preview AGENT_CONSTITUTION.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }; $Timeout = 10 }
            '4' { Write-Host "Preview AGENTS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }; $Timeout = 10 }
            '5' { Write-Host "Preview LICENSE"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }; $Timeout = 10 }
            '6' { Write-Host "Preview CODING_STANDARDS.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }; $Timeout = 10 }
            '7' { Write-Host "Preview CONTRIBUTING.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }; $Timeout = 10 }
            '8' { Write-Host "Preview PROJECT_OVERVIEW.md"; Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }; $Timeout = 10 }
            '9' { Write-Host "Launching dev group menu"; Safe-Run { & $venvPython AGENTS/tools/dev_group_menu.py --record $activeFile @menuArgs }; $Timeout = 10 }
            'q' { Write-Host "Exiting."; break }
            default { Write-Host "Unknown choice: $choice" }
        }
        Write-Host ""
    }
}

Show-Menu
Write-Host "For advanced codebase/group selection, run: python AGENTS/tools/dev_group_menu.py"
Write-Host "Selections recorded to $activeFile"

# All options for this script should be used with single-dash PowerShell-style flags, e.g.:
#   -no-venv -Codebases projectA,projectB -Groups groupX
# Do not use double-dash flags with this script.
