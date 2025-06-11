# Windows PowerShell environment setup script for SpeakToMe
# No Unicode. All pip/python commands run inside venv unless -NoVenv is used.

param(
    [switch]$Extras,
    [switch]$NoExtras,
    [switch]$ml,
    [switch]$gpu,
    [switch]$prefetch,
    [switch]$NoVenv,
    [switch]$headless,
    [switch]$FromDev,
    [string[]]$Codebases,
    [string[]]$Groups
)

$activeFile = $env:SPEAKTOME_ACTIVE_FILE
if (-not $activeFile) {
    $activeFile = Join-Path ([System.IO.Path]::GetTempPath()) 'speaktome_active.json'
}
$env:SPEAKTOME_ACTIVE_FILE = $activeFile
if (-not $Codebases) {
    if ($headless) {
        Write-Host "[DEBUG] No codebases specified - headless mode: auto-loading from codebase_map.json."
        $mapFile = Join-Path $PSScriptRoot 'AGENTS\codebase_map.json'
        if (Test-Path $mapFile) {
            try {
                $Codebases = (Get-Content $mapFile | ConvertFrom-Json).psobject.Properties.Name
                # Optionally load all groups for a full install:
                # $Groups = @( 'groupA','groupB','...' )
            }
            catch {
                Write-Host "[DEBUG] Could not parse codebase_map.json; continuing empty."
                $Codebases = $null
            }
        }
    }
    else {
        Write-Host "[DEBUG] No codebases specified - launching interactive menu."
        # Let dev_group_menu handle interactive selection
    }
}
if ($Extras) { $NoExtras = $false }
$menuArgs = @()
if ($Codebases) { $menuArgs += '--codebases'; $menuArgs += ($Codebases -join ',') }
if ($Groups) {
    foreach ($g in $Groups) {
        if ($g -and $g.Trim().Length -gt 0) {
            $menuArgs += '--groups'
            $menuArgs += $g
        }
    }
}
Write-Host "[DEBUG] Codebases: $($Codebases -join ';')"
Write-Host "[DEBUG] Groups: $($Groups -join ';')"

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# Run a command silently and output logs only once data begins streaming.
function Install-Quiet($exePath, [string]$commandArguments) {
    Write-Host "[DEBUG] Install-Quiet function entered."
    Write-Host "[DEBUG]   exePath Value: '$exePath'"
    Write-Host "[DEBUG]   exePath Type: $($exePath.GetType().FullName)"
    Write-Host "[DEBUG]   commandArguments Value: '$commandArguments'"
    Write-Host "[DEBUG]   commandArguments Type: $($commandArguments.GetType().FullName)"
    if (-not $commandArguments -or $commandArguments.Trim().Length -eq 0) {
        Write-Host "Warning: Install-Quiet called with empty commandArguments ('$commandArguments'). Skipping."
        return
    }
    $log = [System.IO.Path]::GetTempFileName()

    # Prepare command for cmd.exe to handle redirection of both stdout and stderr to a single file
    # Quote executable path if it contains spaces for cmd.exe
    $cmdQuotedExePath = if ($exePath -match ' ') { "`"$exePath`"" } else { $exePath }

    # $commandArguments is already a string.
    # This forms the actual command to be run (e.g., "C:\path\to\pip.exe" install torch)
    $fullNativeCommand = "$cmdQuotedExePath $commandArguments"

    # Quote log file path if it contains spaces for cmd.exe redirection
    $cmdQuotedLogPath = if ($log -match ' ') { "`"$log`"" } else { $log }

    # Arguments for cmd.exe: /D (disable AutoRun) /C "actual_command > log_file 2>&1"
    # The string after /C is executed by cmd.exe, including its redirection.
    $cmdArgumentList = @("/D", "/C", "$fullNativeCommand > $cmdQuotedLogPath 2>&1")

    Write-Host "[DEBUG] Starting process via cmd.exe: cmd.exe $($cmdArgumentList -join ' ')"
    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgumentList -PassThru -NoNewWindow
    $timeout = 60
    $elapsed = 0
    $started = $false
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 1
        $elapsed += 1
        if (-not $started -and (Test-Path $log) -and ((Get-Item $log).Length -gt 0)) {
            $started = $true
            # Instead of tailing indefinitely, read the entire file once:
            Get-Content -Path $log
        }
        if (-not $started -and $elapsed -ge $timeout) {
            $proc | Stop-Process
            Write-Host 'Optional Torch download did not succeed, continuing anyway.'
            Get-Content $log
            Remove-Item $log
            return
        }
    }
    $exitCode = $proc.ExitCode
    if (-not $started) { Get-Content $log }
    Remove-Item $log
    if ($exitCode -ne 0) {
        Write-Host 'Optional Torch download did not succeed, continuing anyway.'
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
    $activateScript = Join-Path $venvDir 'Scripts\Activate'
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
Safe-Run { & $venvPython -m pip install wheel }  # <-- Add this line

# Always install torch first
if ($env:GITHUB_ACTIONS -eq 'true') {
    Write-Host 'Installing CPU-only torch (CI environment)'
    Install-Quiet $venvPip "install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"
} elseif ($gpu) {
    Write-Host 'Installing GPU-enabled torch'
    Install-Quiet $venvPip "install torch -f https://download.pytorch.org/whl/cu118"
} else {
    Write-Host 'Installing CPU-only torch (default)'
    Install-Quiet $venvPip "install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"
}

# If not called from a dev script, launch the dev menu for all codebase/group installs

# Always run dev_group_menu.py at the end
if ($FromDev) {
    Write-Host "Installing AGENTS/tools in headless mode..."
    $env:PIP_CMD = $venvPip
    & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install --codebases 'tools' --record $activeFile
    Write-Host "Launching codebase/group selection tool for editable installs (from-dev)..."
    if ($menuArgs.Count -gt 0) {
        & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install --record $activeFile @menuArgs
    } else {
        & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install --record $activeFile
    }
    Remove-Item Env:PIP_CMD
} else {
    Write-Host "Launching codebase/group selection tool for editable installs..."
    $env:PIP_CMD = $venvPip
    if ($menuArgs.Count -gt 0) {
        & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install --record $activeFile @menuArgs
    } else {
        & $venvPython (Join-Path $PSScriptRoot "AGENTS\tools\dev_group_menu.py") --install --record $activeFile
    }
    Remove-Item Env:PIP_CMD
}

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate' on Windows or 'source .venv/bin/activate' on Unix-like systems"
Write-Host "Selections recorded to $activeFile"
try {
    $torchInfo = & $venvPython -c "import importlib,sys;spec=importlib.util.find_spec('torch');print('missing' if spec is None else __import__('torch').__version__)"
} catch {
    $torchInfo = 'missing'
}
Write-Host "Torch = $torchInfo"
