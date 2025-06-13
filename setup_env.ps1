# Windows PowerShell environment setup script for SpeakToMe
# No Unicode. All pip/python commands run inside venv unless -no-venv is used.

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$args
)

# Manual flag parsing for all arguments (case-insensitive, -flag=value style)
$UseVenv = $true
$FromDev = $false
$Codebases = @()
$Groups = @()
foreach ($arg in $args) {
    $arg_lc = $arg.ToLower()
    if ($arg_lc -eq '-no-venv') { $UseVenv = $false }
    elseif ($arg_lc -eq '-fromdev') { $FromDev = $true }
    elseif ($arg_lc -like '-codebases=*') {
        $cbVal = $arg.Substring($arg.IndexOf('=')+1)
        if ($cbVal -and $cbVal.Trim().Length -gt 0) {
            $Codebases = $cbVal -split ','
        }
    }
    elseif ($arg_lc -like '-groups=*') {
        $grpVal = $arg.Substring($arg.IndexOf('=')+1)
        if ($grpVal -and $grpVal.Trim().Length -gt 0) {
            $Groups = $grpVal -split ','
        }
    }
}

# All options for this script should be used with single-dash PowerShell-style flags, e.g.:
#   -no-venv -Codebases projectA,projectB -Groups groupX
# Do not use double-dash flags with this script.

$activeFile = $env:SPEAKTOME_ACTIVE_FILE
if (-not $activeFile) {
    $activeFile = Join-Path ([System.IO.Path]::GetTempPath()) 'speaktome_active.json'
}
$env:SPEAKTOME_ACTIVE_FILE = $activeFile
if (-not $Codebases) {
    Write-Host "[DEBUG] No codebases specified - launching interactive menu."
    # Let dev_group_menu handle interactive selection
}
$menuArgs = @()
if ($Codebases) { $menuArgs += '-Codebases'; $menuArgs += ($Codebases -join ',') }
if ($Groups) {
    foreach ($g in $Groups) {
        if ($g -and $g.Trim().Length -gt 0) {
            $menuArgs += '-Groups'
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

if ($UseVenv) {
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


# 2. Delegate package installation to dev_group_menu.py

# If not called from a dev script, launch the dev menu for all codebase/group installs

# Always run dev_group_menu.py at the end
if ($FromDev) {
    Write-Host "Installing AGENTS/tools..."
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
