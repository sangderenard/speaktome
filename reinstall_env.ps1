# Windows PowerShell script to reinstall the environment for SpeakToMe
# Removes the .venv directory and runs setup_env.ps1

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

param(
    [switch]$Yes
)

if (-not $Yes) {
    $confirm = Read-Host "This will delete the .venv directory and reinstall dependencies. Continue? [y/N]"
    if (-not $confirm) { $confirm = 'N' }
    if ($confirm -notmatch '^[Yy]$') {
        Write-Host 'Aborted.'
        exit
    }
}

if (Test-Path '.venv') { Safe-Run { Remove-Item -Recurse -Force '.venv' } }
Safe-Run { powershell -ExecutionPolicy Bypass -File setup_env.ps1 @args }
