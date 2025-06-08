# Windows PowerShell environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

Safe-Run { python -m venv .venv }
Safe-Run { & .venv\Scripts\Activate.ps1 }
Safe-Run { pip install --upgrade pip }
Safe-Run { pip install -r requirements.txt }

if ($args.Length -gt 0 -and $args[0] -eq '--prefetch') {
    Safe-Run { powershell -ExecutionPolicy Bypass -File fetch_models.ps1 }
}

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate.ps1'"
