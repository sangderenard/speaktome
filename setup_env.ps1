# Windows PowerShell environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

$ErrorActionPreference = 'Stop'

python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

if ($args.Length -gt 0 -and $args[0] -eq '--prefetch') {
    powershell -ExecutionPolicy Bypass -File fetch_models.ps1
}

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate.ps1'"
