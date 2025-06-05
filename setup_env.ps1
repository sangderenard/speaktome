# Windows PowerShell environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

$ErrorActionPreference = 'Stop'

python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Environment setup complete. Activate with '.venv\Scripts\Activate.ps1'"
