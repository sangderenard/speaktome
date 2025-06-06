# Windows entry point wrapper for SpeakToMe using the local virtual environment.

$ErrorActionPreference = 'Stop'

$venvPython = '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host 'Virtual environment not found. Run setup_env.ps1 first.'
    exit 1
}

& $venvPython -m speaktome.speaktome @args
