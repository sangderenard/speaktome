# Windows entry point wrapper for SpeakToMe using the local virtual environment.

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host 'Virtual environment not found. See ENV_SETUP_OPTIONS.md.'
    exit 1
}

$actual = & $venvPython -c 'import sys; print(sys.executable)'
$expected = [IO.Path]::GetFullPath($venvPython)
if ($actual -ne $expected) {
    Write-Host "Warning: expected $expected but interpreter reports $actual"
}

& $venvPython -m speaktome.speaktome @args
