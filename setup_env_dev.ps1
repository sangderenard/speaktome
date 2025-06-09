# PowerShell developer environment setup script for SpeakToMe

$ErrorActionPreference = 'Stop'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

# Run the regular setup script with forwarded arguments
Safe-Run { powershell -ExecutionPolicy Bypass -File setup_env.ps1 @args }

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment Python not found at $venvPython"
    return
}

Safe-Run { & $venvPython AGENTS/tools/dump_headers.py speaktome --markdown }
Safe-Run { & $venvPython AGENTS/tools/stubfinder.py speaktome }
Safe-Run { & $venvPython AGENTS/tools/list_contributors.py }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py LICENSE }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md }
Safe-Run { & $venvPython AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md }
