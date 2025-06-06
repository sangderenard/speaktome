@echo off
set VENV_PY=.venv\Scripts\python.exe
if not exist %VENV_PY% (
    echo Virtual environment not found. Run setup_env.ps1 or manually create .venv first.
    exit /b 1
)
"%VENV_PY%" -m speaktome.speaktome %*
