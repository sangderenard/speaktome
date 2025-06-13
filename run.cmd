@echo off
setlocal
set SCRIPT_DIR=%~dp0
set "VENV_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Virtual environment not found. See ENV_SETUP_OPTIONS.md.
    exit /b 1
)
for /f "delims=" %%p in ('"%VENV_PY%" -c "import sys; print(sys.executable)"') do set "ACTUAL_PY=%%p"
if /I not "%ACTUAL_PY%"=="%VENV_PY%" (
    echo Error: Script must be run with the virtual environment's Python.
    echo Expected: "%VENV_PY%"
    echo Found:    "%ACTUAL_PY%"
    exit /b 1
)
"%VENV_PY%" -m speaktome.speaktome %*
endlocal
