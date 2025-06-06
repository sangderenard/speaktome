@echo off
setlocal
set SCRIPT_DIR=%~dp0
set "VENV_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Virtual environment not found. Run setup_env.ps1 or manually create .venv first.
    exit /b 1
)
for /f "delims=" %%p in ('"%VENV_PY%" -c "import sys; print(sys.executable)"') do set "ACTUAL_PY=%%p"
if /I not "%ACTUAL_PY%"=="%VENV_PY%" (
    echo Warning: expected "%VENV_PY%" but interpreter reports "%ACTUAL_PY%"
)
"%VENV_PY%" -m speaktome.speaktome %*
endlocal
