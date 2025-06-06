@echo off

REM Navigate to the scripts directory and run install.bat
cd Alpha
call scripts\install.bat
if %errorlevel% neq 0 (
    echo Installation failed. Exiting.
    exit /b %errorlevel%
)

REM Run run.bat after installation is successful
call scripts\run.bat
if %errorlevel% neq 0 (
    echo Running the project failed. Exiting.
    exit /b %errorlevel%
)
cd ..
echo All operations completed successfully.
pause
