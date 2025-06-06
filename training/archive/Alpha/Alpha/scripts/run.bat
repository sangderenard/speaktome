@echo off
REM run.bat - Activate the virtual environment and run the Alpha game

echo ===================================================
echo Activating the Python Virtual Environment...
echo ===================================================


REM Check if venv exists
IF NOT EXIST venv (
    echo Virtual environment not found. Please run 'install.bat' first.
    pause
    exit /b 1
)

REM Activate the virtual environment
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Running the Alpha Game...
echo ===================================================

REM Run the main.py script
python main.py
IF %ERRORLEVEL% NEQ 0 (
    echo The game encountered an error while running.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Game Closed Successfully.
echo ===================================================
echo.
pause
