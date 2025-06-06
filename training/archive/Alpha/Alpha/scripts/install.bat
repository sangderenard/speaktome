@echo off
REM install.bat - Set up the virtual environment and install dependencies for Alpha

echo ===================================================
echo Setting up the Python Virtual Environment...
echo ===================================================

REM Check if venv already exists
IF EXIST venv (
    echo Virtual environment already exists. Skipping creation.
) ELSE (
    echo Creating virtual environment...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Ensure Python is installed and added to PATH.
        pause
        exit /b %ERRORLEVEL%
    )
    echo Virtual environment created successfully.
)

echo.
echo ===================================================
echo Activating the Virtual Environment...
echo ===================================================

REM Activate the virtual environment
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Upgrading pip...
echo ===================================================

REM Upgrade pip to the latest version
python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip.
    pause
    exit /b %ERRORLEVEL%
)
echo Pip upgraded successfully.

echo.
echo ===================================================
echo Installing Dependencies...
echo ===================================================

REM Install required Python packages
pip install -r ..\requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Check requirements.txt for issues.
    pause
    exit /b %ERRORLEVEL%
)
echo Dependencies installed successfully.

echo.
echo ===================================================
echo Installation Complete!
echo ===================================================
echo.
echo To run the game, execute 'run.bat' from the 'scripts' directory.
echo.
pause
