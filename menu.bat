@echo off
REM Drone AI Full - Windows Menu Launcher
REM Simply launches the Python menu

setlocal

REM Get script directory
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Check if Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.9+
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Launch the menu
python "%SCRIPT_DIR%\menu.py"

exit /b 0
