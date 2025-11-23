@echo off
REM Drone AI Full - Windows Installer
REM
REM Usage:
REM   install.bat              - Basic installation
REM   install.bat --all        - Install with all extras
REM   install.bat --venv       - Create virtual environment
REM

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   Drone AI Full - 4-Layer Autonomous Drone System
echo   Installer for Windows
echo ============================================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"

REM Parse arguments
set USE_VENV=0
set INSTALL_ALL=0
set INSTALL_TRAINING=0
set VENV_PATH=venv

:parse_args
if "%~1"=="" goto :check_python
if "%~1"=="--venv" (
    set USE_VENV=1
    if not "%~2"=="" (
        if not "%~2:~0,2%"=="--" (
            set "VENV_PATH=%~2"
            shift
        )
    )
    shift
    goto :parse_args
)
if "%~1"=="--all" (
    set INSTALL_ALL=1
    shift
    goto :parse_args
)
if "%~1"=="--training" (
    set INSTALL_TRAINING=1
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: install.bat [OPTIONS]
echo.
echo Options:
echo   --venv [PATH]  Create virtual environment (default: .\venv)
echo   --all          Install all optional dependencies
echo   --training     Install training dependencies
echo   -h, --help     Show this help message
exit /b 0

:check_python
echo [*] Checking Python installation...

where python >nul 2>&1
if errorlevel 1 (
    echo [!] Python not found. Please install Python 3.9+
    echo     https://www.python.org/downloads/
    exit /b 1
)

REM Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i

for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo [!] Python 3.9+ required, found %PYTHON_VERSION%
    exit /b 1
)
if %MAJOR%==3 if %MINOR% LSS 9 (
    echo [!] Python 3.9+ required, found %PYTHON_VERSION%
    exit /b 1
)

echo [+] Python %PYTHON_VERSION% OK

REM Create virtual environment if requested
if %USE_VENV%==1 (
    echo [*] Creating virtual environment at %VENV_PATH%...
    python -m venv "%VENV_PATH%"

    REM Activate it
    call "%VENV_PATH%\Scripts\activate.bat"

    REM Upgrade pip
    python -m pip install --upgrade pip

    echo [+] Virtual environment created and activated
)

REM Install PyTorch if needed
if %INSTALL_ALL%==1 goto :install_pytorch
if %INSTALL_TRAINING%==1 goto :install_pytorch
goto :install_package

:install_pytorch
echo [*] Installing PyTorch...

REM Check for NVIDIA GPU
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo     No CUDA detected, installing CPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo     CUDA detected, installing GPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo [+] PyTorch installed

:install_package
echo [*] Installing drone-ai-full package...

if %INSTALL_ALL%==1 (
    pip install -e "%SCRIPT_DIR%[all]"
) else (
    pip install -e "%SCRIPT_DIR%"
)

echo [+] Package installed

:verify
echo [*] Verifying installation...

python -c "import drone_ai; print(f'  Version: {drone_ai.__version__}'); from drone_ai import DroneAI; print('  All imports successful!')"

if errorlevel 1 (
    echo [!] Verification failed
    exit /b 1
)

echo [+] Installation verified

:done
echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.

if %USE_VENV%==1 (
    echo To activate the virtual environment:
    echo   %VENV_PATH%\Scripts\activate.bat
    echo.
)

echo Quick Start:
echo.
echo   # Run the demo
echo   python -m examples.demo
echo.
echo   # Or use the CLI
echo   drone-ai demo --deliveries 3
echo.

exit /b 0
