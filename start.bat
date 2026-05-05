@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" goto :help
if "%~1"=="help" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found. Install Python 3.10+ and add it to PATH.
    exit /b 1
)

python -m orchestrator.main %*
exit /b %errorlevel%

:help
echo Aura Agent Windows helper
echo.
echo Usage:
echo   start.bat start --task-file tasks/example_mission.md
echo   start.bat status
echo   start.bat progress
echo   start.bat wake
echo.
echo You can also install the package and use the global command:
echo   python -m pip install -e .
echo   aura setup
echo   aura start --task-file tasks/example_mission.md
exit /b 0
