@echo off
REM Windows batch script for running drift checker as a scheduled task
REM This script ensures proper environment setup and error handling

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Set up Python environment
REM Uncomment and modify if using virtual environment:
REM call "%PROJECT_ROOT%\venv\Scripts\activate.bat"

REM Change to the cron directory
cd /d "%SCRIPT_DIR%"

REM Run the drift checker with Python
python drift_checker.py

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM Log the execution
if %EXIT_CODE% equ 0 (
    echo %date% %time% - Drift checker completed successfully >> cron_execution.log
) else (
    echo %date% %time% - Drift checker failed with exit code: %EXIT_CODE% >> cron_execution.log
)

exit /b %EXIT_CODE%
