@echo off
rem Stress Burnout Warning System Launcher for Windows
rem Run this script to start the application

echo 🧠 Starting Stress Burnout Warning System...
echo ==============================================

rem Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

rem Check if virtual environment exists
if not exist ".venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

rem Check if main.py exists
if not exist "main.py" (
    echo ❌ Error: main.py not found. Please run from project root directory.
    pause
    exit /b 1
)

rem Run the application
echo 🚀 Launching application...
python main.py

echo 👋 Application closed.
pause