#!/bin/bash
# Stress Burnout Warning System Launcher for Unix/Linux/macOS
# Run this script to start the application

echo "🧠 Starting Stress Burnout Warning System..."
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run from project root directory."
    exit 1
fi

# Run the application
echo "🚀 Launching application..."
python3 main.py

echo "👋 Application closed."