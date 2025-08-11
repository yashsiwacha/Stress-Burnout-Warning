#!/bin/bash
# Simple build script for Stress & Burnout Warning System
# Uses the project's virtual environment

echo "🚀 Building Stress & Burnout Warning System Executable..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "📦 Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist *.spec

# Build with PyInstaller
echo "🔨 Creating executable..."
pyinstaller \
    --onefile \
    --windowed \
    --name="StressBurnoutWarning" \
    --hidden-import=cv2 \
    --hidden-import=customtkinter \
    --hidden-import=numpy \
    --hidden-import=pyaudio \
    --hidden-import=sklearn \
    --add-data="models:models" \
    --exclude-module=pytest \
    main.py

# Check if build was successful
if [ -f "dist/StressBurnoutWarning" ]; then
    echo "✅ Build completed successfully!"
    echo "📦 Executable created: dist/StressBurnoutWarning"
    echo "📊 File size: $(du -h dist/StressBurnoutWarning | cut -f1)"
    
    # Make executable
    chmod +x dist/StressBurnoutWarning
    
    echo ""
    echo "🎉 Ready to distribute!"
    echo "Run with: ./dist/StressBurnoutWarning"
else
    echo "❌ Build failed!"
    exit 1
fi
