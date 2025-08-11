#!/usr/bin/env python3
"""
Build Script for Stress & Burnout Warning System
Creates standalone executable for distribution
Uses the project's virtual environment
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def ensure_venv_activated():
    """Ensure we're running in the virtual environment"""
    venv_path = Path(__file__).parent / '.venv'
    
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Please create it first with: python -m venv .venv")
        sys.exit(1)
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("Please activate it first with: source .venv/bin/activate")
        return False

def main():
    """Main build process"""
    print("🚀 Building Stress & Burnout Warning System Executable...")
    
    # Ensure virtual environment
    if not ensure_venv_activated():
        print("💡 Attempting to use virtual environment python...")
        venv_python = Path(__file__).parent / '.venv' / 'bin' / 'python'
        if venv_python.exists():
            # Re-run this script with virtual environment python
            subprocess.run([str(venv_python), __file__], check=True)
            return True
        else:
            sys.exit(1)
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    build_dirs = ['build', 'dist', '__pycache__']
    for dir_name in build_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_name}/")
    
    # Remove .spec files
    for spec_file in project_root.glob("*.spec"):
        spec_file.unlink()
        print(f"   Removed {spec_file.name}")
    
    print("✅ Cleanup completed")
    
    # PyInstaller command
    pyinstaller_args = [
        'pyinstaller',
        '--onefile',                    # Single executable file
        '--windowed',                   # No console window (GUI app)
        '--name=StressBurnoutWarning',  # Executable name
        '--icon=assets/icon.ico',       # App icon (if exists)
        '--add-data=models:models',     # Include ML models
        '--add-data=assets:assets',     # Include assets
        '--add-data=config:config',     # Include config files
        '--hidden-import=cv2',          # Ensure OpenCV is included
        '--hidden-import=customtkinter', # Ensure CustomTkinter is included
        '--hidden-import=numpy',        # Ensure NumPy is included
        '--hidden-import=tensorflow',   # Ensure TensorFlow is included (if used)
        '--hidden-import=pyaudio',      # Ensure PyAudio is included
        '--hidden-import=sklearn',      # Ensure scikit-learn is included
        '--exclude-module=pytest',     # Exclude test modules
        '--exclude-module=sphinx',     # Exclude documentation modules
        'main.py'                      # Main application file
    ]
    
    print("🔨 Building executable...")
    print(f"Command: {' '.join(pyinstaller_args)}")
    
    # Run PyInstaller
    try:
        result = subprocess.run(pyinstaller_args, cwd=project_root, check=True, 
                              capture_output=True, text=True)
        print("✅ Build completed successfully!")
        
        # Check if executable was created
        exe_path = project_root / 'dist' / 'StressBurnoutWarning'
        if sys.platform == 'win32':
            exe_path = exe_path.with_suffix('.exe')
        
        if exe_path.exists():
            print(f"📦 Executable created: {exe_path}")
            print(f"📊 File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            print("❌ Executable not found in expected location")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    # Create distribution package
    print("\n📦 Creating distribution package...")
    create_distribution_package(project_root)
    
    return True

def create_distribution_package(project_root):
    """Create a distribution package with executable and resources"""
    
    dist_dir = project_root / 'dist'
    package_dir = dist_dir / 'StressBurnoutWarning_Package'
    
    # Create package directory
    package_dir.mkdir(exist_ok=True)
    
    # Copy executable
    exe_name = 'StressBurnoutWarning'
    if sys.platform == 'win32':
        exe_name += '.exe'
    
    exe_src = dist_dir / exe_name
    exe_dst = package_dir / exe_name
    
    if exe_src.exists():
        shutil.copy2(exe_src, exe_dst)
        print(f"   Copied {exe_name}")
    
    # Copy essential files
    essential_files = [
        'README.md',
        'LICENSE',
        'requirements.txt'
    ]
    
    for file_name in essential_files:
        src_file = project_root / file_name
        if src_file.exists():
            shutil.copy2(src_file, package_dir / file_name)
            print(f"   Copied {file_name}")
    
    # Copy sample data (if exists)
    sample_dirs = ['data', 'config']
    for dir_name in sample_dirs:
        src_dir = project_root / dir_name
        if src_dir.exists():
            dst_dir = package_dir / dir_name
            shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
            print(f"   Copied {dir_name}/ directory")
    
    # Create launch script
    create_launch_script(package_dir, exe_name)
    
    # Create user guide
    create_user_guide(package_dir)
    
    print(f"✅ Distribution package created: {package_dir}")

def create_launch_script(package_dir, exe_name):
    """Create platform-specific launch script"""
    
    if sys.platform == 'win32':
        # Windows batch script
        script_content = f"""@echo off
echo Starting Stress & Burnout Warning System...
echo.
"{exe_name}"
if errorlevel 1 (
    echo.
    echo Application encountered an error.
    echo Please check the error message above.
    pause
)
"""
        script_path = package_dir / 'Launch.bat'
    else:
        # Unix/macOS shell script
        script_content = f"""#!/bin/bash
echo "Starting Stress & Burnout Warning System..."
echo
cd "$(dirname "$0")"
./{exe_name}
if [ $? -ne 0 ]; then
    echo
    echo "Application encountered an error."
    echo "Please check the error message above."
    read -p "Press Enter to continue..."
fi
"""
        script_path = package_dir / 'Launch.sh'
    
    script_path.write_text(script_content)
    
    # Make script executable on Unix systems
    if sys.platform != 'win32':
        os.chmod(script_path, 0o755)
    
    print(f"   Created launch script: {script_path.name}")

def create_user_guide(package_dir):
    """Create a user guide for the executable"""
    
    guide_content = """# Stress & Burnout Warning System - User Guide

## Quick Start

1. **Run the Application**:
   - Windows: Double-click `Launch.bat` or `StressBurnoutWarning.exe`
   - macOS/Linux: Double-click `Launch.sh` or run `./StressBurnoutWarning`

2. **First Time Setup**:
   - Grant camera and microphone permissions when prompted
   - The application will start monitoring automatically
   - All data is processed locally for privacy

## Features

- **Real-time Stress Monitoring**: Uses your camera and microphone
- **Privacy First**: All processing happens on your device
- **Smart Alerts**: Get notified when stress levels are elevated
- **Wellness Tools**: Built-in breathing exercises and break reminders

## System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Camera**: Built-in or external webcam
- **Microphone**: Built-in or external microphone

## Troubleshooting

### Camera Issues
- Ensure no other applications are using the camera
- Check camera permissions in system settings
- Try restarting the application

### Performance Issues
- Close other resource-intensive applications
- Ensure good lighting for camera detection
- Check system meets minimum requirements

### Application Won't Start
- Try running as administrator (Windows) or with sudo (Linux/macOS)
- Check antivirus isn't blocking the application
- Ensure all files in the package are present

## Support

For support and updates, visit: https://github.com/yashsiwacha/Stress-Burnout-Warning

## Privacy Policy

This application:
- Processes all data locally on your device
- Does not send any personal data to external servers
- Does not store biometric data permanently
- Respects your privacy and data security

---

Version 2.0 | Built with ❤️ for mental wellness
"""
    
    guide_path = package_dir / 'USER_GUIDE.md'
    guide_path.write_text(guide_content)
    print(f"   Created user guide: {guide_path.name}")

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
