# 📦 Executable Build System for Stress & Burnout Warning System

## Overview
The application can now be built as a standalone executable using PyInstaller, allowing users to run the app without installing Python or dependencies.

## Build Requirements
- Virtual environment (`.venv`) activated
- PyInstaller installed (`pip install pyinstaller`)
- All project dependencies installed

## Build Scripts

### 1. Quick Build (`build.sh`)
Simple bash script for rapid builds:
```bash
./build.sh
```

### 2. Advanced Build (`build_executable.py`)
Python script with comprehensive packaging:
```bash
python build_executable.py
```

## Build Outputs

### Files Created
- **`dist/StressBurnoutWarning`** - Standalone executable (125MB)
- **`dist/StressBurnoutWarning.app`** - macOS app bundle
- **`build/`** - Temporary build files
- **`StressBurnoutWarning.spec`** - PyInstaller configuration

### Distribution Package
The advanced build script creates:
- Executable file
- User guide
- Launch scripts
- Essential documentation
- Sample configuration files

## Usage

### Running the Executable
```bash
# Direct execution
./dist/StressBurnoutWarning

# Or double-click the .app bundle on macOS
open dist/StressBurnoutWarning.app
```

### Distribution
The executable includes all dependencies and can be distributed to users without Python installed.

## Technical Details

### PyInstaller Configuration
- **Mode**: `--onefile` (single executable)
- **GUI**: `--windowed` (no console window)
- **Includes**: OpenCV, CustomTkinter, NumPy, PyAudio, scikit-learn
- **Size**: ~125MB (includes all ML libraries)

### Hidden Imports
- `cv2` (OpenCV)
- `customtkinter` (GUI framework)
- `numpy` (Numerical computing)
- `pyaudio` (Audio processing)
- `sklearn` (Machine learning)

### Data Files
- `models/` - Pre-trained ML models
- `assets/` - Application assets
- `config/` - Configuration files

## Platform Support
- **macOS**: Native .app bundle + executable
- **Windows**: .exe executable (requires Windows build)
- **Linux**: Native executable (requires Linux build)

## Build Performance
- **Build Time**: ~2-3 minutes
- **File Size**: 125MB
- **Startup Time**: ~3-5 seconds
- **Memory Usage**: ~200MB runtime

## Troubleshooting

### Common Issues
1. **Build Fails**: Ensure virtual environment is activated
2. **Large Size**: Normal for ML applications with all dependencies
3. **Slow Startup**: First launch includes library initialization
4. **macOS Security**: May need to allow in System Preferences

### Solutions
- Use `source .venv/bin/activate` before building
- Consider `--onedir` mode for faster startup
- Sign the app for macOS distribution

## Future Improvements
- Code signing for macOS/Windows
- Automated CI/CD builds for all platforms
- Size optimization techniques
- Faster startup optimizations
- Update mechanism integration
