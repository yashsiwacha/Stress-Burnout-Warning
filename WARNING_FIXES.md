# Warning Resolution Guide

## Quick Fix for Clean Startup

**For the cleanest experience with minimal warnings:**

```bash
python3 clean_start.py
```

## Complete Setup (Resolves All Warnings)

**1. Run the complete setup:**
```bash
python3 setup_complete.py
```

**2. Or install dependencies manually:**
```bash
pip install -r requirements.txt
python3 setup_nltk.py
```

## Warning Categories Explained

### 🟢 Resolved by Clean Startup
- ✅ TensorFlow/Protobuf version warnings (suppressed)
- ✅ SSL certificate warnings (bypassed for NLTK)
- ✅ Deprecation warnings (filtered out)

### 🟡 Resolved by Installing Dependencies  
- 📦 `MediaPipe not available` → `pip install mediapipe`
- 📦 `ML facial models not available` → `pip install tensorflow`
- 📦 `Typing monitoring not available` → Advanced modules
- 📦 `Alert system not available` → Advanced modules

### 🔵 NLTK SSL Issues
- 🔧 Run `python3 setup_nltk.py` to fix
- 🔧 Or use the complete setup script

### 🟠 macOS Camera Warning
- ℹ️ This is a system warning from macOS
- ℹ️ App functionality is not affected
- ℹ️ Can be ignored safely

## Startup Options

| Script | Description | Warnings Level |
|--------|-------------|----------------|
| `clean_start.py` | Cleanest startup | Minimal ⭐⭐⭐⭐⭐ |
| `launch.py` | User-friendly | Low ⭐⭐⭐⭐ |
| `main.py` | Standard | Medium ⭐⭐⭐ |

## Dependencies Status

**Core (Required):**
- ✅ CustomTkinter - Modern UI
- ✅ NumPy - Scientific computing
- ✅ Pillow - Image processing

**Enhanced Features (Optional):**
- 📷 OpenCV - Camera monitoring
- 🎤 PyAudio - Microphone monitoring  
- 🧠 TensorFlow - ML models
- 👁️ MediaPipe - Facial analysis
- 📝 NLTK - Text analysis

**All warnings are cosmetic and don't affect functionality!**
