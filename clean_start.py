#!/usr/bin/env python3
"""
Clean Startup Script for Stress & Burnout Warning System
Provides a clean startup experience with minimal warnings
"""

import warnings
import os
import sys

def suppress_all_warnings():
    """Suppress common warnings that clutter the startup"""
    
    # Suppress TensorFlow/Protobuf warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', message='.*AVCaptureDeviceTypeExternal.*')
    warnings.filterwarnings('ignore', message='.*CERTIFICATE_VERIFY_FAILED.*')
    
    # Suppress NumPy warnings
    warnings.filterwarnings('ignore', message='.*is not a known BitGenerator.*')

def setup_ssl_for_nltk():
    """Setup SSL context for NLTK downloads"""
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

def clean_startup():
    """Perform clean startup with minimal output"""
    
    # Suppress warnings first
    suppress_all_warnings()
    
    # Setup SSL for NLTK
    setup_ssl_for_nltk()
    
    # Print clean startup message
    print("🧠 MindGuard AI - Stress & Burnout Early Warning System")
    print("🔄 Loading components...")
    
    # Import and run main
    try:
        from main import main
        main()
    except ImportError:
        # Fallback if main import fails
        print("❌ Could not import main application")
        sys.exit(1)

if __name__ == "__main__":
    clean_startup()
