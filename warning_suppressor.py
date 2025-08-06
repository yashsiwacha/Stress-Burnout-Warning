#!/usr/bin/env python3
"""
Warning Suppressor and Environment Setup
Handles various warnings and improves startup experience
"""

import warnings
import os
import sys
import ssl

def suppress_warnings():
    """Suppress non-critical warnings to improve user experience"""
    
    # Suppress TensorFlow/Protobuf warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    
    # Suppress other common warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress TensorFlow info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Set environment variables to reduce TensorFlow verbosity
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def setup_ssl_context():
    """Setup SSL context to handle certificate issues"""
    try:
        # Create unverified SSL context for NLTK downloads
        ssl._create_default_https_context = ssl._create_unverified_context
        print("🔧 SSL context configured for NLTK downloads")
    except Exception as e:
        print(f"⚠️ Could not configure SSL context: {e}")

def download_nltk_data():
    """Download required NLTK data with improved error handling"""
    try:
        import nltk
        
        # Set download directory
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Required NLTK packages
        packages = ['vader_lexicon', 'punkt', 'stopwords']
        
        print("📦 Downloading NLTK data packages...")
        
        for package in packages:
            try:
                # Check if package exists
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else 
                              f'corpora/{package}' if package == 'stopwords' else 
                              f'vader_lexicon/{package}')
                print(f"✅ {package} already available")
            except LookupError:
                try:
                    print(f"📥 Downloading {package}...")
                    result = nltk.download(package, quiet=True)
                    if result:
                        print(f"✅ {package} downloaded successfully")
                    else:
                        print(f"⚠️ Failed to download {package}")
                except Exception as e:
                    print(f"⚠️ Error downloading {package}: {e}")
        
        print("📦 NLTK data setup complete")
        return True
        
    except ImportError:
        print("⚠️ NLTK not available - skipping data download")
        return False
    except Exception as e:
        print(f"⚠️ Error setting up NLTK data: {e}")
        return False

def check_dependencies():
    """Check and report on optional dependencies"""
    dependencies = {
        'customtkinter': 'Modern UI framework',
        'opencv-cv2': 'Camera monitoring',
        'pyaudio': 'Microphone monitoring', 
        'mediapipe': 'Advanced facial analysis',
        'tensorflow': 'ML models',
        'nltk': 'Natural language processing',
        'numpy': 'Numerical computing'
    }
    
    available = []
    missing = []
    
    for dep, description in dependencies.items():
        try:
            if dep == 'opencv-cv2':
                import cv2
            elif dep == 'pyaudio':
                import pyaudio
            elif dep == 'mediapipe':
                import mediapipe
            elif dep == 'tensorflow':
                import tensorflow
            elif dep == 'nltk':
                import nltk
            elif dep == 'numpy':
                import numpy
            elif dep == 'customtkinter':
                import customtkinter
            
            available.append((dep, description))
        except ImportError:
            missing.append((dep, description))
    
    return available, missing

def print_dependency_status():
    """Print a clean dependency status report"""
    available, missing = check_dependencies()
    
    print("\n🔍 Dependency Status Report:")
    print("="*50)
    
    if available:
        print("✅ Available Dependencies:")
        for dep, desc in available:
            print(f"   • {dep}: {desc}")
    
    if missing:
        print("\n⚠️ Missing Optional Dependencies:")
        for dep, desc in missing:
            print(f"   • {dep}: {desc}")
        print("\n💡 Install missing dependencies with:")
        print("   pip install opencv-python pyaudio mediapipe tensorflow nltk numpy")
    
    print("="*50)

def initialize_environment():
    """Initialize the environment with proper warning suppression and setup"""
    print("🚀 Initializing environment...")
    
    # Suppress warnings first
    suppress_warnings()
    
    # Setup SSL context
    setup_ssl_context()
    
    # Print dependency status
    print_dependency_status()
    
    # Download NLTK data
    download_nltk_data()
    
    print("✅ Environment initialization complete")

if __name__ == "__main__":
    initialize_environment()
