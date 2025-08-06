#!/usr/bin/env python3
"""
Setup Script for Stress Burnout Warning System

This script sets up the complete development environment including:
- Installing dependencies
- Setting up datasets directory structure  
- Verifying system requirements
- Configuring initial models

Usage:
    python setup_project.py [options]

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print("Please upgrade Python and try again.")
        return False
    
    print("✅ Python version is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print_header("INSTALLING DEPENDENCIES")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found!")
        return False
    
    try:
        print("Installing packages from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def setup_directories():
    """Create necessary directory structure."""
    print_header("SETTING UP DIRECTORY STRUCTURE")
    
    directories = [
        "datasets",
        "datasets/facial_emotion",
        "datasets/facial_emotion/fer2013",
        "datasets/facial_emotion/affectnet", 
        "datasets/facial_emotion/ck_plus",
        "datasets/vocal_emotion",
        "datasets/vocal_emotion/ravdess",
        "datasets/vocal_emotion/savee",
        "datasets/vocal_emotion/iemocap",
        "datasets/custom",
        "datasets/processed",
        "datasets/processed/facial",
        "datasets/processed/vocal",
        "models",
        "logs",
        "data",
        "config"
    ]
    
    created_dirs = []
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"✅ Created/verified: {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}/: {e}")
    
    print(f"\n📁 Created {len(created_dirs)} directories")
    return True


def verify_system_requirements():
    """Verify system-specific requirements."""
    print_header("VERIFYING SYSTEM REQUIREMENTS")
    
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Check for system-specific requirements
    if system == "Darwin":  # macOS
        print("🍎 macOS detected")
        print("Note: You may need to install portaudio for PyAudio:")
        print("  brew install portaudio")
    elif system == "Linux":
        print("🐧 Linux detected")
        print("Note: You may need to install system packages:")
        print("  sudo apt-get install portaudio19-dev python3-pyaudio")
    elif system == "Windows":
        print("🪟 Windows detected")
        print("Note: PyAudio should install automatically")
    
    # Check for essential imports
    essential_packages = [
        "numpy",
        "opencv-python", 
        "sklearn",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in essential_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All essential packages are available")
    return True


def check_optional_dependencies():
    """Check for optional dependencies."""
    print_header("CHECKING OPTIONAL DEPENDENCIES")
    
    optional_packages = {
        "tensorflow": "Deep learning models (CNN, LSTM)",
        "librosa": "Advanced audio processing",
        "kaggle": "Dataset downloading",
        "mediapipe": "Face landmark detection"
    }
    
    available_features = []
    missing_features = []
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
            available_features.append(package)
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_features.append(package)
    
    print(f"\n📊 Available features: {len(available_features)}/{len(optional_packages)}")
    
    if missing_features:
        print("\n⚠️  Some features will be limited due to missing packages:")
        for package in missing_features:
            print(f"  - {package}: {optional_packages[package]}")
        print("\nInstall with: pip install " + " ".join(missing_features))
    
    return True


def create_initial_config():
    """Create initial configuration files."""
    print_header("CREATING INITIAL CONFIGURATION")
    
    # Create config file
    config_content = '''# Stress Burnout Warning System Configuration
# Generated automatically by setup_project.py

# Model Configuration
[models]
facial_model_path = models/facial_emotion_cnn_final.h5
vocal_model_path = models/vocal_emotion_lstm_final.h5
stress_model_path = models/stress_detection_rules.pkl

# Training Configuration
[training]
batch_size = 32
epochs = 50
learning_rate = 0.001

# Data Configuration
[data]
datasets_path = datasets/
processed_data_path = datasets/processed/
models_path = models/

# System Configuration
[system]
camera_index = 0
microphone_index = 0
sample_rate = 22050
frame_rate = 30
'''
    
    config_path = "config/system_config.ini"
    try:
        os.makedirs("config", exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✅ Created configuration file: {config_path}")
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
    
    # Create training config
    training_config = '''{
    "facial_models": {
        "enabled": true,
        "architectures": ["custom"],
        "epochs": 20,
        "batch_size": 32,
        "use_transfer_learning": false,
        "dataset": "fer2013"
    },
    "vocal_models": {
        "enabled": true,
        "epochs": 20,
        "batch_size": 32,
        "dataset": "ravdess"
    },
    "stress_models": {
        "enabled": true,
        "fusion_enabled": false
    },
    "evaluation": {
        "cross_validation": false,
        "generate_plots": true,
        "save_predictions": true
    },
    "logging": {
        "verbose": true,
        "save_logs": true
    }
}'''
    
    training_config_path = "config/training_config.json"
    try:
        with open(training_config_path, 'w') as f:
            f.write(training_config)
        print(f"✅ Created training configuration: {training_config_path}")
    except Exception as e:
        print(f"❌ Failed to create training config: {e}")
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print_header("NEXT STEPS")
    
    print("🎉 Setup completed successfully!")
    print("\nTo get started:")
    print()
    print("1. Download datasets:")
    print("   python dataset_downloader.py --info")
    print("   python dataset_downloader.py --download-all")
    print()
    print("2. Or create sample data for testing:")
    print("   python dataset_downloader.py --create-sample")
    print()
    print("3. Train models:")
    print("   python train_complete_system.py")
    print()
    print("4. Run the application:")
    print("   python main.py")
    print()
    print("📁 Project structure:")
    print("   datasets/     - Training datasets")
    print("   models/       - Trained models")
    print("   src/          - Source code")
    print("   config/       - Configuration files")
    print("   logs/         - Training logs")
    print()
    print("📖 For more information, see:")
    print("   README.md")
    print("   datasets/README.md")


def main():
    """Main setup function."""
    print("STRESS BURNOUT WARNING SYSTEM - PROJECT SETUP")
    print("=" * 60)
    print("This script will set up your development environment.")
    print()
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies), 
        ("Setting up directories", setup_directories),
        ("Verifying system requirements", verify_system_requirements),
        ("Checking optional dependencies", check_optional_dependencies),
        ("Creating initial configuration", create_initial_config)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        try:
            if not step_function():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    if failed_steps:
        print_header("SETUP COMPLETED WITH WARNINGS")
        print("⚠️  Some steps failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nYou may need to address these issues manually.")
    else:
        print_header("SETUP COMPLETED SUCCESSFULLY")
    
    print_next_steps()


if __name__ == "__main__":
    main()
