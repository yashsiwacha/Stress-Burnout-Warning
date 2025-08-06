#!/usr/bin/env python3
"""
Complete Setup Script for Stress & Burnout Warning System
Installs dependencies and configures the environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Core dependencies that are essential
    core_deps = [
        "customtkinter>=5.2.0",
        "numpy>=1.24.0", 
        "Pillow>=10.0.0"
    ]
    
    # Optional dependencies for enhanced features
    optional_deps = [
        "opencv-python>=4.8.0",
        "pyaudio>=0.2.11", 
        "tensorflow>=2.13.0",
        "mediapipe>=0.10.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        success = run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}")
        if not success:
            print(f"⚠️ Failed to install core dependency: {dep}")
    
    # Install optional dependencies (continue even if some fail)
    for dep in optional_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}")

def setup_nltk_data():
    """Setup NLTK data with SSL bypass"""
    print("\n📚 Setting up NLTK data...")
    
    try:
        # Run the NLTK setup script
        success = run_command("python3 setup_nltk.py", "NLTK data setup")
        if success:
            print("✅ NLTK data configured")
        else:
            print("⚠️ NLTK data setup had issues (app will still work)")
    except Exception:
        print("⚠️ Could not run NLTK setup (app will still work)")

def create_launcher_script():
    """Create a clean launcher script"""
    launcher_content = '''#!/usr/bin/env python3
"""
Launch Stress & Burnout Warning System with clean output
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run clean startup
try:
    from clean_start import clean_startup
    clean_startup()
except ImportError:
    # Fallback to main
    try:
        from main import main
        print("🧠 Starting Stress & Burnout Warning System...")
        main()
    except ImportError:
        print("❌ Could not start application")
        sys.exit(1)
'''
    
    try:
        with open('launch.py', 'w') as f:
            f.write(launcher_content)
        print("✅ Created launcher script (launch.py)")
        
        # Make it executable on Unix systems
        if os.name != 'nt':
            os.chmod('launch.py', 0o755)
            
    except Exception as e:
        print(f"⚠️ Could not create launcher: {e}")

def main():
    """Main setup function"""
    print("🚀 Stress & Burnout Warning System - Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    install_dependencies()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Create launcher
    create_launcher_script()
    
    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("\n🚀 To run the application:")
    print("   python3 launch.py")
    print("\n   Or for minimal warnings:")
    print("   python3 clean_start.py")
    print("="*50)

if __name__ == "__main__":
    main()
