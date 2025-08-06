#!/usr/bin/env python3
"""
NLTK Data Setup Script
Fixes SSL certificate issues and downloads required NLTK data
"""

import ssl
import os
import sys

def setup_nltk():
    """Setup NLTK with SSL certificate bypass"""
    
    # Create unverified SSL context (for development/testing)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context
    
    try:
        import nltk
        
        print("📦 Setting up NLTK data...")
        
        # Create NLTK data directory
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required packages
        packages = [
            'vader_lexicon',
            'punkt', 
            'stopwords'
        ]
        
        for package in packages:
            try:
                print(f"📥 Downloading {package}...")
                success = nltk.download(package, quiet=True)
                if success:
                    print(f"✅ {package} downloaded successfully")
                else:
                    print(f"⚠️ Failed to download {package}")
            except Exception as e:
                print(f"⚠️ Error downloading {package}: {e}")
        
        print("✅ NLTK setup complete")
        return True
        
    except ImportError:
        print("⚠️ NLTK not installed")
        print("💡 Install with: pip install nltk")
        return False
    except Exception as e:
        print(f"❌ NLTK setup error: {e}")
        return False

if __name__ == "__main__":
    setup_nltk()
