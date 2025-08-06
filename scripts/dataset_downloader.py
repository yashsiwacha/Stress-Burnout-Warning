#!/usr/bin/env python3
"""
Dataset Download and Setup Script for Stress Burnout Warning System

This script helps download and organize the required datasets for training
the facial emotion recognition and vocal stress detection models.

Supported Datasets:
- FER-2013 (Facial Expression Recognition)
- RAVDESS (Audio-Visual Emotional Speech)
- Custom data collection utilities

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import requests
import zipfile
import tarfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class DatasetDownloader:
    """
    Automated dataset downloader and organizer.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the dataset downloader.
        
        Args:
            base_path (str): Base path for datasets
        """
        if base_path is None:
            self.base_path = os.path.join(os.path.dirname(__file__), '..', 'datasets')
        else:
            self.base_path = base_path
        
        os.makedirs(self.base_path, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'fer2013': {
                'name': 'FER-2013',
                'description': 'Facial Expression Recognition 2013 dataset',
                'size': '~96 MB',
                'samples': '35,887 images',
                'emotions': ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'],
                'download_url': 'https://www.kaggle.com/datasets/msambare/fer2013',
                'local_path': os.path.join(self.base_path, 'facial_emotion', 'fer2013'),
                'format': 'directory structure',
                'requires_kaggle': True
            },
            'ravdess': {
                'name': 'RAVDESS',
                'description': 'Ryerson Audio-Visual Database of Emotional Speech and Song',
                'size': '~24 GB',
                'samples': '7,356 files (24 actors)',
                'emotions': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
                'download_url': 'https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio',
                'local_path': os.path.join(self.base_path, 'vocal_emotion', 'ravdess'),
                'format': 'audio files (.wav)',
                'requires_kaggle': True
            },
            'affectnet': {
                'name': 'AffectNet',
                'description': 'Large-scale facial expression dataset',
                'size': '~4 GB (manual annotations)',
                'samples': '1M+ images',
                'emotions': ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
                'download_url': 'http://mohammadmahoor.com/affectnet/',
                'local_path': os.path.join(self.base_path, 'facial_emotion', 'affectnet'),
                'format': 'images with annotations',
                'requires_registration': True
            }
        }
    
    def show_dataset_info(self) -> None:
        """
        Display information about available datasets.
        """
        print("AVAILABLE DATASETS FOR STRESS DETECTION TRAINING")
        print("=" * 60)
        
        for dataset_id, config in self.dataset_configs.items():
            print(f"\n{config['name']} ({dataset_id.upper()})")
            print("-" * 40)
            print(f"Description: {config['description']}")
            print(f"Size: {config['size']}")
            print(f"Samples: {config['samples']}")
            print(f"Emotions: {', '.join(config['emotions'])}")
            print(f"Download URL: {config['download_url']}")
            print(f"Local Path: {config['local_path']}")
            print(f"Format: {config['format']}")
            
            if config.get('requires_kaggle'):
                print("⚠️  Requires Kaggle account and API key")
            if config.get('requires_registration'):
                print("⚠️  Requires manual registration")
            
            # Check if dataset exists
            if os.path.exists(config['local_path']) and os.listdir(config['local_path']):
                print("✅ Dataset found locally")
            else:
                print("❌ Dataset not found locally")
    
    def setup_kaggle_api(self) -> bool:
        """
        Setup Kaggle API for dataset downloads.
        
        Returns:
            bool: True if setup successful
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize API
            api = KaggleApi()
            api.authenticate()
            
            print("✅ Kaggle API configured successfully")
            return True
            
        except ImportError:
            print("❌ Kaggle package not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Kaggle API setup failed: {e}")
            print("\nTo setup Kaggle API:")
            print("1. Create account at https://www.kaggle.com")
            print("2. Go to Account settings and create API token")
            print("3. Download kaggle.json and place in ~/.kaggle/")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_fer2013(self) -> bool:
        """
        Download FER-2013 dataset from Kaggle.
        
        Returns:
            bool: True if download successful
        """
        print("Downloading FER-2013 dataset...")
        
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            
            dataset_path = self.dataset_configs['fer2013']['local_path']
            os.makedirs(dataset_path, exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(
                'msambare/fer2013',
                path=dataset_path,
                unzip=True
            )
            
            print(f"✅ FER-2013 downloaded to {dataset_path}")
            
            # Verify download
            if self.verify_fer2013_structure(dataset_path):
                print("✅ FER-2013 structure verified")
                return True
            else:
                print("⚠️  FER-2013 structure verification failed")
                return False
                
        except Exception as e:
            print(f"❌ FER-2013 download failed: {e}")
            return False
    
    def download_ravdess(self) -> bool:
        """
        Download RAVDESS dataset from Kaggle.
        
        Returns:
            bool: True if download successful
        """
        print("Downloading RAVDESS dataset...")
        
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            
            dataset_path = self.dataset_configs['ravdess']['local_path']
            os.makedirs(dataset_path, exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(
                'uwrfkaggle/ravdess-emotional-speech-audio',
                path=dataset_path,
                unzip=True
            )
            
            print(f"✅ RAVDESS downloaded to {dataset_path}")
            
            # Verify download
            if self.verify_ravdess_structure(dataset_path):
                print("✅ RAVDESS structure verified")
                return True
            else:
                print("⚠️  RAVDESS structure verification failed")
                return False
                
        except Exception as e:
            print(f"❌ RAVDESS download failed: {e}")
            return False
    
    def verify_fer2013_structure(self, path: str) -> bool:
        """
        Verify FER-2013 dataset structure.
        
        Args:
            path (str): Dataset path
            
        Returns:
            bool: True if structure is correct
        """
        required_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Check for train and test directories
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print("Missing train or test directories")
            return False
        
        # Check emotion subdirectories
        for split in ['train', 'test']:
            split_path = os.path.join(path, split)
            for emotion in required_emotions:
                emotion_path = os.path.join(split_path, emotion)
                if not os.path.exists(emotion_path):
                    print(f"Missing emotion directory: {emotion_path}")
                    return False
                
                # Check for images
                images = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) == 0:
                    print(f"No images found in {emotion_path}")
                    return False
        
        return True
    
    def verify_ravdess_structure(self, path: str) -> bool:
        """
        Verify RAVDESS dataset structure.
        
        Args:
            path (str): Dataset path
            
        Returns:
            bool: True if structure is correct
        """
        # Check for audio files
        audio_files = [f for f in os.listdir(path) if f.endswith('.wav')]
        
        if len(audio_files) == 0:
            print("No audio files found in RAVDESS directory")
            return False
        
        # Check filename format (Actor_XX-emotion-intensity.wav)
        valid_files = 0
        for filename in audio_files:
            parts = filename.split('-')
            if len(parts) >= 3 and filename.startswith('Actor_'):
                valid_files += 1
        
        if valid_files < len(audio_files) * 0.8:  # At least 80% should have correct format
            print("Invalid RAVDESS filename format detected")
            return False
        
        return True
    
    def create_sample_dataset(self) -> bool:
        """
        Create a small sample dataset for testing.
        
        Returns:
            bool: True if sample dataset created
        """
        print("Creating sample dataset for testing...")
        
        sample_path = os.path.join(self.base_path, 'sample')
        os.makedirs(sample_path, exist_ok=True)
        
        # Create sample facial emotion data
        facial_sample_path = os.path.join(sample_path, 'facial_emotion', 'fer2013')
        os.makedirs(facial_sample_path, exist_ok=True)
        
        emotions = ['happy', 'sad', 'angry', 'neutral']
        for split in ['train', 'test']:
            for emotion in emotions:
                emotion_dir = os.path.join(facial_sample_path, split, emotion)
                os.makedirs(emotion_dir, exist_ok=True)
                
                # Create dummy image files
                for i in range(5):  # 5 samples per emotion
                    import numpy as np
                    import cv2
                    
                    # Create random 48x48 grayscale image
                    img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
                    img_path = os.path.join(emotion_dir, f'sample_{i}.jpg')
                    cv2.imwrite(img_path, img)
        
        # Create sample audio data
        audio_sample_path = os.path.join(sample_path, 'vocal_emotion', 'ravdess')
        os.makedirs(audio_sample_path, exist_ok=True)
        
        # Create dummy audio files (will need actual audio library for real files)
        for actor in range(1, 3):  # 2 actors
            for emotion in range(1, 5):  # 4 emotions
                filename = f'Actor_{actor:02d}-01-{emotion:02d}-01-01-01-24.wav'
                filepath = os.path.join(audio_sample_path, filename)
                
                # Create empty file (placeholder)
                with open(filepath, 'wb') as f:
                    f.write(b'dummy_audio_data')
        
        print(f"✅ Sample dataset created at {sample_path}")
        return True
    
    def download_all(self) -> Dict[str, bool]:
        """
        Download all available datasets.
        
        Returns:
            dict: Download results for each dataset
        """
        results = {}
        
        print("DOWNLOADING ALL DATASETS")
        print("=" * 40)
        
        # Download FER-2013
        print("\n1. Downloading FER-2013...")
        results['fer2013'] = self.download_fer2013()
        
        # Download RAVDESS
        print("\n2. Downloading RAVDESS...")
        results['ravdess'] = self.download_ravdess()
        
        # Create sample dataset
        print("\n3. Creating sample dataset...")
        results['sample'] = self.create_sample_dataset()
        
        # Summary
        print("\n" + "=" * 40)
        print("DOWNLOAD SUMMARY")
        print("=" * 40)
        
        for dataset, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{dataset.upper()}: {status}")
        
        return results
    
    def setup_instructions(self) -> None:
        """
        Print detailed setup instructions.
        """
        print("DATASET SETUP INSTRUCTIONS")
        print("=" * 50)
        
        print("\n1. AUTOMATIC DOWNLOAD (Recommended)")
        print("-" * 30)
        print("For FER-2013 and RAVDESS datasets:")
        print("   a. Install kaggle: pip install kaggle")
        print("   b. Create Kaggle account: https://www.kaggle.com")
        print("   c. Generate API token in Account settings")
        print("   d. Download kaggle.json to ~/.kaggle/")
        print("   e. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("   f. Run: python dataset_downloader.py --download-all")
        
        print("\n2. MANUAL DOWNLOAD")
        print("-" * 30)
        print("FER-2013:")
        print("   URL: https://www.kaggle.com/datasets/msambare/fer2013")
        print("   Extract to: datasets/facial_emotion/fer2013/")
        print("\nRAVDESS:")
        print("   URL: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio")
        print("   Extract to: datasets/vocal_emotion/ravdess/")
        
        print("\n3. SAMPLE DATASET")
        print("-" * 30)
        print("For quick testing:")
        print("   Run: python dataset_downloader.py --create-sample")
        
        print("\n4. VERIFICATION")
        print("-" * 30)
        print("Verify dataset structure:")
        print("   Run: python dataset_downloader.py --verify")


def main():
    """
    Main function for dataset download script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for Stress Burnout Warning System')
    parser.add_argument('--info', action='store_true', help='Show dataset information')
    parser.add_argument('--download-all', action='store_true', help='Download all datasets')
    parser.add_argument('--download-fer2013', action='store_true', help='Download FER-2013 dataset')
    parser.add_argument('--download-ravdess', action='store_true', help='Download RAVDESS dataset')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--verify', action='store_true', help='Verify dataset structure')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.info:
        downloader.show_dataset_info()
    elif args.download_all:
        downloader.download_all()
    elif args.download_fer2013:
        downloader.download_fer2013()
    elif args.download_ravdess:
        downloader.download_ravdess()
    elif args.create_sample:
        downloader.create_sample_dataset()
    elif args.verify:
        # Verify all datasets
        for dataset_id, config in downloader.dataset_configs.items():
            path = config['local_path']
            if os.path.exists(path):
                if dataset_id == 'fer2013':
                    valid = downloader.verify_fer2013_structure(path)
                elif dataset_id == 'ravdess':
                    valid = downloader.verify_ravdess_structure(path)
                else:
                    valid = os.path.exists(path)
                
                status = "✅ Valid" if valid else "❌ Invalid"
                print(f"{config['name']}: {status}")
            else:
                print(f"{config['name']}: ❌ Not found")
    elif args.setup:
        downloader.setup_instructions()
    else:
        print("Use --help for available options")
        downloader.setup_instructions()


if __name__ == "__main__":
    main()
