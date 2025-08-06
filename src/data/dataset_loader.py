"""
Dataset Loader Module for Stress Burnout Warning System

This module provides functionality to load, preprocess, and manage datasets
for facial emotion recognition and vocal stress detection.

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import numpy as np
import pandas as pd
import cv2
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DatasetLoader:
    """
    A comprehensive dataset loader for facial and vocal emotion recognition datasets.
    Supports FER-2013, AffectNet, CK+, RAVDESS, SAVEE, and IEMOCAP datasets.
    """
    
    def __init__(self, dataset_type: str = 'facial'):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_type (str): Type of dataset ('facial' or 'vocal')
        """
        self.dataset_type = dataset_type
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
        self.processed_path = os.path.join(self.base_path, 'processed')
        
        # Emotion mappings for different datasets
        self.fer2013_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.ravdess_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Stress-related emotion mapping (for stress detection)
        self.stress_mapping = {
            'angry': 'high_stress',
            'disgusted': 'medium_stress', 
            'fearful': 'high_stress',
            'happy': 'low_stress',
            'neutral': 'low_stress',
            'sad': 'medium_stress',
            'surprised': 'medium_stress',
            'calm': 'low_stress'
        }
        
    def load_fer2013(self, img_size: Tuple[int, int] = (48, 48), 
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess FER-2013 dataset.
        
        Args:
            img_size (tuple): Target image size for preprocessing
            test_size (float): Proportion of dataset for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        fer_path = os.path.join(self.base_path, 'facial_emotion', 'fer2013')
        
        if not os.path.exists(fer_path):
            raise FileNotFoundError(f"FER-2013 dataset not found at {fer_path}")
        
        print("Loading FER-2013 dataset...")
        
        images = []
        labels = []
        
        # Load training data
        train_path = os.path.join(fer_path, 'train')
        if os.path.exists(train_path):
            for emotion_idx, emotion in enumerate(self.fer2013_emotions):
                emotion_path = os.path.join(train_path, emotion)
                if os.path.exists(emotion_path):
                    for img_file in os.listdir(emotion_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(emotion_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = cv2.resize(img, img_size)
                                img = img.astype('float32') / 255.0
                                images.append(img)
                                labels.append(emotion_idx)
        
        # Load test data
        test_path = os.path.join(fer_path, 'test')
        if os.path.exists(test_path):
            for emotion_idx, emotion in enumerate(self.fer2013_emotions):
                emotion_path = os.path.join(test_path, emotion)
                if os.path.exists(emotion_path):
                    for img_file in os.listdir(emotion_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(emotion_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = cv2.resize(img, img_size)
                                img = img.astype('float32') / 255.0
                                images.append(img)
                                labels.append(emotion_idx)
        
        if not images:
            raise ValueError("No images found in FER-2013 dataset")
        
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape for CNN input (add channel dimension)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=len(self.fer2013_emotions))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"FER-2013 loaded: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def load_ravdess(self, sample_rate: int = 22050, 
                     duration: float = 2.5, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess RAVDESS audio dataset.
        
        Args:
            sample_rate (int): Audio sample rate
            duration (float): Duration of audio clips in seconds
            test_size (float): Proportion of dataset for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ravdess_path = os.path.join(self.base_path, 'vocal_emotion', 'ravdess')
        
        if not os.path.exists(ravdess_path):
            raise FileNotFoundError(f"RAVDESS dataset not found at {ravdess_path}")
        
        print("Loading RAVDESS dataset...")
        
        features = []
        labels = []
        
        for audio_file in os.listdir(ravdess_path):
            if audio_file.endswith('.wav'):
                # Parse RAVDESS filename format: Actor_XX-emotion-intensity.wav
                parts = audio_file.split('-')
                if len(parts) >= 3:
                    emotion_code = int(parts[2])
                    
                    # Map RAVDESS emotion codes to emotion names
                    emotion_mapping = {
                        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                    }
                    
                    if emotion_code in emotion_mapping:
                        emotion = emotion_mapping[emotion_code]
                        emotion_idx = self.ravdess_emotions.index(emotion)
                        
                        # Load and process audio
                        audio_path = os.path.join(ravdess_path, audio_file)
                        try:
                            # Load audio file
                            y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
                            
                            # Extract audio features
                            audio_features = self._extract_audio_features(y, sr)
                            
                            features.append(audio_features)
                            labels.append(emotion_idx)
                            
                        except Exception as e:
                            print(f"Error processing {audio_file}: {e}")
                            continue
        
        if not features:
            raise ValueError("No audio files processed from RAVDESS dataset")
        
        X = np.array(features)
        y = np.array(labels)
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=len(self.ravdess_emotions))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"RAVDESS loaded: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract comprehensive audio features for emotion recognition.
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Extracted feature vector
        """
        features = []
        
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1)
        ])
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # 4. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend([
            np.mean(tonnetz, axis=1),
            np.std(tonnetz, axis=1)
        ])
        
        # Flatten and concatenate all features
        feature_vector = np.concatenate([np.array(f).flatten() for f in features])
        
        return feature_vector
    
    def load_stress_labels(self, emotion_labels: np.ndarray, dataset_name: str) -> np.ndarray:
        """
        Convert emotion labels to stress levels for stress detection task.
        
        Args:
            emotion_labels (np.ndarray): Original emotion labels
            dataset_name (str): Name of the dataset ('fer2013' or 'ravdess')
            
        Returns:
            np.ndarray: Stress level labels (0: low, 1: medium, 2: high)
        """
        if dataset_name == 'fer2013':
            emotion_names = self.fer2013_emotions
        elif dataset_name == 'ravdess':
            emotion_names = self.ravdess_emotions
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        stress_labels = []
        stress_mapping_numeric = {'low_stress': 0, 'medium_stress': 1, 'high_stress': 2}
        
        for label in emotion_labels:
            if len(label.shape) > 0:  # If one-hot encoded
                emotion_idx = np.argmax(label)
            else:
                emotion_idx = label
                
            emotion_name = emotion_names[emotion_idx]
            stress_level = self.stress_mapping[emotion_name]
            stress_labels.append(stress_mapping_numeric[stress_level])
        
        return to_categorical(np.array(stress_labels), num_classes=3)
    
    def save_processed_data(self, data: dict, filename: str) -> None:
        """
        Save processed data to disk for faster loading.
        
        Args:
            data (dict): Dictionary containing processed data
            filename (str): Name of the file to save
        """
        os.makedirs(self.processed_path, exist_ok=True)
        
        if self.dataset_type == 'facial':
            save_path = os.path.join(self.processed_path, 'facial', filename)
        else:
            save_path = os.path.join(self.processed_path, 'vocal', filename)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, filename: str) -> dict:
        """
        Load previously processed data from disk.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            dict: Dictionary containing processed data
        """
        if self.dataset_type == 'facial':
            load_path = os.path.join(self.processed_path, 'facial', filename)
        else:
            load_path = os.path.join(self.processed_path, 'vocal', filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Processed data not found at {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Processed data loaded from {load_path}")
        return data


# Example usage and testing
if __name__ == "__main__":
    # Test facial dataset loading
    try:
        facial_loader = DatasetLoader('facial')
        print("Testing FER-2013 dataset loading...")
        # X_train, X_test, y_train, y_test = facial_loader.load_fer2013()
        # print(f"Facial data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    except Exception as e:
        print(f"FER-2013 test failed: {e}")
    
    # Test vocal dataset loading
    try:
        vocal_loader = DatasetLoader('vocal')
        print("Testing RAVDESS dataset loading...")
        # X_train, X_test, y_train, y_test = vocal_loader.load_ravdess()
        # print(f"Audio data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    except Exception as e:
        print(f"RAVDESS test failed: {e}")
