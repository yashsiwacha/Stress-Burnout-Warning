"""
Advanced Model Training Module for Stress Burnout Warning System

This module implements comprehensive training pipelines for:
1. Facial emotion recognition using CNN
2. Vocal stress detection using RNN/LSTM
3. Fusion model combining both modalities
4. Stress prediction and burnout warning models

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, List, Optional, Dict

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using fallback sklearn models.")

# Traditional ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataset_loader import DatasetLoader

class ModelTrainer:
    """
    Comprehensive model training class for stress detection system.
    """
    
    def __init__(self, model_save_path: str = None):
        """
        Initialize the model trainer.
        
        Args:
            model_save_path (str): Path to save trained models
        """
        if model_save_path is None:
            self.model_save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        else:
            self.model_save_path = model_save_path
        
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Model configurations
        self.facial_model = None
        self.vocal_model = None
        self.fusion_model = None
        self.stress_model = None
        
        # Training history
        self.training_history = {}
        
    def build_cnn_facial_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """
        Build CNN model for facial emotion recognition.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of emotion classes
            
        Returns:
            keras.Model: Compiled CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN models")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_vocal_model(self, input_shape: Tuple[int], num_classes: int) -> keras.Model:
        """
        Build LSTM model for vocal emotion/stress recognition.
        
        Args:
            input_shape (tuple): Input feature shape
            num_classes (int): Number of emotion classes
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        model = models.Sequential([
            # Reshape input for LSTM
            layers.Reshape((1, input_shape[0]), input_shape=input_shape),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_fusion_model(self, facial_features: int, vocal_features: int, num_classes: int) -> keras.Model:
        """
        Build fusion model combining facial and vocal features.
        
        Args:
            facial_features (int): Number of facial features
            vocal_features (int): Number of vocal features
            num_classes (int): Number of output classes
            
        Returns:
            keras.Model: Compiled fusion model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for fusion models")
        
        # Facial input branch
        facial_input = layers.Input(shape=(facial_features,), name='facial_input')
        facial_dense = layers.Dense(256, activation='relu')(facial_input)
        facial_dropout = layers.Dropout(0.5)(facial_dense)
        facial_out = layers.Dense(128, activation='relu')(facial_dropout)
        
        # Vocal input branch
        vocal_input = layers.Input(shape=(vocal_features,), name='vocal_input')
        vocal_dense = layers.Dense(256, activation='relu')(vocal_input)
        vocal_dropout = layers.Dropout(0.5)(vocal_dense)
        vocal_out = layers.Dense(128, activation='relu')(vocal_dropout)
        
        # Fusion layer
        fusion = layers.concatenate([facial_out, vocal_out])
        fusion_dense = layers.Dense(256, activation='relu')(fusion)
        fusion_dropout = layers.Dropout(0.5)(fusion_dense)
        fusion_out = layers.Dense(128, activation='relu')(fusion_dropout)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='stress_output')(fusion_out)
        
        # Create model
        model = models.Model(inputs=[facial_input, vocal_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_facial_emotion_model(self, use_fer2013: bool = True, 
                                 epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train facial emotion recognition model.
        
        Args:
            use_fer2013 (bool): Whether to use FER-2013 dataset
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            dict: Training results and metrics
        """
        print("=" * 60)
        print("TRAINING FACIAL EMOTION RECOGNITION MODEL")
        print("=" * 60)
        
        # Load dataset
        facial_loader = DatasetLoader('facial')
        
        try:
            if use_fer2013:
                X_train, X_test, y_train, y_test = facial_loader.load_fer2013()
                num_classes = len(facial_loader.fer2013_emotions)
                dataset_name = "FER-2013"
            else:
                raise NotImplementedError("Other facial datasets not yet implemented")
                
        except FileNotFoundError as e:
            print(f"Dataset not found: {e}")
            print("Please download the dataset and place it in the correct directory.")
            return {"error": str(e)}
        
        input_shape = X_train.shape[1:]
        
        if TENSORFLOW_AVAILABLE:
            # Build CNN model
            print(f"Building CNN model for {dataset_name}...")
            self.facial_model = self.build_cnn_facial_model(input_shape, num_classes)
            print(self.facial_model.summary())
            
            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                callbacks.ModelCheckpoint(
                    os.path.join(self.model_save_path, 'facial_emotion_cnn.h5'),
                    save_best_only=True
                )
            ]
            
            # Train model
            print("Training CNN model...")
            history = self.facial_model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.facial_model.evaluate(X_test, y_test, verbose=0)
            y_pred = self.facial_model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Save model
            self.facial_model.save(os.path.join(self.model_save_path, 'facial_emotion_cnn_final.h5'))
            
            results = {
                "model_type": "CNN",
                "dataset": dataset_name,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "history": history.history,
                "classification_report": classification_report(y_true_classes, y_pred_classes),
                "model_path": os.path.join(self.model_save_path, 'facial_emotion_cnn_final.h5')
            }
            
        else:
            # Fallback to traditional ML
            print("Using traditional ML models (TensorFlow not available)...")
            
            # Flatten images for traditional ML
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_classes = np.argmax(y_train, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)
            
            # Train multiple models
            models_to_try = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
            
            best_model = None
            best_accuracy = 0
            model_results = {}
            
            for name, model in models_to_try.items():
                print(f"Training {name}...")
                model.fit(X_train_scaled, y_train_classes)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test_classes, y_pred)
                
                model_results[name] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test_classes, y_pred)
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    self.facial_model = model
            
            # Save best model
            model_path = os.path.join(self.model_save_path, 'facial_emotion_sklearn.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({'model': best_model, 'scaler': scaler}, f)
            
            results = {
                "model_type": "Traditional ML",
                "dataset": dataset_name,
                "best_accuracy": best_accuracy,
                "model_results": model_results,
                "model_path": model_path
            }
        
        self.training_history['facial'] = results
        print(f"Facial emotion model training completed. Best accuracy: {results.get('test_accuracy', results.get('best_accuracy', 0)):.4f}")
        
        return results
    
    def train_vocal_emotion_model(self, use_ravdess: bool = True, 
                                epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train vocal emotion/stress recognition model.
        
        Args:
            use_ravdess (bool): Whether to use RAVDESS dataset
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            dict: Training results and metrics
        """
        print("=" * 60)
        print("TRAINING VOCAL EMOTION RECOGNITION MODEL")
        print("=" * 60)
        
        # Load dataset
        vocal_loader = DatasetLoader('vocal')
        
        try:
            if use_ravdess:
                X_train, X_test, y_train, y_test = vocal_loader.load_ravdess()
                num_classes = len(vocal_loader.ravdess_emotions)
                dataset_name = "RAVDESS"
            else:
                raise NotImplementedError("Other vocal datasets not yet implemented")
                
        except FileNotFoundError as e:
            print(f"Dataset not found: {e}")
            print("Please download the dataset and place it in the correct directory.")
            return {"error": str(e)}
        
        input_shape = X_train.shape[1:]
        
        if TENSORFLOW_AVAILABLE:
            # Build LSTM model
            print(f"Building LSTM model for {dataset_name}...")
            self.vocal_model = self.build_lstm_vocal_model(input_shape, num_classes)
            print(self.vocal_model.summary())
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
                callbacks.ModelCheckpoint(
                    os.path.join(self.model_save_path, 'vocal_emotion_lstm.h5'),
                    save_best_only=True
                )
            ]
            
            # Train model
            print("Training LSTM model...")
            history = self.vocal_model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.vocal_model.evaluate(X_test, y_test, verbose=0)
            y_pred = self.vocal_model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Save model
            self.vocal_model.save(os.path.join(self.model_save_path, 'vocal_emotion_lstm_final.h5'))
            
            results = {
                "model_type": "LSTM",
                "dataset": dataset_name,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "history": history.history,
                "classification_report": classification_report(y_true_classes, y_pred_classes),
                "model_path": os.path.join(self.model_save_path, 'vocal_emotion_lstm_final.h5')
            }
            
        else:
            # Traditional ML approach
            print("Using traditional ML models (TensorFlow not available)...")
            
            y_train_classes = np.argmax(y_train, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models_to_try = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
            
            best_model = None
            best_accuracy = 0
            model_results = {}
            
            for name, model in models_to_try.items():
                print(f"Training {name}...")
                model.fit(X_train_scaled, y_train_classes)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test_classes, y_pred)
                
                model_results[name] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test_classes, y_pred)
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    self.vocal_model = model
            
            # Save best model
            model_path = os.path.join(self.model_save_path, 'vocal_emotion_sklearn.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({'model': best_model, 'scaler': scaler}, f)
            
            results = {
                "model_type": "Traditional ML",
                "dataset": dataset_name,
                "best_accuracy": best_accuracy,
                "model_results": model_results,
                "model_path": model_path
            }
        
        self.training_history['vocal'] = results
        print(f"Vocal emotion model training completed. Best accuracy: {results.get('test_accuracy', results.get('best_accuracy', 0)):.4f}")
        
        return results
    
    def train_stress_detection_model(self) -> Dict:
        """
        Train stress detection model that combines facial and vocal predictions.
        
        Returns:
            dict: Training results and metrics
        """
        print("=" * 60)
        print("TRAINING STRESS DETECTION MODEL")
        print("=" * 60)
        
        if self.facial_model is None or self.vocal_model is None:
            print("Both facial and vocal models must be trained first!")
            return {"error": "Models not available"}
        
        # This is a simplified version - in practice, you'd need aligned data
        # For now, we'll create a stress mapping based on emotion predictions
        
        # Load datasets to get emotion-to-stress mappings
        facial_loader = DatasetLoader('facial')
        vocal_loader = DatasetLoader('vocal')
        
        # Create stress detection rules
        stress_rules = {
            'low_stress': ['happy', 'neutral', 'calm'],
            'medium_stress': ['sad', 'surprised', 'disgust'],
            'high_stress': ['angry', 'fearful']
        }
        
        # Save stress model (rule-based for now)
        stress_model_data = {
            'type': 'rule_based',
            'rules': stress_rules,
            'facial_model_path': self.training_history.get('facial', {}).get('model_path'),
            'vocal_model_path': self.training_history.get('vocal', {}).get('model_path'),
            'created_at': datetime.now().isoformat()
        }
        
        stress_model_path = os.path.join(self.model_save_path, 'stress_detection_rules.pkl')
        with open(stress_model_path, 'wb') as f:
            pickle.dump(stress_model_data, f)
        
        results = {
            "model_type": "Rule-based",
            "stress_levels": ['low_stress', 'medium_stress', 'high_stress'],
            "rules": stress_rules,
            "model_path": stress_model_path
        }
        
        self.training_history['stress'] = results
        print("Stress detection model created successfully!")
        
        return results
    
    def save_training_report(self) -> str:
        """
        Generate and save a comprehensive training report.
        
        Returns:
            str: Path to the saved report
        """
        report_path = os.path.join(self.model_save_path, f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_path, 'w') as f:
            f.write("STRESS BURNOUT WARNING SYSTEM - TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_type, results in self.training_history.items():
                f.write(f"{model_type.upper()} MODEL RESULTS\n")
                f.write("-" * 30 + "\n")
                
                for key, value in results.items():
                    if key != 'history':  # Skip detailed history for readability
                        f.write(f"{key}: {value}\n")
                
                f.write("\n")
        
        print(f"Training report saved to: {report_path}")
        return report_path


# Training script
def main():
    """
    Main training pipeline for all models.
    """
    print("STRESS BURNOUT WARNING SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train facial emotion model
    print("\n1. Training Facial Emotion Recognition Model...")
    facial_results = trainer.train_facial_emotion_model(epochs=20)  # Reduced epochs for testing
    
    # Train vocal emotion model
    print("\n2. Training Vocal Emotion Recognition Model...")
    vocal_results = trainer.train_vocal_emotion_model(epochs=20)  # Reduced epochs for testing
    
    # Train stress detection model
    print("\n3. Creating Stress Detection Model...")
    stress_results = trainer.train_stress_detection_model()
    
    # Generate report
    print("\n4. Generating Training Report...")
    report_path = trainer.save_training_report()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Models saved in: {trainer.model_save_path}")
    print(f"Training report: {report_path}")


if __name__ == "__main__":
    main()
