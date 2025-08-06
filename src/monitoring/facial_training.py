"""
Enhanced Facial Expression Training Module for Stress Detection

This module implements advanced CNN architectures for facial emotion recognition
with specific focus on stress-related emotions and micro-expressions.

Features:
- Multiple CNN architectures (ResNet, VGG-like, Custom)
- Transfer learning capabilities
- Data augmentation and preprocessing
- Real-time inference optimization
- Stress-level mapping from emotions

Author: Stress Burnout Warning System Team
Date: August 2025
"""

import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Optional, Dict

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks, applications
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. CNN training disabled.")

# Traditional ML fallback
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataset_loader import DatasetLoader


class FacialExpressionTrainer:
    """
    Advanced facial expression trainer with multiple architectures and stress mapping.
    """
    
    def __init__(self, model_save_path: str = None):
        """
        Initialize the facial expression trainer.
        
        Args:
            model_save_path (str): Path to save trained models
        """
        if model_save_path is None:
            self.model_save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        else:
            self.model_save_path = model_save_path
        
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Model configurations
        self.models = {}
        self.training_history = {}
        
        # Emotion to stress mapping
        self.emotion_stress_mapping = {
            'angry': 'high_stress',
            'disgusted': 'medium_stress',
            'fearful': 'high_stress',
            'happy': 'low_stress',
            'neutral': 'low_stress',
            'sad': 'medium_stress',
            'surprised': 'medium_stress'
        }
        
    def build_custom_cnn(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """
        Build custom CNN architecture optimized for facial emotion recognition.
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of emotion classes
            
        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and classification
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def build_transfer_learning_model(self, input_shape: Tuple[int, int, int], 
                                    num_classes: int, base_model: str = 'VGG16') -> keras.Model:
        """
        Build transfer learning model using pre-trained networks.
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of classes
            base_model (str): Base model architecture
            
        Returns:
            keras.Model: Transfer learning model
        """
        # Load pre-trained base model
        if base_model == 'VGG16':
            base = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model == 'ResNet50':
            base = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model == 'MobileNetV2':
            base = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Freeze base model layers
        base.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base,
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
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_facial_data(self, images: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Advanced preprocessing for facial images.
        
        Args:
            images (np.ndarray): Input images
            target_size (tuple): Target image size
            
        Returns:
            np.ndarray: Preprocessed images
        """
        processed_images = []
        
        for img in images:
            # Resize if needed
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size)
            
            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            # Apply histogram equalization for better contrast
            img_yuv = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype('float32') / 255.0
            
            processed_images.append(img)
        
        return np.array(processed_images)
    
    def create_advanced_data_generator(self) -> ImageDataGenerator:
        """
        Create advanced data augmentation generator.
        
        Returns:
            ImageDataGenerator: Configured data generator
        """
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
    
    def train_model(self, architecture: str = 'custom', dataset: str = 'fer2013',
                   epochs: int = 50, batch_size: int = 32, 
                   use_transfer_learning: bool = False) -> Dict:
        """
        Train facial emotion recognition model.
        
        Args:
            architecture (str): Model architecture ('custom', 'vgg16', 'resnet50', 'mobilenet')
            dataset (str): Dataset to use ('fer2013')
            epochs (int): Training epochs
            batch_size (int): Batch size
            use_transfer_learning (bool): Whether to use transfer learning
            
        Returns:
            dict: Training results
        """
        print(f"Training {architecture} model on {dataset} dataset...")
        
        # Load data
        data_loader = DatasetLoader('facial')
        
        try:
            if dataset == 'fer2013':
                X_train, X_test, y_train, y_test = data_loader.load_fer2013()
                num_classes = len(data_loader.fer2013_emotions)
                class_names = data_loader.fer2013_emotions
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
        except FileNotFoundError as e:
            return {"error": f"Dataset not found: {e}"}
        
        if not TENSORFLOW_AVAILABLE:
            return self._train_traditional_ml(X_train, X_test, y_train, y_test, class_names)
        
        # Preprocess data for transfer learning if needed
        if use_transfer_learning:
            input_shape = (224, 224, 3)
            X_train = self.preprocess_facial_data(X_train, (224, 224))
            X_test = self.preprocess_facial_data(X_test, (224, 224))
        else:
            input_shape = X_train.shape[1:]
        
        # Build model
        if architecture == 'custom':
            model = self.build_custom_cnn(input_shape, num_classes)
        elif use_transfer_learning:
            model = self.build_transfer_learning_model(input_shape, num_classes, architecture.upper())
        else:
            raise ValueError(f"Architecture {architecture} requires transfer learning")
        
        print(f"Model architecture: {architecture}")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        # Create data generator
        datagen = self.create_advanced_data_generator()
        
        # Callbacks
        model_name = f"{architecture}_{dataset}"
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            callbacks.ModelCheckpoint(
                os.path.join(self.model_save_path, f'{model_name}_best.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Generate detailed metrics
        classification_rep = classification_report(
            y_true_classes, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Create stress prediction model
        stress_model = self._create_stress_mapping_model(model, class_names)
        
        # Save final model
        final_model_path = os.path.join(self.model_save_path, f'{model_name}_final.h5')
        model.save(final_model_path)
        
        # Save stress model
        stress_model_path = os.path.join(self.model_save_path, f'{model_name}_stress.pkl')
        with open(stress_model_path, 'wb') as f:
            pickle.dump(stress_model, f)
        
        results = {
            "architecture": architecture,
            "dataset": dataset,
            "use_transfer_learning": use_transfer_learning,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "classification_report": classification_rep,
            "history": history.history,
            "model_path": final_model_path,
            "stress_model_path": stress_model_path,
            "class_names": class_names,
            "training_time": datetime.now().isoformat()
        }
        
        self.models[model_name] = model
        self.training_history[model_name] = results
        
        print(f"Training completed! Test accuracy: {test_accuracy:.4f}")
        
        return results
    
    def _train_traditional_ml(self, X_train, X_test, y_train, y_test, class_names) -> Dict:
        """
        Fallback training using traditional ML when TensorFlow is not available.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            class_names: List of class names
            
        Returns:
            dict: Training results
        """
        print("Training with traditional ML (TensorFlow not available)...")
        
        # Flatten images
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Train models
        models_to_try = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0
        model_results = {}
        
        for name, model in models_to_try.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train_classes)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = (y_pred == y_test_classes).mean()
            
            model_results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test_classes, y_pred, target_names=class_names)
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        # Save best model
        model_path = os.path.join(self.model_save_path, 'facial_emotion_traditional_ml.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'model': best_model, 'scaler': scaler, 'class_names': class_names}, f)
        
        # Create stress mapping
        stress_model = {
            'type': 'traditional_ml_with_stress',
            'model_path': model_path,
            'emotion_to_stress': self.emotion_stress_mapping,
            'class_names': class_names
        }
        
        stress_model_path = os.path.join(self.model_save_path, 'facial_stress_traditional.pkl')
        with open(stress_model_path, 'wb') as f:
            pickle.dump(stress_model, f)
        
        return {
            "architecture": "Traditional ML",
            "dataset": "fer2013",
            "best_accuracy": best_accuracy,
            "model_results": model_results,
            "model_path": model_path,
            "stress_model_path": stress_model_path,
            "class_names": class_names
        }
    
    def _create_stress_mapping_model(self, emotion_model: keras.Model, class_names: List[str]) -> Dict:
        """
        Create stress prediction model from emotion model.
        
        Args:
            emotion_model (keras.Model): Trained emotion recognition model
            class_names (list): List of emotion class names
            
        Returns:
            dict: Stress mapping model configuration
        """
        stress_model = {
            'type': 'emotion_to_stress_mapping',
            'emotion_model': emotion_model,
            'class_names': class_names,
            'emotion_to_stress': self.emotion_stress_mapping,
            'stress_levels': ['low_stress', 'medium_stress', 'high_stress']
        }
        
        return stress_model
    
    def predict_stress_from_emotion(self, emotion_probabilities: np.ndarray, 
                                  class_names: List[str]) -> Tuple[str, float]:
        """
        Predict stress level from emotion probabilities.
        
        Args:
            emotion_probabilities (np.ndarray): Emotion prediction probabilities
            class_names (list): List of emotion class names
            
        Returns:
            tuple: (stress_level, confidence)
        """
        # Calculate stress probabilities
        stress_probs = {'low_stress': 0, 'medium_stress': 0, 'high_stress': 0}
        
        for i, prob in enumerate(emotion_probabilities):
            emotion = class_names[i]
            stress_level = self.emotion_stress_mapping.get(emotion, 'low_stress')
            stress_probs[stress_level] += prob
        
        # Get dominant stress level
        predicted_stress = max(stress_probs, key=stress_probs.get)
        confidence = stress_probs[predicted_stress]
        
        return predicted_stress, confidence
    
    def save_training_summary(self) -> str:
        """
        Save comprehensive training summary.
        
        Returns:
            str: Path to saved summary
        """
        summary_path = os.path.join(
            self.model_save_path, 
            f'facial_training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        
        with open(summary_path, 'w') as f:
            f.write("FACIAL EXPRESSION TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_name, results in self.training_history.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                for key, value in results.items():
                    if key not in ['history', 'classification_report']:
                        f.write(f"{key}: {value}\n")
                
                f.write("\n")
        
        print(f"Training summary saved to: {summary_path}")
        return summary_path


# Training script
def main():
    """
    Main training script for facial emotion recognition.
    """
    print("FACIAL EXPRESSION TRAINING FOR STRESS DETECTION")
    print("=" * 60)
    
    trainer = FacialExpressionTrainer()
    
    # Train different architectures
    architectures_to_try = [
        {'architecture': 'custom', 'use_transfer_learning': False},
    ]
    
    # Add transfer learning models if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        architectures_to_try.extend([
            {'architecture': 'vgg16', 'use_transfer_learning': True},
            {'architecture': 'resnet50', 'use_transfer_learning': True},
            {'architecture': 'mobilenet', 'use_transfer_learning': True}
        ])
    
    results = []
    
    for config in architectures_to_try:
        print(f"\nTraining {config['architecture']} architecture...")
        try:
            result = trainer.train_model(
                architecture=config['architecture'],
                use_transfer_learning=config['use_transfer_learning'],
                epochs=20,  # Reduced for testing
                batch_size=32
            )
            results.append(result)
            
            if 'error' not in result:
                accuracy = result.get('test_accuracy', result.get('best_accuracy', 0))
                print(f"✓ {config['architecture']} trained successfully! Accuracy: {accuracy:.4f}")
            else:
                print(f"✗ {config['architecture']} training failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ Error training {config['architecture']}: {e}")
    
    # Save summary
    if results:
        summary_path = trainer.save_training_summary()
        print(f"\nTraining completed! Summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    main()
