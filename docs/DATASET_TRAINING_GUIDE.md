# Stress Burnout Warning System - Dataset Training Guide

This guide provides comprehensive instructions for setting up datasets and training models for the Stress Burnout Warning System.

## 🎯 Project Overview

The Stress Burnout Warning System uses AI/ML models to detect early signs of stress and burnout through:

- **Facial Emotion Recognition**: CNN models trained on emotion datasets (FER-2013, AffectNet)
- **Vocal Stress Detection**: LSTM/RNN models trained on audio emotion datasets (RAVDESS, SAVEE)
- **Fusion Models**: Combined analysis for improved stress prediction
- **Real-time Processing**: Optimized models for live camera and microphone input

## 📁 Directory Structure

After setup, your project will have this structure:

```
Stress-Burnout-Warning-System/
├── datasets/                          # Training datasets
│   ├── facial_emotion/
│   │   ├── fer2013/                   # FER-2013 dataset
│   │   ├── affectnet/                 # AffectNet dataset  
│   │   └── ck_plus/                   # CK+ dataset
│   ├── vocal_emotion/
│   │   ├── ravdess/                   # RAVDESS audio dataset
│   │   ├── savee/                     # SAVEE dataset
│   │   └── iemocap/                   # IEMOCAP dataset
│   ├── custom/                        # Custom collected data
│   └── processed/                     # Preprocessed datasets
├── models/                            # Trained models
├── src/                              # Source code
│   ├── data/
│   │   └── dataset_loader.py         # Dataset loading utilities
│   ├── ai/
│   │   └── model_training.py         # Main training module
│   └── monitoring/
│       └── facial_training.py        # Facial model training
├── config/                           # Configuration files
├── logs/                            # Training logs
├── train_complete_system.py         # Main training script
├── dataset_downloader.py            # Dataset download utility
└── setup_project.py                # Project setup script
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd Stress-Burnout-Warning-System

# Run the setup script
python setup_project.py

# This will:
# - Check Python version (3.8+ required)
# - Install dependencies from requirements.txt
# - Create directory structure
# - Verify system requirements
# - Create initial configuration files
```

### 2. Dataset Download

```bash
# Option A: Download all datasets automatically (requires Kaggle API)
python dataset_downloader.py --download-all

# Option B: Download specific datasets
python dataset_downloader.py --download-fer2013
python dataset_downloader.py --download-ravdess

# Option C: Create sample dataset for testing
python dataset_downloader.py --create-sample

# Check dataset status
python dataset_downloader.py --verify
```

### 3. Model Training

```bash
# Train all models with default settings
python train_complete_system.py

# Train specific model types
python train_complete_system.py --facial-only
python train_complete_system.py --vocal-only

# Quick test with reduced epochs
python train_complete_system.py --quick-test

# Custom training with specific parameters
python train_complete_system.py --epochs 50 --batch-size 64 --architectures custom vgg16
```

## 📊 Supported Datasets

### Facial Emotion Recognition

#### FER-2013 (Primary)
- **Description**: 35,887 grayscale 48x48 pixel facial images
- **Emotions**: angry, disgusted, fearful, happy, neutral, sad, surprised
- **Size**: ~96 MB
- **Download**: Automatic via Kaggle API
- **Usage**: Primary dataset for facial emotion recognition

#### AffectNet (Advanced)
- **Description**: 1M+ facial images with emotion annotations
- **Emotions**: 8 basic emotions + compound emotions
- **Size**: ~4 GB
- **Download**: Manual registration required
- **Usage**: Advanced training with more diverse data

### Vocal Emotion Recognition

#### RAVDESS (Primary)
- **Description**: 24 professional actors, 7 emotions, audio-visual
- **Emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Size**: ~24 GB (audio subset ~2 GB)
- **Download**: Automatic via Kaggle API
- **Usage**: Primary dataset for vocal emotion recognition

#### SAVEE & IEMOCAP (Advanced)
- **Description**: Additional audio-visual emotion databases
- **Usage**: Extended training and validation

## 🤖 Model Architectures

### Facial Models

#### Custom CNN
- Optimized for 48x48 grayscale emotion images
- 4 convolutional blocks with batch normalization
- Global average pooling and dense layers
- Dropout for regularization

#### Transfer Learning Models
- **VGG16**: Fine-tuned for emotion recognition
- **ResNet50**: Deep residual learning
- **MobileNetV2**: Lightweight for real-time inference

### Vocal Models

#### LSTM/RNN
- Audio feature extraction (MFCCs, spectral features)
- Bidirectional LSTM layers
- Sequence modeling for temporal patterns

#### Traditional ML (Fallback)
- Random Forest, SVM, Gradient Boosting
- Used when TensorFlow is not available
- Feature-based classification

### Stress Detection

#### Rule-Based Mapping
- Maps emotions to stress levels:
  - **High Stress**: angry, fearful
  - **Medium Stress**: sad, surprised, disgusted
  - **Low Stress**: happy, neutral, calm

#### Fusion Models (Future)
- Combines facial and vocal predictions
- Weighted ensemble methods
- Temporal stress level tracking

## 🔧 Configuration

### Training Configuration

Edit `config/training_config.json`:

```json
{
    "facial_models": {
        "enabled": true,
        "architectures": ["custom", "vgg16"],
        "epochs": 50,
        "batch_size": 32,
        "use_transfer_learning": true,
        "dataset": "fer2013"
    },
    "vocal_models": {
        "enabled": true,
        "epochs": 30,
        "batch_size": 32,
        "dataset": "ravdess"
    },
    "stress_models": {
        "enabled": true,
        "fusion_enabled": false
    }
}
```

### System Configuration

Edit `config/system_config.ini`:

```ini
[models]
facial_model_path = models/facial_emotion_cnn_final.h5
vocal_model_path = models/vocal_emotion_lstm_final.h5
stress_model_path = models/stress_detection_rules.pkl

[training]
batch_size = 32
epochs = 50
learning_rate = 0.001
```

## 📈 Training Process

### 1. Data Loading and Preprocessing
- Loads datasets using `DatasetLoader` class
- Applies preprocessing (normalization, augmentation)
- Splits data into train/validation/test sets

### 2. Model Training
- Builds model architecture based on configuration
- Applies data augmentation for better generalization
- Uses callbacks (early stopping, learning rate reduction)
- Saves best models during training

### 3. Evaluation and Validation
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrices
- Creates classification reports
- Saves model performance metrics

### 4. Stress Mapping
- Maps emotion predictions to stress levels
- Creates rule-based stress detection models
- Saves stress prediction configurations

## 📊 Expected Performance

### Facial Emotion Recognition
- **FER-2013 Baseline**: 65-75% accuracy
- **Custom CNN**: 70-80% accuracy
- **Transfer Learning**: 75-85% accuracy

### Vocal Emotion Recognition
- **RAVDESS Audio**: 60-70% accuracy
- **LSTM Models**: 65-75% accuracy
- **Traditional ML**: 55-65% accuracy

### Stress Detection
- **Rule-based**: 70-80% correlation with self-reported stress
- **Combined Models**: 75-85% accuracy (target)

## 🔍 Troubleshooting

### Common Issues

#### Dataset Download Fails
```bash
# Setup Kaggle API
pip install kaggle
# Download kaggle.json from Kaggle account settings
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### TensorFlow Import Errors
```bash
# Install TensorFlow
pip install tensorflow>=2.13.0

# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal
```

#### Audio Processing Issues
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Windows
pip install pyaudio
```

#### Memory Issues During Training
- Reduce batch size: `--batch-size 16`
- Use smaller epochs: `--epochs 10`
- Enable mixed precision training
- Use model checkpointing

### Performance Optimization

#### For CPU Training
- Use traditional ML models
- Reduce image resolution
- Limit dataset size
- Use multiprocessing

#### For GPU Training
- Install CUDA-compatible TensorFlow
- Use larger batch sizes
- Enable mixed precision
- Use data parallelism

## 🔬 Advanced Usage

### Custom Dataset Integration

1. **Add Custom Data**:
   ```python
   # Place data in datasets/custom/
   # Update DatasetLoader to handle custom format
   ```

2. **Modify Training Pipeline**:
   ```python
   # Edit src/ai/model_training.py
   # Add custom dataset loading logic
   ```

### Model Customization

1. **Custom Architecture**:
   ```python
   # Edit src/monitoring/facial_training.py
   # Add new model building methods
   ```

2. **Hyperparameter Tuning**:
   ```python
   # Use GridSearchCV for traditional ML
   # Use Keras Tuner for deep learning
   ```

### Real-time Optimization

1. **Model Quantization**:
   ```python
   # Convert models to TensorFlow Lite
   # Use int8 quantization for mobile deployment
   ```

2. **Inference Optimization**:
   ```python
   # Use TensorRT for NVIDIA GPUs
   # OpenVINO for Intel processors
   ```

## 📚 References

### Datasets
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)
- [AffectNet](http://mohammadmahoor.com/affectnet/)

### Papers
- "Challenges in Representation Learning: A report on three machine learning contests" (FER-2013)
- "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" 
- "Going deeper with convolutions" (Inception/GoogLeNet)

### Libraries
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Librosa](https://librosa.org/)
- [MediaPipe](https://mediapipe.dev/)

## 🤝 Contributing

1. **Dataset Contributions**: Add support for new emotion datasets
2. **Model Improvements**: Implement state-of-the-art architectures
3. **Performance Optimization**: Improve training speed and accuracy
4. **Documentation**: Enhance guides and tutorials

## 📄 License

This project is part of the Stress Burnout Warning System and follows the same licensing terms.

---

For additional support or questions, please refer to the main project documentation or create an issue in the project repository.
