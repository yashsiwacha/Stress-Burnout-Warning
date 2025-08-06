# 🧠 Stress Burnout Warning System

An AI/ML-driven application designed to proactively identify early signs of stress and potential burnout through real-time analysis of facial expressions and vocal patterns. The system provides timely interventions and personalized recovery strategies to enhance mental well-being and productivity.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.13+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

The Stress Burnout Warning System leverages cutting-edge AI/ML techniques to:

- **Detect Early Stress Signs**: Real-time facial emotion recognition and vocal pattern analysis
- **Predict Burnout Risk**: Multi-modal fusion of visual and auditory cues
- **Provide Immediate Interventions**: Actionable stress relief suggestions
- **Offer Long-term Strategies**: Personalized coping mechanisms and resources
- **Ensure Privacy**: Local processing with no data transmission

## 🚀 Key Features

### 🤖 Advanced AI Models
- **Facial Emotion Recognition**: CNN models trained on FER-2013, AffectNet datasets
- **Vocal Stress Detection**: LSTM models analyzing prosodic and spectral features
- **Transfer Learning**: Pre-trained VGG16, ResNet50, MobileNetV2 architectures
- **Fusion Models**: Multi-modal stress prediction combining visual and audio analysis

### 📊 Comprehensive Dataset Support
- **FER-2013**: 35,887 facial emotion images (7 emotions)
- **RAVDESS**: Audio-visual emotion database (24 actors, 8 emotions)
- **AffectNet**: Large-scale facial expression dataset (1M+ images)
- **Custom Data**: Support for user-specific stress data collection

### 🛠️ Production-Ready Infrastructure
- **Automated Dataset Management**: Download and preprocessing pipelines
- **Cross-Platform Compatibility**: Windows, macOS, Linux support
- **Real-time Processing**: Optimized for live camera and microphone input
- **Fallback Support**: Traditional ML when deep learning unavailable

## 📁 Project Structure

```
Stress-Burnout-Warning-System/
├── � main.py                      # Main application entry point
├── 📄 requirements.txt             # Python dependencies
├── 📂 src/                        # Source code modules
│   ├── ai/                        # AI/ML models and training
│   ├── data/                      # Data management utilities
│   ├── monitoring/                # Real-time monitoring modules
│   ├── ui/                        # User interface components
│   └── analysis/                  # Stress analysis algorithms
├── 📂 datasets/                   # Training and test datasets
├── 📂 models/                     # Trained ML models
├── 📂 docs/                       # Documentation
├── 📂 scripts/                    # Setup and utility scripts
├── 📂 demos/                      # Demo and example files
├── 📂 tools/                      # Development tools
└── 📂 archive/                    # Archived/backup files
```

📖 **For detailed structure**: See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
│   ├── monitoring/                # Real-time monitoring systems
│   └── ui/                        # User interface components
├── 📂 models/                     # Trained model storage
├── 📂 config/                     # Configuration files
├── 🐍 train_complete_system.py    # Main training orchestrator
├── 🐍 dataset_downloader.py       # Automated dataset downloader
├── 🐍 setup_project.py           # Project setup utility
└── 📖 DATASET_TRAINING_GUIDE.md   # Comprehensive training guide
```

## 🔧 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yashsiwacha/Stress-Burnout-Warning.git
cd Stress-Burnout-Warning

# Setup environment and dependencies
python scripts/setup_project.py
```

### 2. Dataset Preparation

```bash
# Option A: Download real datasets (requires Kaggle API)
python scripts/dataset_downloader.py --setup
python scripts/dataset_downloader.py --download-all

# Option B: Create sample data for testing
python scripts/dataset_downloader.py --create-sample
```

### 3. Model Training

```bash
# Train all models with default settings
python scripts/train_complete_system.py

# Quick test with sample data
python train_complete_system.py --quick-test

# Custom training
python train_complete_system.py --epochs 50 --architectures custom vgg16
```

### 4. Run the Application

```bash
# Launch the main stress detection system
python main.py
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended for training)
- **Storage**: 10GB free space for datasets and models
- **Optional**: CUDA-compatible GPU for faster training

## 📦 Dependencies

Key dependencies include:
- `tensorflow>=2.13.0` - Deep learning framework
- `opencv-python>=4.8.0` - Computer vision
- `librosa>=0.10.1` - Audio processing
- `scikit-learn>=1.3.0` - Traditional ML algorithms
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation

See `requirements.txt` for complete list.

## 🎯 Model Performance

### Expected Results
- **Facial Emotion Recognition**: 70-85% accuracy
- **Vocal Emotion Detection**: 60-75% accuracy
- **Stress Level Prediction**: 75-85% accuracy
- **Real-time Processing**: <100ms inference time

### Supported Emotions
- **Facial**: angry, disgusted, fearful, happy, neutral, sad, surprised
- **Vocal**: neutral, calm, happy, sad, angry, fearful, disgust, surprised

### Stress Level Mapping
- **High Stress**: angry, fearful emotions
- **Medium Stress**: sad, surprised, disgusted emotions
- **Low Stress**: happy, neutral, calm emotions

## 🔬 Technical Approach

### Facial Analysis Pipeline
1. **Face Detection**: MediaPipe/OpenCV face detection
2. **Preprocessing**: Normalization, augmentation, histogram equalization
3. **Feature Extraction**: CNN-based emotion classification
4. **Stress Mapping**: Emotion-to-stress level conversion

### Vocal Analysis Pipeline
1. **Audio Capture**: Real-time microphone input
2. **Feature Extraction**: MFCCs, spectral features, prosodic analysis
3. **Sequence Modeling**: LSTM-based emotion recognition
4. **Stress Prediction**: Temporal pattern analysis

### Fusion Strategy
- **Late Fusion**: Combine predictions from facial and vocal models
- **Weighted Ensemble**: Confidence-based prediction weighting
- **Temporal Smoothing**: Reduce noise in real-time predictions

## 📊 Dataset Information

### Download Instructions

#### FER-2013
```bash
# Automatic download via Kaggle API
python dataset_downloader.py --download-fer2013

# Manual download
# 1. Visit: https://www.kaggle.com/datasets/msambare/fer2013
# 2. Extract to: datasets/facial_emotion/fer2013/
```

#### RAVDESS
```bash
# Automatic download via Kaggle API
python dataset_downloader.py --download-ravdess

# Manual download
# 1. Visit: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
# 2. Extract to: datasets/vocal_emotion/ravdess/
```

## 🔧 Configuration

### Training Configuration (`config/training_config.json`)
```json
{
    "facial_models": {
        "enabled": true,
        "architectures": ["custom", "vgg16", "resnet50"],
        "epochs": 50,
        "batch_size": 32,
        "use_transfer_learning": true
    },
    "vocal_models": {
        "enabled": true,
        "epochs": 30,
        "batch_size": 32
    },
    "stress_models": {
        "enabled": true,
        "fusion_enabled": false
    }
}
```

## 🔍 Usage Examples

### Basic Stress Detection
```python
from src.monitoring.facial_monitor import FacialMonitor
from src.monitoring.voice_monitor import VoiceMonitor

# Initialize monitors
facial_monitor = FacialMonitor()
voice_monitor = VoiceMonitor()

# Start real-time monitoring
facial_monitor.start()
voice_monitor.start()

# Get stress predictions
stress_level = facial_monitor.get_current_stress_level()
print(f"Current stress level: {stress_level}")
```

### Custom Model Training
```python
from src.ai.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train facial emotion model
facial_results = trainer.train_facial_emotion_model(epochs=50)

# Train vocal emotion model
vocal_results = trainer.train_vocal_emotion_model(epochs=30)

# Create stress detection model
stress_results = trainer.train_stress_detection_model()
```

## 📈 Performance Monitoring

The system includes comprehensive logging and monitoring:
- **Training Metrics**: Accuracy, loss, validation scores
- **Real-time Performance**: Inference time, prediction confidence
- **System Resources**: CPU/GPU usage, memory consumption
- **User Feedback**: Stress level tracking over time

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- Additional emotion datasets
- New model architectures
- Performance optimizations
- UI/UX improvements
- Documentation enhancements

## 📝 Documentation

- **[Dataset Training Guide](DATASET_TRAINING_GUIDE.md)**: Comprehensive training instructions
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Technical implementation details
- **[API Documentation](docs/api.md)**: Code API reference
- **[Deployment Guide](docs/deployment.md)**: Production deployment instructions

## 🔒 Privacy & Ethics

- **Local Processing**: All analysis performed locally, no data transmission
- **User Consent**: Explicit permission required for data collection
- **Data Security**: Secure storage of any collected data
- **Ethical AI**: Bias mitigation and fair representation in training data

## 🐛 Troubleshooting

### Common Issues

#### TensorFlow Installation
```bash
# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal

# For CUDA-enabled systems
pip install tensorflow-gpu
```

#### Audio Processing Issues
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### Dataset Download Fails
```bash
# Setup Kaggle API
pip install kaggle
# Download kaggle.json from account settings
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: FER-2013, RAVDESS, AffectNet contributors
- **Libraries**: TensorFlow, OpenCV, Librosa, scikit-learn teams
- **Research**: Emotion recognition and stress detection research community
- **Inspiration**: Mental health awareness and workplace wellness initiatives

## 📞 Contact

- **Author**: Yash Siwacha
- **GitHub**: [@yashsiwacha](https://github.com/yashsiwacha)
- **Project Link**: [https://github.com/yashsiwacha/Stress-Burnout-Warning](https://github.com/yashsiwacha/Stress-Burnout-Warning)

---

⭐ **Star this repository if you find it helpful!** ⭐

**Made with ❤️ for mental health and workplace wellness**