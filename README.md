# 🧠 Stress Burnout Warning System V2.0

An advanced AI/ML-driven wellness platform designed to proactively identify early signs of stress and potential burnout through real-time analysis of facial expressions and vocal patterns. The system provides timely interventions, personalized recovery strategies, and comprehensive wellness monitoring to enhance mental well-being and productivity.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.13+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.8+-green.svg)
![PySide6](https://img.shields.io/badge/PySide6-v6.6+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-2.0-brightgreen.svg)

## 🎯 Project Overview

The Stress Burnout Warning System V2.0 leverages cutting-edge AI/ML techniques to provide a comprehensive mental wellness platform:

- **Real-time Stress Detection**: Advanced facial emotion recognition and vocal pattern analysis
- **Predictive Analytics**: Multi-modal fusion for accurate stress prediction
- **Modern GUI Interface**: Beautiful native desktop application with real-time monitoring
- **Immediate Interventions**: Smart notifications and actionable stress relief suggestions
- **Wellness Ecosystem**: Integrated meditation, breathing exercises, and wellness tracking
- **Privacy-First Design**: Local processing with end-to-end encryption
- **Enterprise Ready**: Multi-user support with admin dashboards

## 🚀 What's New in V2.0

### ✨ **Major Enhancements**
- **🎨 Modern Native GUI**: Beautiful PySide6-based interface with dark theme
- **⚡ Optimized Performance**: 5x faster camera processing, 9x faster audio analysis
- **🧠 Advanced AI**: MediaPipe integration and TensorFlow optimization
- **📊 Real-time Analytics**: Live performance metrics and stress monitoring
- **🔧 Better Architecture**: Modular design with improved error handling
- **🎯 Hardware Integration**: Direct camera and microphone access
- **📱 Cross-Platform**: Enhanced compatibility across Windows, macOS, and Linux

## 🚀 Key Features

### 🎨 **Modern User Interface**
- **Native Desktop GUI**: Beautiful PySide6-based application with professional styling
- **Dark Theme**: Eye-friendly interface with modern design patterns
- **Real-time Dashboard**: Live stress monitoring with visual indicators
- **Tabbed Interface**: Organized layout for Camera Feed, Analytics, and Settings
- **Performance Metrics**: FPS monitoring and processing time display

### 🤖 **Advanced AI & Machine Learning**
- **Optimized AI Service**: MediaPipe integration for fast face detection
- **Facial Emotion Recognition**: CNN models with real-time processing
- **Voice Stress Analysis**: Advanced vocal pattern detection with minimal latency
- **Multi-modal Fusion**: Combined visual and audio analysis for accurate predictions
- **Performance Optimized**: <100ms processing targets achieved

### � **Technical Excellence**
- **Robust Architecture**: Modular design with comprehensive error handling
- **Hardware Integration**: Direct camera and microphone access via OpenCV and PyAudio
- **Cross-Platform Support**: Windows, macOS, and Linux compatibility
- **Virtual Environment**: Isolated dependency management
- **Memory Optimized**: Efficient resource usage and cleanup

## 📁 Project Structure

```
Stress-Burnout-Warning-System/
├── 🚀 main.py                      # Main application entry point (V2.0)
├── 📄 requirements.txt             # Python dependencies
├── � V2_ROADMAP.md               # Development roadmap and progress
├── �📂 src/                        # Source code modules
│   ├── ai/                        # AI/ML models and services
│   │   ├── optimized_ai_service.py    # Core AI processing engine
│   │   ├── conversational_ai.py       # AI chat interface
│   │   └── model_training.py          # Training utilities
│   ├── monitoring/                # Real-time monitoring modules
│   │   ├── facial_monitor.py          # Camera-based monitoring
│   │   ├── voice_monitor.py           # Audio analysis
│   │   └── typing_monitor.py          # Behavioral monitoring
│   ├── ui/                        # User interface components
│   │   ├── chat_interface.py          # Conversational UI
│   │   └── theme.py                   # UI styling and themes
│   ├── analysis/                  # Data analysis modules
│   ├── alerts/                    # Notification system
│   ├── config/                    # Configuration management
│   └── wellbeing/                 # Wellness features
├── 📂 models/                     # Trained ML models
├── 📂 config/                     # Configuration files
├── 📂 data/                       # Data storage and logs
└── 📂 docs/                       # Documentation and guides
```

##  Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yashsiwacha/Stress-Burnout-Warning.git
cd Stress-Burnout-Warning-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Launch the V2.0 stress monitoring system
python main.py
```

### 3. Using the System

1. **Camera Setup**: Ensure your camera is connected and accessible
2. **Microphone Setup**: Verify microphone permissions are granted
3. **Start Monitoring**: Click the "Start Monitoring" button in the GUI
4. **Real-time Analysis**: View live stress analysis and face detection
5. **Monitor Wellness**: Track your stress levels throughout the day

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