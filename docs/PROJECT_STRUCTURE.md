# Project Structure

This document outlines the organized structure of the Stress Burnout Warning System project.

## 📁 Root Directory Structure

```
Stress-Burnout-Warning-System/
├── 📄 main.py                    # Main application entry point
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project overview and setup guide
├── 📄 .gitignore                 # Git ignore rules
│
├── 📁 src/                       # Source code modules
│   ├── 📁 ai/                    # AI and ML components
│   ├── 📁 alerts/                # Notification system
│   ├── 📁 analysis/              # Stress analysis algorithms
│   ├── 📁 config/                # Configuration management
│   ├── 📁 data/                  # Data handling and logging
│   ├── 📁 monitoring/            # Real-time monitoring modules
│   ├── 📁 ui/                    # User interface components
│   └── 📁 wellbeing/             # Wellness resources
│
├── 📁 datasets/                  # Training and test datasets
│   ├── 📁 facial_emotion/        # Facial expression datasets
│   ├── 📁 vocal_emotion/         # Voice emotion datasets
│   ├── 📁 custom/                # Custom collected data
│   └── 📁 processed/             # Preprocessed data
│
├── 📁 models/                    # Trained ML models
│   ├── 📄 *.pkl                  # Scikit-learn models
│   ├── 📄 *.h5                   # TensorFlow/Keras models
│   └── 📄 metadata.json          # Model metadata
│
├── 📁 config/                    # Configuration files
│   └── 📄 *.json                 # JSON configuration files
│
├── 📁 data/                      # Runtime data and logs
│   ├── 📁 logs/                  # Application logs
│   └── 📁 user_data/             # User-specific data
│
├── 📁 docs/                      # Documentation
│   ├── 📄 AI_MODEL_TRAINING.md   # ML model training guide
│   ├── 📄 CNN_NLP_ARCHITECTURE.md # Technical architecture
│   ├── 📄 DATASET_TRAINING_GUIDE.md # Dataset usage guide
│   └── 📄 *.md                   # Other documentation
│
├── 📁 scripts/                   # Setup and utility scripts
│   ├── 📄 setup_project.py       # Project setup script
│   ├── 📄 dataset_downloader.py  # Dataset download utility
│   ├── 📄 train_complete_system.py # Model training script
│   └── 📄 *.py                   # Other utility scripts
│
├── 📁 demos/                     # Demo and example files
│   ├── 📄 demo_ml_system.py      # ML system demo
│   ├── 📄 demo_facial_training.py # Facial training demo
│   └── 📄 *.py                   # Other demo files
│
├── 📁 tools/                     # Development tools
│   ├── 📄 launcher.py            # Application launcher
│   ├── 📄 run_app.sh             # Shell script launcher
│   └── 📄 run_app.bat            # Windows batch launcher
│
├── 📁 archive/                   # Archived/backup files
│   └── 📄 main_*.py              # Old versions of main files
│
├── 📁 Plan/                      # Project planning documents
│   └── 📄 *.pdf                  # Project planning PDFs
│
└── 📁 .venv/                     # Python virtual environment
    └── ...                       # Virtual environment files
```

## 🎯 Key Components

### **Core Application**
- `main.py` - Main application with GUI and monitoring
- `src/` - Modular source code organization
- `requirements.txt` - All Python dependencies

### **Data & Models**
- `datasets/` - Training datasets for ML models
- `models/` - Trained machine learning models
- `data/` - Runtime data and user logs

### **Documentation**
- `docs/` - Comprehensive project documentation
- `README.md` - Main project overview

### **Development Tools**
- `scripts/` - Setup and utility scripts
- `demos/` - Example implementations
- `tools/` - Development and deployment tools

### **Project Management**
- `archive/` - Historical versions and backups
- `Plan/` - Project planning documents
- `.venv/` - Isolated Python environment

## 🚀 Quick Start

1. **Setup Environment**: `python scripts/setup_project.py`
2. **Download Datasets**: `python scripts/dataset_downloader.py --download-all`
3. **Train Models**: `python scripts/train_complete_system.py`
4. **Run Application**: `python main.py`

## 📖 Documentation Index

- **Technical**: `docs/CNN_NLP_ARCHITECTURE.md`
- **Training**: `docs/AI_MODEL_TRAINING.md`
- **Datasets**: `docs/DATASET_TRAINING_GUIDE.md`
- **Implementation**: `docs/IMPLEMENTATION_SUMMARY.md`

This organized structure ensures maintainability, scalability, and professional development practices.
