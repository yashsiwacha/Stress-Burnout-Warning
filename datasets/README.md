# Datasets for Stress Burnout Warning System

This directory contains all datasets used for training and testing the stress detection models.

## Directory Structure

### Facial Emotion Recognition Datasets
- `facial_emotion/fer2013/` - FER-2013 dataset (35,887 grayscale 48x48 images)
- `facial_emotion/affectnet/` - AffectNet dataset (1M+ facial images)
- `facial_emotion/ck_plus/` - Extended Cohn-Kanade dataset

### Vocal Emotion/Stress Detection Datasets
- `vocal_emotion/ravdess/` - RAVDESS dataset (24 actors, 7 emotions)
- `vocal_emotion/savee/` - SAVEE dataset (4 male actors, 7 emotions)
- `vocal_emotion/iemocap/` - IEMOCAP dataset (multimodal dyadic interactions)

### Custom Data
- `custom/` - Custom collected data with self-reported stress levels

### Processed Data
- `processed/facial/` - Preprocessed and augmented facial data
- `processed/vocal/` - Preprocessed audio features and spectrograms

## Dataset Download Instructions

### FER-2013
1. Download from: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract to `facial_emotion/fer2013/`
3. Expected structure:
   ```
   fer2013/
   ├── train/
   │   ├── angry/
   │   ├── disgusted/
   │   ├── fearful/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprised/
   └── test/
       ├── angry/
       ├── disgusted/
       ├── fearful/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprised/
   ```

### RAVDESS
1. Download from: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
2. Extract to `vocal_emotion/ravdess/`
3. Audio files should be in format: Actor_XX-emotion-intensity.wav

### Data Privacy and Ethics
- All data processing follows ethical guidelines
- No personal data is stored without consent
- Custom data collection requires explicit user permission
- Local processing ensures privacy protection

## Usage

Use the `src/data/dataset_loader.py` module to load and preprocess datasets for training.

```python
from src.data.dataset_loader import DatasetLoader

# Load facial emotion dataset
facial_loader = DatasetLoader('facial')
X_train, X_test, y_train, y_test = facial_loader.load_fer2013()

# Load vocal emotion dataset
vocal_loader = DatasetLoader('vocal')
X_audio_train, X_audio_test, y_audio_train, y_audio_test = vocal_loader.load_ravdess()
```
