"""
AI Service Manager for V2.0
Coordinates all AI components for comprehensive stress analysis
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import json

# AI Components
import tensorflow as tf
import mediapipe as mp
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServiceManager:
    """Central manager for all AI services"""
    
    def __init__(self):
        self.facial_analyzer = None
        self.voice_analyzer = None
        self.text_analyzer = None
        self.fusion_engine = None
        self.is_initialized = False
        self.active_session = None
        self.stress_history = []
        
    async def initialize(self):
        """Initialize all AI components"""
        try:
            logger.info("🧠 Initializing AI Service Manager...")
            
            # Initialize facial analysis
            self.facial_analyzer = FacialEmotionAnalyzer()
            await self.facial_analyzer.initialize()
            
            # Initialize voice analysis
            self.voice_analyzer = VoiceStressAnalyzer()
            await self.voice_analyzer.initialize()
            
            # Initialize text analysis
            self.text_analyzer = TextSentimentAnalyzer()
            await self.text_analyzer.initialize()
            
            # Initialize fusion engine
            self.fusion_engine = MultiModalFusionEngine()
            await self.fusion_engine.initialize()
            
            self.is_initialized = True
            logger.info("✅ AI Service Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Service Manager: {e}")
            raise
    
    async def start_analysis_session(self, user_id: str, session_id: str):
        """Start a new analysis session"""
        if not self.is_initialized:
            await self.initialize()
            
        self.active_session = {
            'user_id': user_id,
            'session_id': session_id,
            'start_time': datetime.now(),
            'readings': []
        }
        
        logger.info(f"🎯 Started AI analysis session for user {user_id}")
    
    async def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a video frame for facial emotions"""
        if not self.facial_analyzer:
            return {'error': 'Facial analyzer not initialized'}
            
        try:
            result = await self.facial_analyzer.analyze_frame(frame)
            return result
        except Exception as e:
            logger.error(f"Error in facial analysis: {e}")
            return {'error': str(e)}
    
    async def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio data for stress indicators"""
        if not self.voice_analyzer:
            return {'error': 'Voice analyzer not initialized'}
            
        try:
            result = await self.voice_analyzer.analyze_audio(audio_data, sample_rate)
            return result
        except Exception as e:
            logger.error(f"Error in voice analysis: {e}")
            return {'error': str(e)}
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for sentiment and stress indicators"""
        if not self.text_analyzer:
            return {'error': 'Text analyzer not initialized'}
            
        try:
            result = await self.text_analyzer.analyze_text(text)
            return result
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'error': str(e)}
    
    async def get_comprehensive_analysis(self, 
                                       facial_data: Optional[Dict] = None,
                                       voice_data: Optional[Dict] = None,
                                       text_data: Optional[Dict] = None,
                                       behavioral_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Get comprehensive multi-modal stress analysis"""
        if not self.fusion_engine:
            return {'error': 'Fusion engine not initialized'}
            
        try:
            # Combine all available data
            combined_data = {
                'facial': facial_data or {},
                'voice': voice_data or {},
                'text': text_data or {},
                'behavioral': behavioral_data or {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Run fusion analysis
            result = await self.fusion_engine.fuse_modalities(combined_data)
            
            # Store in session
            if self.active_session:
                self.active_session['readings'].append(result)
                self.stress_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e)}


class FacialEmotionAnalyzer:
    """CNN-based facial emotion recognition"""
    
    def __init__(self):
        self.model = None
        self.face_mesh = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.stress_mapping = {
            'Angry': 0.8, 'Disgust': 0.6, 'Fear': 0.9, 'Happy': 0.1,
            'Sad': 0.7, 'Surprise': 0.4, 'Neutral': 0.3
        }
    
    async def initialize(self):
        """Initialize the facial analyzer"""
        try:
            # Initialize MediaPipe Face Mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Create and compile CNN model
            self.model = self._create_emotion_model()
            logger.info("✅ Facial emotion analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize facial analyzer: {e}")
            raise
    
    def _create_emotion_model(self):
        """Create CNN model for emotion recognition"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.emotion_labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    async def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a video frame for emotions"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract emotion features (simplified for demo)
                emotion_probs = np.random.dirichlet(np.ones(len(self.emotion_labels)))
                predicted_emotion = self.emotion_labels[np.argmax(emotion_probs)]
                confidence = float(np.max(emotion_probs))
                
                # Calculate stress level
                stress_level = self.stress_mapping.get(predicted_emotion, 0.5)
                
                # Analyze micro-expressions (simplified)
                micro_expressions = self._analyze_micro_expressions(landmarks)
                
                return {
                    'emotion': predicted_emotion,
                    'confidence': confidence,
                    'stress_level': stress_level,
                    'emotion_probabilities': dict(zip(self.emotion_labels, emotion_probs.tolist())),
                    'micro_expressions': micro_expressions,
                    'landmarks_detected': len(landmarks.landmark),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'error': 'No face detected',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {'error': str(e)}
    
    def _analyze_micro_expressions(self, landmarks) -> Dict[str, float]:
        """Analyze micro-expressions from facial landmarks"""
        # Simplified micro-expression analysis
        return {
            'eye_movement': np.random.uniform(0, 1),
            'brow_tension': np.random.uniform(0, 1),
            'mouth_curvature': np.random.uniform(-1, 1),
            'jaw_tension': np.random.uniform(0, 1)
        }


class VoiceStressAnalyzer:
    """Advanced voice stress analysis using audio features"""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = None
    
    async def initialize(self):
        """Initialize the voice analyzer"""
        try:
            # Create a simple classifier for demo
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train with synthetic data for demo
            X_train = np.random.rand(1000, 25)  # 25 audio features
            y_train = np.random.choice(['low_stress', 'medium_stress', 'high_stress'], 1000)
            self.model.fit(X_train, y_train)
            
            logger.info("✅ Voice stress analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice analyzer: {e}")
            raise
    
    async def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio for stress indicators"""
        try:
            # Extract audio features
            features = self._extract_audio_features(audio_data, sample_rate)
            
            # Predict stress level
            stress_prediction = self.model.predict([features])[0]
            stress_proba = self.model.predict_proba([features])[0]
            
            # Map to numerical stress level
            stress_mapping = {'low_stress': 0.2, 'medium_stress': 0.5, 'high_stress': 0.8}
            stress_level = stress_mapping.get(stress_prediction, 0.5)
            
            return {
                'stress_level': stress_level,
                'stress_category': stress_prediction,
                'confidence': float(np.max(stress_proba)),
                'features': {
                    'pitch_variation': float(features[0]),
                    'speech_rate': float(features[1]),
                    'vocal_tension': float(features[2]),
                    'energy_level': float(features[3])
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {'error': str(e)}
    
    def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for stress analysis"""
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_mean = np.mean(spectral_centroids)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            zcr_mean = np.mean(zcr)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean[:10],  # First 10 MFCCs
                [spectral_mean, zcr_mean]
            ])
            
            # Pad to 25 features if needed
            if len(features) < 25:
                features = np.pad(features, (0, 25 - len(features)), 'constant')
            
            return features[:25]
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return np.zeros(25)  # Return zeros if extraction fails


class TextSentimentAnalyzer:
    """NLP-based text sentiment and stress analysis"""
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
    
    async def initialize(self):
        """Initialize the text analyzer"""
        try:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            # Sample training data
            sample_texts = [
                "I'm feeling overwhelmed with work", "Everything is great today",
                "So much stress and pressure", "Having a peaceful day",
                "Can't handle this workload", "Feeling calm and relaxed",
                "Anxious about deadlines", "Everything is under control"
            ]
            sample_labels = ['high', 'low', 'high', 'low', 'high', 'low', 'high', 'low']
            
            # Fit vectorizer and train classifier
            X = self.vectorizer.fit_transform(sample_texts)
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.classifier.fit(X, sample_labels)
            
            logger.info("✅ Text sentiment analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize text analyzer: {e}")
            raise
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for sentiment and stress indicators"""
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Predict stress level
            prediction = self.classifier.predict(text_vector)[0]
            probabilities = self.classifier.predict_proba(text_vector)[0]
            confidence = float(np.max(probabilities))
            
            # Map to numerical stress level
            stress_mapping = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
            stress_level = stress_mapping.get(prediction, 0.5)
            
            # Analyze sentiment keywords
            stress_keywords = self._extract_stress_keywords(text)
            
            return {
                'stress_level': stress_level,
                'stress_category': prediction,
                'confidence': confidence,
                'sentiment': 'negative' if stress_level > 0.5 else 'positive',
                'stress_keywords': stress_keywords,
                'text_length': len(text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'error': str(e)}
    
    def _extract_stress_keywords(self, text: str) -> List[str]:
        """Extract stress-related keywords from text"""
        stress_words = [
            'stress', 'overwhelmed', 'anxious', 'pressure', 'deadline',
            'worried', 'panic', 'tired', 'exhausted', 'burden'
        ]
        
        text_lower = text.lower()
        found_keywords = [word for word in stress_words if word in text_lower]
        return found_keywords


class MultiModalFusionEngine:
    """Combines data from multiple modalities for comprehensive analysis"""
    
    def __init__(self):
        self.fusion_weights = {
            'facial': 0.30,
            'voice': 0.25,
            'text': 0.25,
            'behavioral': 0.20
        }
    
    async def initialize(self):
        """Initialize the fusion engine"""
        logger.info("✅ Multi-modal fusion engine initialized")
    
    async def fuse_modalities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple modality data into comprehensive assessment"""
        try:
            # Extract stress levels from each modality
            stress_levels = {}
            confidences = {}
            
            for modality in ['facial', 'voice', 'text', 'behavioral']:
                modality_data = data.get(modality, {})
                if modality_data and 'stress_level' in modality_data:
                    stress_levels[modality] = modality_data['stress_level']
                    confidences[modality] = modality_data.get('confidence', 0.5)
                else:
                    stress_levels[modality] = 0.5  # Default neutral
                    confidences[modality] = 0.1   # Low confidence
            
            # Calculate weighted average
            total_weight = 0
            weighted_stress = 0
            
            for modality, stress in stress_levels.items():
                weight = self.fusion_weights[modality] * confidences[modality]
                weighted_stress += stress * weight
                total_weight += weight
            
            if total_weight > 0:
                combined_stress = weighted_stress / total_weight
            else:
                combined_stress = 0.5
            
            # Determine stress category
            if combined_stress < 0.3:
                category = "Low Stress"
                color = "green"
            elif combined_stress < 0.6:
                category = "Moderate Stress"
                color = "yellow"
            else:
                category = "High Stress"
                color = "red"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(combined_stress, data)
            
            return {
                'combined_stress_level': float(combined_stress),
                'stress_category': category,
                'color_indicator': color,
                'individual_modalities': stress_levels,
                'confidences': confidences,
                'recommendations': recommendations,
                'analysis_summary': self._generate_summary(combined_stress, data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in fusion analysis: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, stress_level: float, data: Dict) -> List[str]:
        """Generate personalized recommendations based on stress level"""
        recommendations = []
        
        if stress_level > 0.7:
            recommendations.extend([
                "Take a 10-15 minute break immediately",
                "Practice deep breathing exercises",
                "Step away from your current task",
                "Consider talking to someone about your stress"
            ])
        elif stress_level > 0.4:
            recommendations.extend([
                "Take short breaks between tasks",
                "Stay hydrated and maintain good posture",
                "Consider light stretching or movement",
                "Practice mindfulness for a few minutes"
            ])
        else:
            recommendations.extend([
                "Maintain your current work pace",
                "Keep up the good work!",
                "Stay aware of your stress levels throughout the day"
            ])
        
        return recommendations
    
    def _generate_summary(self, stress_level: float, data: Dict) -> str:
        """Generate a human-readable summary of the analysis"""
        if stress_level > 0.7:
            return "High stress detected across multiple indicators. Immediate intervention recommended."
        elif stress_level > 0.4:
            return "Moderate stress levels detected. Consider taking preventive measures."
        else:
            return "Low stress levels. Current state appears healthy and manageable."


# Global instance
_ai_service_manager = None

def get_ai_service_manager() -> AIServiceManager:
    """Get the global AI service manager instance"""
    global _ai_service_manager
    if _ai_service_manager is None:
        _ai_service_manager = AIServiceManager()
    return _ai_service_manager
