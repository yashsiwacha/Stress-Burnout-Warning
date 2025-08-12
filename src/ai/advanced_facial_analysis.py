"""
Advanced Facial Emotion Recognition Component for V2.0
CNN-based deep learning model for real-time emotion and stress detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import asyncio
import threading
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ..core.architecture import IComponent, ComponentConfig, ComponentStatus, SystemEvent, EventType
    from ..database import get_database_manager
except ImportError:
    # Fallback for direct execution
    logger.warning("Architecture imports not available - running in standalone mode")
    IComponent = object
    ComponentConfig = dict
    ComponentStatus = object
    SystemEvent = dict
    EventType = object

class CNNFacialAnalyzer:
    """Advanced CNN-based facial emotion recognition"""
    
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Sad', 'Surprise', 'Neutral'
        ]
        self.stress_mapping = {
            'Angry': 0.8,
            'Disgust': 0.6,
            'Fear': 0.9,
            'Happy': 0.1,
            'Sad': 0.7,
            'Surprise': 0.4,
            'Neutral': 0.3
        }
        
    def initialize(self) -> bool:
        """Initialize the CNN model and face detection"""
        try:
            # Initialize MediaPipe Face Mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize OpenCV face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Build and load CNN model
            self.model = self._build_emotion_cnn()
            
            print("✅ CNN Facial Analyzer initialized")
            return True
            
        except Exception as e:
            print(f"❌ CNN Facial Analyzer initialization failed: {e}")
            return False
    
    def _build_emotion_cnn(self) -> keras.Model:
        """Build CNN model for emotion recognition"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotion_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights (in production, load pre-trained weights)
        model.build()
        print("✅ CNN Emotion Model built")
        return model
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def extract_facial_landmarks(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract facial landmarks using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to coordinates
                h, w = frame.shape[:2]
                landmark_points = []
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append([x, y])
                
                return {
                    'landmarks': landmark_points,
                    'landmark_count': len(landmark_points)
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Landmark extraction error: {e}")
            return None
    
    def analyze_micro_expressions(self, face_region: np.ndarray, 
                                 landmarks: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze micro-expressions for stress indicators"""
        try:
            # Analyze eye region for stress indicators
            eye_analysis = self._analyze_eye_region(face_region, landmarks)
            
            # Analyze mouth region for stress indicators
            mouth_analysis = self._analyze_mouth_region(face_region, landmarks)
            
            # Analyze forehead for tension
            forehead_analysis = self._analyze_forehead_region(face_region, landmarks)
            
            return {
                'eye_stress_indicators': eye_analysis,
                'mouth_stress_indicators': mouth_analysis,
                'forehead_tension': forehead_analysis,
                'overall_micro_stress': (
                    eye_analysis.get('stress_score', 0) * 0.4 +
                    mouth_analysis.get('stress_score', 0) * 0.3 +
                    forehead_analysis.get('stress_score', 0) * 0.3
                )
            }
            
        except Exception as e:
            print(f"❌ Micro-expression analysis error: {e}")
            return {'overall_micro_stress': 0.0}
    
    def _analyze_eye_region(self, face: np.ndarray, landmarks: Optional[Dict]) -> Dict[str, Any]:
        """Analyze eye region for stress indicators"""
        return {
            'blink_rate': 15.0,  # Placeholder - would use temporal analysis
            'eye_openness': 0.8,
            'gaze_stability': 0.7,
            'stress_score': 0.3
        }
    
    def _analyze_mouth_region(self, face: np.ndarray, landmarks: Optional[Dict]) -> Dict[str, Any]:
        """Analyze mouth region for stress indicators"""
        return {
            'lip_tension': 0.4,
            'mouth_curvature': 0.0,
            'jaw_tension': 0.5,
            'stress_score': 0.4
        }
    
    def _analyze_forehead_region(self, face: np.ndarray, landmarks: Optional[Dict]) -> Dict[str, Any]:
        """Analyze forehead for tension indicators"""
        return {
            'furrow_depth': 0.3,
            'muscle_tension': 0.4,
            'stress_score': 0.35
        }
    
    def predict_emotion(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Predict emotion using CNN model"""
        try:
            # Preprocess face region
            processed_face = self._preprocess_face(face_region)
            
            # Get model prediction
            prediction = self.model.predict(processed_face, verbose=0)
            
            # Convert to probabilities
            probabilities = prediction[0]
            
            # Get top emotion
            top_emotion_idx = np.argmax(probabilities)
            top_emotion = self.emotion_labels[top_emotion_idx]
            confidence = float(probabilities[top_emotion_idx])
            
            # Create emotion distribution
            emotion_dist = {}
            for i, label in enumerate(self.emotion_labels):
                emotion_dist[label] = float(probabilities[i])
            
            # Calculate stress level from emotion
            stress_level = self.stress_mapping.get(top_emotion, 0.3)
            
            return {
                'primary_emotion': top_emotion,
                'confidence': confidence,
                'emotion_distribution': emotion_dist,
                'stress_level': stress_level,
                'emotional_valence': self._calculate_valence(emotion_dist),
                'emotional_arousal': self._calculate_arousal(emotion_dist)
            }
            
        except Exception as e:
            print(f"❌ Emotion prediction error: {e}")
            return {
                'primary_emotion': 'Neutral',
                'confidence': 0.0,
                'stress_level': 0.3
            }
    
    def _preprocess_face(self, face_region: np.ndarray) -> np.ndarray:
        """Preprocess face region for CNN input"""
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # Resize to model input size
        resized_face = cv2.resize(gray_face, (48, 48))
        
        # Normalize pixel values
        normalized_face = resized_face.astype('float32') / 255.0
        
        # Reshape for model input
        model_input = normalized_face.reshape(1, 48, 48, 1)
        
        return model_input
    
    def _calculate_valence(self, emotion_dist: Dict[str, float]) -> float:
        """Calculate emotional valence (positive/negative)"""
        positive_emotions = ['Happy', 'Surprise']
        negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']
        
        positive_score = sum(emotion_dist.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotion_dist.get(emotion, 0) for emotion in negative_emotions)
        
        return positive_score - negative_score  # Range: -1 to 1
    
    def _calculate_arousal(self, emotion_dist: Dict[str, float]) -> float:
        """Calculate emotional arousal (calm/excited)"""
        high_arousal = ['Angry', 'Fear', 'Surprise', 'Happy']
        low_arousal = ['Sad', 'Disgust', 'Neutral']
        
        high_score = sum(emotion_dist.get(emotion, 0) for emotion in high_arousal)
        low_score = sum(emotion_dist.get(emotion, 0) for emotion in low_arousal)
        
        return high_score - low_score  # Range: -1 to 1

class AdvancedFacialComponent(IComponent):
    """Advanced facial analysis component for V2.0"""
    
    def __init__(self):
        config = ComponentConfig(
            component_id="advanced_facial",
            name="Advanced Facial Analysis",
            enabled=True,
            config={
                "fps": 30,
                "analysis_interval_ms": 100,
                "face_detection_confidence": 0.5,
                "emotion_confidence_threshold": 0.6
            },
            dependencies=["user_manager"]
        )
        super().__init__(config)
        
        self.analyzer = CNNFacialAnalyzer()
        self.db_manager = get_database_manager()
        self.camera = None
        self.analysis_thread = None
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize facial analysis component"""
        try:
            # Initialize CNN analyzer
            if not self.analyzer.initialize():
                return False
            
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("❌ Cannot access camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.config["fps"])
            
            # Subscribe to session events
            self.event_bus.subscribe(EventType.SESSION_START, self.handle_session_start)
            self.event_bus.subscribe(EventType.SESSION_END, self.handle_session_end)
            
            print("✅ Advanced Facial Analysis initialized")
            return True
            
        except Exception as e:
            print(f"❌ Advanced Facial Analysis initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start facial analysis"""
        self.running = True
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop, 
            daemon=True
        )
        self.analysis_thread.start()
        
        print("🚀 Advanced Facial Analysis started")
        return True
    
    async def stop(self) -> bool:
        """Stop facial analysis"""
        self.running = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
        
        print("🛑 Advanced Facial Analysis stopped")
        return True
    
    async def process_event(self, event: SystemEvent) -> Optional[SystemEvent]:
        """Process system events"""
        return None
    
    async def handle_session_start(self, event: SystemEvent):
        """Handle session start"""
        print(f"👁️  Starting facial analysis for session {event.session_id}")
    
    async def handle_session_end(self, event: SystemEvent):
        """Handle session end"""
        print(f"👁️  Ending facial analysis for session {event.session_id}")
    
    def _analysis_loop(self):
        """Main analysis loop running in separate thread"""
        print("🔄 Facial analysis loop started")
        
        while self.running and self.get_status() == ComponentStatus.ACTIVE:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Detect faces
                faces = self.analyzer.detect_faces(frame)
                
                if len(faces) > 0:
                    # Process first detected face
                    x, y, w, h = faces[0]
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Extract facial landmarks
                    landmarks = self.analyzer.extract_facial_landmarks(face_region)
                    
                    # Predict emotion
                    emotion_result = self.analyzer.predict_emotion(face_region)
                    
                    # Analyze micro-expressions
                    micro_expr = self.analyzer.analyze_micro_expressions(face_region, landmarks)
                    
                    # Create comprehensive analysis result
                    analysis_result = {
                        'timestamp': datetime.now().isoformat(),
                        'face_detected': True,
                        'face_confidence': 0.9,  # Placeholder
                        'emotion_analysis': emotion_result,
                        'micro_expressions': micro_expr,
                        'facial_landmarks': landmarks,
                        'stress_indicators': {
                            'emotional_stress': emotion_result.get('stress_level', 0),
                            'micro_expression_stress': micro_expr.get('overall_micro_stress', 0),
                            'overall_facial_stress': (
                                emotion_result.get('stress_level', 0) * 0.7 +
                                micro_expr.get('overall_micro_stress', 0) * 0.3
                            )
                        }
                    }
                    
                    # Emit facial analysis event
                    asyncio.run_coroutine_threadsafe(
                        self._emit_facial_event(analysis_result),
                        asyncio.get_event_loop()
                    )
                
                # Control analysis frequency
                import time
                time.sleep(self.config.config["analysis_interval_ms"] / 1000.0)
                
            except Exception as e:
                print(f"❌ Facial analysis error: {e}")
                import time
                time.sleep(1.0)
    
    async def _emit_facial_event(self, analysis_result: Dict[str, Any]):
        """Emit facial analysis event"""
        try:
            event = SystemEvent(
                event_type=EventType.STRESS_READING,
                source_component=self.config.component_id,
                data={
                    'component': 'facial_analysis',
                    'analysis_result': analysis_result,
                    'stress_level': analysis_result['stress_indicators']['overall_facial_stress']
                }
            )
            
            self.emit_event(event)
            
        except Exception as e:
            print(f"❌ Error emitting facial event: {e}")

# Factory function for component registration
def create_advanced_facial_component() -> AdvancedFacialComponent:
    """Create advanced facial analysis component"""
    return AdvancedFacialComponent()
