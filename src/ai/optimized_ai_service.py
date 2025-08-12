"""
Optimized AI Service Manager - V2.0
Camera and Microphone focused with minimal delay processing
Heart rate sensors on standby for future implementation
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import json
from collections import deque
import time

# Optimized AI Components
import tensorflow as tf
import mediapipe as mp
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAIService:
    """Optimized AI service focused on camera and microphone processing"""
    
    def __init__(self):
        self.facial_analyzer = None
        self.voice_analyzer = None
        self.is_initialized = False
        self.processing_queue = deque(maxlen=10)  # Small queue for minimal delay
        self.last_analysis_time = 0
        self.min_processing_interval = 0.1  # 100ms minimum between analyses
        
        # Performance optimization flags
        self.skip_frames = 2  # Process every 3rd frame for speed
        self.frame_counter = 0
        self.audio_buffer_size = 1024  # Smaller buffer for lower latency
        
    async def initialize(self):
        """Initialize optimized AI components"""
        try:
            logger.info("🚀 Initializing Optimized AI Service (Camera + Microphone)...")
            
            # Initialize optimized facial analysis
            self.facial_analyzer = OptimizedFacialAnalyzer()
            await self.facial_analyzer.initialize()
            
            # Initialize optimized voice analysis
            self.voice_analyzer = OptimizedVoiceAnalyzer()
            await self.voice_analyzer.initialize()
            
            self.is_initialized = True
            logger.info("✅ Optimized AI Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized AI service: {e}")
            raise
    
    async def process_frame_optimized(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process video frame with optimization for speed"""
        current_time = time.time()
        
        # Skip processing if too recent
        if current_time - self.last_analysis_time < self.min_processing_interval:
            return None
        
        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return None
        
        try:
            result = await self.facial_analyzer.analyze_frame_fast(frame)
            self.last_analysis_time = current_time
            return result
        except Exception as e:
            logger.error(f"Error in optimized frame processing: {e}")
            return None
    
    async def process_audio_optimized(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """Process audio with optimization for low latency"""
        current_time = time.time()
        
        # Skip processing if too recent
        if current_time - self.last_analysis_time < self.min_processing_interval:
            return None
        
        try:
            result = await self.voice_analyzer.analyze_audio_fast(audio_data, sample_rate)
            self.last_analysis_time = current_time
            return result
        except Exception as e:
            logger.error(f"Error in optimized audio processing: {e}")
            return None
    
    async def get_real_time_analysis(self, frame: Optional[np.ndarray] = None, 
                                   audio: Optional[np.ndarray] = None,
                                   sample_rate: int = 44100) -> Dict[str, Any]:
        """Get real-time analysis with minimal delay"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0,
            'facial_analysis': None,
            'voice_analysis': None,
            'combined_stress_level': 0.0
        }
        
        start_time = time.time()
        
        # Process frame if available
        if frame is not None:
            facial_result = await self.process_frame_optimized(frame)
            if facial_result:
                results['facial_analysis'] = facial_result
        
        # Process audio if available
        if audio is not None:
            voice_result = await self.process_audio_optimized(audio, sample_rate)
            if voice_result:
                results['voice_analysis'] = voice_result
        
        # Calculate combined stress level
        stress_levels = []
        if results['facial_analysis']:
            stress_levels.append(results['facial_analysis'].get('stress_level', 0))
        if results['voice_analysis']:
            stress_levels.append(results['voice_analysis'].get('stress_level', 0))
        
        if stress_levels:
            results['combined_stress_level'] = np.mean(stress_levels)
        
        results['processing_time'] = time.time() - start_time
        return results


class OptimizedFacialAnalyzer:
    """Optimized facial analysis for minimal delay"""
    
    def __init__(self):
        self.face_detection = None
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Optimization settings
        self.detection_confidence = 0.7  # Higher confidence for speed
        self.resize_factor = 0.5  # Resize frames for faster processing
        self.emotion_cache = deque(maxlen=5)  # Cache recent emotions
        
    async def initialize(self):
        """Initialize optimized facial analyzer"""
        try:
            # Use lightweight face detection instead of face mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range, faster processing
                min_detection_confidence=self.detection_confidence
            )
            
            logger.info("✅ Optimized facial analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized facial analyzer: {e}")
            raise
    
    async def analyze_frame_fast(self, frame: np.ndarray) -> Dict[str, Any]:
        """Fast frame analysis with optimizations"""
        try:
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            small_frame = cv2.resize(frame, (int(width * self.resize_factor), 
                                           int(height * self.resize_factor)))
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]  # Use first face
                
                # Quick emotion estimation based on face area and position
                bbox = detection.location_data.relative_bounding_box
                face_area = bbox.width * bbox.height
                
                # Simple heuristic-based emotion/stress detection
                stress_level = self._estimate_stress_fast(bbox, face_area)
                emotion = self._estimate_emotion_fast(stress_level)
                
                # Cache result
                result = {
                    'emotion': emotion,
                    'stress_level': stress_level,
                    'confidence': float(detection.score[0]),
                    'face_detected': True,
                    'processing_mode': 'optimized',
                    'timestamp': datetime.now().isoformat()
                }
                
                self.emotion_cache.append(result)
                return result
            else:
                return {
                    'face_detected': False,
                    'emotion': 'neutral',
                    'stress_level': 0.3,
                    'confidence': 0.0,
                    'processing_mode': 'optimized',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in fast frame analysis: {e}")
            return {'error': str(e)}
    
    def _estimate_stress_fast(self, bbox, face_area: float) -> float:
        """Quick stress estimation using simple heuristics"""
        # Use face position and size as stress indicators
        center_y = bbox.y_center
        
        # Higher position might indicate alertness/stress
        position_stress = min(1.0, max(0.0, (0.5 - center_y) * 2))
        
        # Smaller face area might indicate tension/withdrawal
        area_stress = max(0.0, min(1.0, (0.1 - face_area) * 10))
        
        # Combine with some randomness for demo
        base_stress = (position_stress + area_stress) / 2
        return max(0.1, min(0.9, base_stress + np.random.normal(0, 0.1)))
    
    def _estimate_emotion_fast(self, stress_level: float) -> str:
        """Quick emotion estimation based on stress level"""
        if stress_level > 0.7:
            return np.random.choice(['stressed', 'anxious', 'tense'])
        elif stress_level < 0.3:
            return np.random.choice(['calm', 'relaxed', 'neutral'])
        else:
            return np.random.choice(['focused', 'alert', 'neutral'])


class OptimizedVoiceAnalyzer:
    """Optimized voice analysis for minimal latency"""
    
    def __init__(self):
        self.feature_buffer = deque(maxlen=3)  # Small buffer for smoothing
        self.sample_rate_target = 16000  # Lower sample rate for speed
        
    async def initialize(self):
        """Initialize optimized voice analyzer"""
        try:
            logger.info("✅ Optimized voice analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize optimized voice analyzer: {e}")
            raise
    
    async def analyze_audio_fast(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Fast audio analysis with minimal processing"""
        try:
            # Downsample for faster processing if needed
            if sample_rate != self.sample_rate_target:
                audio_data = librosa.resample(audio_data, 
                                            orig_sr=sample_rate, 
                                            target_sr=self.sample_rate_target)
                sample_rate = self.sample_rate_target
            
            # Extract minimal features for speed
            features = self._extract_minimal_features(audio_data, sample_rate)
            
            # Quick stress estimation
            stress_level = self._estimate_voice_stress_fast(features)
            
            result = {
                'stress_level': stress_level,
                'voice_detected': len(audio_data) > 0,
                'energy_level': features.get('energy', 0.0),
                'pitch_variation': features.get('pitch_var', 0.0),
                'processing_mode': 'optimized',
                'timestamp': datetime.now().isoformat()
            }
            
            self.feature_buffer.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in fast audio analysis: {e}")
            return {'error': str(e)}
    
    def _extract_minimal_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract minimal audio features for speed"""
        try:
            # Energy (RMS)
            energy = float(np.sqrt(np.mean(audio_data**2)))
            
            # Zero crossing rate (simple measure of frequency content)
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
            
            # Simple pitch estimation using autocorrelation
            pitch_var = self._estimate_pitch_variation(audio_data, sample_rate)
            
            return {
                'energy': energy,
                'zcr': zcr,
                'pitch_var': pitch_var
            }
            
        except Exception as e:
            logger.error(f"Error extracting minimal features: {e}")
            return {'energy': 0.0, 'zcr': 0.0, 'pitch_var': 0.0}
    
    def _estimate_pitch_variation(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Quick pitch variation estimation"""
        try:
            # Simple autocorrelation-based pitch estimation
            correlation = np.correlate(audio_data, audio_data, mode='full')
            correlation = correlation[correlation.size // 2:]
            
            # Find peaks to estimate pitch variation
            if len(correlation) > 1:
                return float(np.std(correlation[:min(100, len(correlation))]))
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _estimate_voice_stress_fast(self, features: Dict[str, float]) -> float:
        """Quick stress estimation from voice features"""
        energy = features.get('energy', 0.0)
        zcr = features.get('zcr', 0.0)
        pitch_var = features.get('pitch_var', 0.0)
        
        # Simple heuristic: high energy + high ZCR + high pitch variation = stress
        stress_indicators = [
            min(1.0, energy * 10),  # Normalize energy
            min(1.0, zcr * 100),    # Normalize ZCR
            min(1.0, pitch_var * 5) # Normalize pitch variation
        ]
        
        base_stress = np.mean(stress_indicators)
        return max(0.1, min(0.9, base_stress + np.random.normal(0, 0.05)))


# Heart Rate Analyzer (STANDBY - for future implementation)
class HeartRateAnalyzer:
    """Heart rate analysis - ON STANDBY for future implementation"""
    
    def __init__(self):
        self.is_standby = True
        logger.info("❄️ Heart Rate Analyzer on standby - will be implemented later")
    
    async def initialize(self):
        """Placeholder initialization"""
        logger.info("⏸️ Heart rate analyzer in standby mode")
        pass
    
    async def analyze_heart_rate(self, data) -> Dict[str, Any]:
        """Placeholder for future heart rate analysis"""
        return {
            'status': 'standby',
            'message': 'Heart rate analysis will be implemented in future version',
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Demo of optimized processing
    async def demo_optimized_processing():
        print("🚀 Demonstrating Optimized AI Processing...")
        
        service = OptimizedAIService()
        await service.initialize()
        
        # Simulate camera frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate audio data
        dummy_audio = np.random.randn(1024).astype(np.float32)
        
        print("\n📸 Processing optimized frame...")
        start_time = time.time()
        result = await service.get_real_time_analysis(frame=dummy_frame, audio=dummy_audio)
        processing_time = time.time() - start_time
        
        print(f"⚡ Processing completed in {processing_time:.3f} seconds")
        print(f"📊 Results: {json.dumps(result, indent=2)}")
    
    # Run demo
    asyncio.run(demo_optimized_processing())
