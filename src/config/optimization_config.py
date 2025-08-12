"""
Optimization Configuration for V2.0 Stress Monitoring System
Camera and Microphone focused processing with minimal delay
"""

import json
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class CameraOptimization:
    """Camera processing optimization settings"""
    target_fps: int = 30
    resolution_width: int = 640
    resolution_height: int = 480
    resize_factor: float = 0.5  # Resize frames for faster processing
    skip_frames: int = 2  # Process every 3rd frame
    detection_confidence: float = 0.7  # Higher confidence for speed
    max_faces: int = 1  # Process only one face for speed

@dataclass
class MicrophoneOptimization:
    """Microphone processing optimization settings"""
    sample_rate: int = 16000  # Lower sample rate for speed
    buffer_size: int = 1024  # Smaller buffer for lower latency
    min_audio_length: float = 0.1  # Minimum audio length in seconds
    feature_cache_size: int = 3  # Cache recent features for smoothing

@dataclass
class ProcessingOptimization:
    """General processing optimization settings"""
    min_processing_interval: float = 0.1  # 100ms minimum between analyses
    max_queue_size: int = 10  # Small queue for minimal delay
    enable_async_processing: bool = True
    use_lightweight_models: bool = True
    enable_result_caching: bool = True
    cache_size: int = 5

@dataclass
class HeartRateConfig:
    """Heart rate sensor configuration (ON STANDBY)"""
    enabled: bool = False  # Currently disabled
    standby_message: str = "Heart rate sensors on standby for future implementation"
    planned_implementation: str = "Future version will include heart rate analysis"

@dataclass
class OptimizationConfig:
    """Complete optimization configuration"""
    camera: CameraOptimization
    microphone: MicrophoneOptimization
    processing: ProcessingOptimization
    heart_rate: HeartRateConfig
    
    # Performance targets
    target_processing_time: float = 0.1  # Target: under 100ms
    target_total_latency: float = 0.2  # Target: under 200ms total
    
    # Feature flags
    enable_performance_monitoring: bool = True
    enable_adaptive_quality: bool = True  # Adapt quality based on performance
    enable_debug_output: bool = False

# Default optimized configuration
DEFAULT_CONFIG = OptimizationConfig(
    camera=CameraOptimization(),
    microphone=MicrophoneOptimization(),
    processing=ProcessingOptimization(),
    heart_rate=HeartRateConfig()
)

class ConfigManager:
    """Manager for optimization configuration"""
    
    def __init__(self, config_path: str = "config/optimization_config.json"):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG
    
    def load_config(self) -> OptimizationConfig:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                
            # Convert dict back to config objects
            self.config = OptimizationConfig(
                camera=CameraOptimization(**config_dict.get('camera', {})),
                microphone=MicrophoneOptimization(**config_dict.get('microphone', {})),
                processing=ProcessingOptimization(**config_dict.get('processing', {})),
                heart_rate=HeartRateConfig(**config_dict.get('heart_rate', {}))
            )
            
            # Update performance targets
            self.config.target_processing_time = config_dict.get('target_processing_time', 0.1)
            self.config.target_total_latency = config_dict.get('target_total_latency', 0.2)
            self.config.enable_performance_monitoring = config_dict.get('enable_performance_monitoring', True)
            self.config.enable_adaptive_quality = config_dict.get('enable_adaptive_quality', True)
            self.config.enable_debug_output = config_dict.get('enable_debug_output', False)
            
            print(f"✅ Configuration loaded from {self.config_path}")
            
        except FileNotFoundError:
            print(f"⚠️ Config file not found, using defaults. Will create {self.config_path}")
            self.save_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}. Using defaults.")
            self.config = DEFAULT_CONFIG
        
        return self.config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config_dict = asdict(self.config)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration saved to {self.config_path}")
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def update_camera_settings(self, **kwargs):
        """Update camera optimization settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.camera, key):
                setattr(self.config.camera, key, value)
        self.save_config()
    
    def update_microphone_settings(self, **kwargs):
        """Update microphone optimization settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.microphone, key):
                setattr(self.config.microphone, key, value)
        self.save_config()
    
    def update_processing_settings(self, **kwargs):
        """Update processing optimization settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.processing, key):
                setattr(self.config.processing, key, value)
        self.save_config()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance configuration summary"""
        return {
            "target_processing_time": self.config.target_processing_time,
            "target_total_latency": self.config.target_total_latency,
            "camera_resolution": f"{self.config.camera.resolution_width}x{self.config.camera.resolution_height}",
            "camera_resize_factor": self.config.camera.resize_factor,
            "frame_skip_rate": self.config.camera.skip_frames,
            "microphone_sample_rate": self.config.microphone.sample_rate,
            "processing_interval": self.config.processing.min_processing_interval,
            "heart_rate_status": "STANDBY" if not self.config.heart_rate.enabled else "ACTIVE",
            "optimization_level": "HIGH"
        }

# Predefined optimization profiles
OPTIMIZATION_PROFILES = {
    "maximum_speed": OptimizationConfig(
        camera=CameraOptimization(
            resolution_width=320,
            resolution_height=240,
            resize_factor=0.3,
            skip_frames=3,
            detection_confidence=0.8
        ),
        microphone=MicrophoneOptimization(
            sample_rate=8000,
            buffer_size=512
        ),
        processing=ProcessingOptimization(
            min_processing_interval=0.15,
            max_queue_size=5
        ),
        heart_rate=HeartRateConfig(enabled=False),
        target_processing_time=0.05,
        target_total_latency=0.1
    ),
    
    "balanced": DEFAULT_CONFIG,
    
    "maximum_quality": OptimizationConfig(
        camera=CameraOptimization(
            resolution_width=1280,
            resolution_height=720,
            resize_factor=0.8,
            skip_frames=1,
            detection_confidence=0.5
        ),
        microphone=MicrophoneOptimization(
            sample_rate=44100,
            buffer_size=2048
        ),
        processing=ProcessingOptimization(
            min_processing_interval=0.05,
            max_queue_size=20
        ),
        heart_rate=HeartRateConfig(enabled=False),
        target_processing_time=0.2,
        target_total_latency=0.3
    )
}

def create_optimization_config(profile: str = "balanced") -> OptimizationConfig:
    """Create optimization configuration with specified profile"""
    if profile in OPTIMIZATION_PROFILES:
        return OPTIMIZATION_PROFILES[profile]
    else:
        print(f"⚠️ Unknown profile '{profile}', using 'balanced'")
        return OPTIMIZATION_PROFILES["balanced"]

def print_optimization_status():
    """Print current optimization status"""
    print("\n🚀 OPTIMIZATION STATUS - V2.0")
    print("=" * 50)
    print("📸 CAMERA PROCESSING:")
    print("  ✅ Frame skipping enabled")
    print("  ✅ Resolution optimization")
    print("  ✅ Lightweight face detection")
    print("  ✅ Single face processing")
    
    print("\n🎤 MICROPHONE PROCESSING:")
    print("  ✅ Reduced sample rate")
    print("  ✅ Minimal feature extraction")
    print("  ✅ Real-time processing")
    print("  ✅ Latency optimization")
    
    print("\n⚡ PROCESSING OPTIMIZATIONS:")
    print("  ✅ Async processing")
    print("  ✅ Result caching")
    print("  ✅ Queue size limiting")
    print("  ✅ Interval-based processing")
    
    print("\n❄️ STANDBY FEATURES:")
    print("  ⏸️ Heart rate sensors (future)")
    print("  ⏸️ Advanced emotion models (future)")
    print("  ⏸️ Deep learning features (future)")
    
    print("\n🎯 PERFORMANCE TARGETS:")
    print("  ⏱️ Processing time: <100ms")
    print("  🚀 Total latency: <200ms")
    print("  📊 Frame rate: 30fps capable")
    print("=" * 50)

if __name__ == "__main__":
    # Demo configuration management
    print("🔧 Optimization Configuration Demo")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    print("\n📋 Current configuration:")
    print(json.dumps(config_manager.get_performance_summary(), indent=2))
    
    print_optimization_status()
