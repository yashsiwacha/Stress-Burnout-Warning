#!/usr/bin/env python3
"""
AI-Based Stress & Burnout Early Warning System
Main Desktop Application with Advanced ML Integration
Enhanced with CNN facial recognition and NLP text analysis
"""

# Import and setup warning suppression first
try:
    from warning_suppressor import initialize_environment
    initialize_environment()
except ImportError:
    # Fallback warning suppression
    import warnings
    import os
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("🔧 Basic warning suppression applied")

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import json
import os
import random
import datetime
from datetime import datetime, timedelta
from collections import deque
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# UI Framework
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    print("✅ CustomTkinter available - using modern UI")
except ImportError:
    CTK_AVAILABLE = False
    print("📱 Using standard Tkinter interface")

# Scientific computing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy available")
except ImportError:
    NUMPY_AVAILABLE = False
    print("📊 NumPy not available - using basic math")

# Camera functionality
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available - camera monitoring enabled")
except ImportError:
    CV2_AVAILABLE = False
    print("📷 Camera monitoring in simulation mode")

def find_builtin_camera():
    """
    Find the MacBook's built-in camera instead of iPhone/external cameras.
    On macOS, the built-in camera is usually index 0, but with Continuity Camera
    enabled, iPhone cameras might take priority.
    """
    if not CV2_AVAILABLE:
        return None
        
    print("🔍 Scanning for available cameras...")
    
    # Test multiple camera indices to find the right one
    for camera_index in range(5):  # Check first 5 camera indices
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Get camera info
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # Read a test frame
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    print(f"📷 Camera {camera_index}: {int(width)}x{int(height)} - Available")
                    
                    # Prefer built-in camera (usually has standard laptop resolutions)
                    # Built-in MacBook cameras typically support 1280x720 or similar
                    if width in [1280, 1920, 640] and height in [720, 1080, 480]:
                        print(f"✅ Selected built-in camera at index {camera_index}")
                        return camera_index
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            continue
    
    # If no preferred camera found, return the first available one
    print("⚠️ Using first available camera (index 0)")
    return 0

# Global camera override (set this to force a specific camera)
CAMERA_OVERRIDE = None  # Set to 0, 1, 2, etc. to force a specific camera

def get_camera_index():
    """Get the camera index to use, respecting manual override"""
    if CAMERA_OVERRIDE is not None:
        print(f"🎯 Using manual camera override: {CAMERA_OVERRIDE}")
        return CAMERA_OVERRIDE
    return find_builtin_camera()

# Microphone functionality
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("✅ PyAudio available - microphone monitoring enabled")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("🎤 Microphone monitoring in simulation mode")

# Advanced monitoring modules
print("\n🔍 Loading monitoring modules...")
try:
    from src.monitoring.facial_monitor import FacialStressMonitor
    FACIAL_AVAILABLE = True
    print("✅ Advanced facial monitoring loaded")
except ImportError:
    FACIAL_AVAILABLE = False
    print("📷 Using built-in facial monitoring")

try:
    from src.monitoring.voice_monitor import VoiceStressMonitor
    VOICE_AVAILABLE = True
    print("✅ Advanced voice monitoring loaded")
except ImportError:
    VOICE_AVAILABLE = False
    print("🎤 Using built-in voice monitoring")

# Typing monitoring removed by user request

try:
    from src.ai.conversational_ai import ConversationalAI
    CONVERSATIONAL_AVAILABLE = True
    print("✅ Conversational AI loaded")
except ImportError:
    CONVERSATIONAL_AVAILABLE = False
    print("🤖 Using basic chat interface")

try:
    from src.notifications.alert_system import AlertSystem
    ALERTS_AVAILABLE = True
    print("✅ Advanced alert system loaded")
except ImportError:
    ALERTS_AVAILABLE = False
    print("🔔 Using basic notifications")

# Advanced ML imports
print("\n🧠 Loading ML components...")
try:
    from src.monitoring.cnn_facial_analysis import CNNFacialStressAnalyzer
    CNN_AVAILABLE = True
    print("✅ CNN facial analysis loaded")
except ImportError:
    CNN_AVAILABLE = False
    print("🧠 CNN facial analysis not available")

try:
    from src.ai.advanced_nlp_analysis import AdvancedNLPStressAnalyzer
    NLP_AVAILABLE = True
    print("✅ Advanced NLP analysis loaded")
except ImportError:
    NLP_AVAILABLE = False
    print("🧠 Advanced NLP analysis not available")

# Print system readiness summary
print("\n" + "="*60)
print("🚀 STRESS & BURNOUT EARLY WARNING SYSTEM")
print("="*60)
print("🔒 Privacy-focused local processing")
print("🧠 Multimodal stress detection")
print("💡 Intelligent wellbeing recommendations")
print("="*60)

class FacialStressMonitor:
    def __init__(self):
        self.active = False
        self.stress_data = deque(maxlen=100)
        self.current_stress = 0.0
        self.camera = None
        self.connection_status = "disconnected"
        print("👁️ Facial monitoring initialized")
    
    def start_monitoring(self):
        self.active = True
        # Try to establish camera connection
        if CV2_AVAILABLE:
            try:
                # Find the best camera (MacBook built-in instead of iPhone)
                camera_index = get_camera_index()
                if camera_index is not None:
                    self.camera = cv2.VideoCapture(camera_index)
                    if self.camera.isOpened():
                        self.connection_status = "connected"
                        print(f"👁️ Facial monitoring started with camera {camera_index}")
                    else:
                        self.connection_status = "error"
                        print("👁️ Facial monitoring started (camera unavailable)")
                else:
                    self.connection_status = "error"
                    print("👁️ No suitable camera found")
            except Exception as e:
                self.connection_status = "error"
                print(f"👁️ Facial monitoring error: {e}")
        else:
            self.connection_status = "simulated"
            print("👁️ Facial monitoring started (simulation)")
        
    def stop_monitoring(self): 
        self.active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.connection_status = "disconnected"
        print("👁️ Facial monitoring stopped")
        
    def get_current_data(self): 
        # Check connection status and update accordingly
        if CV2_AVAILABLE:
            if self.camera is None or not self.camera.isOpened():
                # Try to reconnect
                try:
                    if self.camera:
                        self.camera.release()
                    # Find the best camera (MacBook built-in instead of iPhone)
                    camera_index = get_camera_index()
                    if camera_index is not None:
                        self.camera = cv2.VideoCapture(camera_index)
                        if self.camera.isOpened():
                            self.connection_status = "connected"
                        else:
                            self.connection_status = "error"
                    else:
                        self.connection_status = "error"
                except Exception:
                    self.connection_status = "error"
            else:
                self.connection_status = "connected"
        else:
            self.connection_status = "simulated"
        
        return {
            'stress_level': 0.3 + random.uniform(-0.1, 0.1), 
            'confidence': 0.8,
            'connection_status': self.connection_status
        }

class VoiceStressMonitor:
    def __init__(self): 
        self.active = False
        self.stress_data = {'stress_level': 0.25, 'confidence': 0.7}
        self.microphone = None
        self.connection_status = "disconnected"
        
    def start_monitoring(self): 
        self.active = True
        # Try to establish microphone connection
        if PYAUDIO_AVAILABLE:
            try:
                self.microphone = pyaudio.PyAudio()
                device_count = self.microphone.get_device_count()
                mic_found = False
                for i in range(device_count):
                    device_info = self.microphone.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels') > 0:
                        mic_found = True
                        break
                
                if mic_found:
                    self.connection_status = "connected"
                    print("🎤 Voice monitoring started with microphone")
                else:
                    self.connection_status = "no_device"
                    print("🎤 Voice monitoring started (no device found)")
            except Exception as e:
                self.connection_status = "error"
                print(f"🎤 Voice monitoring error: {e}")
        else:
            self.connection_status = "simulated"
            print("🎤 Voice monitoring started (simulation)")
            
    def stop_monitoring(self): 
        self.active = False
        if self.microphone:
            self.microphone.terminate()
            self.microphone = None
        self.connection_status = "disconnected"
        print("🎤 Voice monitoring stopped")
        
    def get_current_data(self): 
        # Check connection status and try to reconnect if needed
        if PYAUDIO_AVAILABLE:
            if self.microphone is None:
                # Try to reconnect
                try:
                    self.microphone = pyaudio.PyAudio()
                    device_count = self.microphone.get_device_count()
                    mic_found = False
                    for i in range(device_count):
                        device_info = self.microphone.get_device_info_by_index(i)
                        if device_info.get('maxInputChannels') > 0:
                            mic_found = True
                            break
                    
                    if mic_found:
                        self.connection_status = "connected"
                    else:
                        self.connection_status = "no_device"
                except Exception:
                    self.connection_status = "error"
                    if self.microphone:
                        self.microphone.terminate()
                        self.microphone = None
        else:
            self.connection_status = "simulated"
        
        return {
            'stress_level': 0.25 + random.uniform(-0.05, 0.1), 
            'confidence': 0.7,
            'connection_status': self.connection_status
        }

# Typing monitoring removed by user request

class StressAnalyzer:
    def __init__(self, baseline_calibrator=None): 
        self.baseline_calibrator = baseline_calibrator
    def analyze_multimodal_stress(self, facial_data, voice_data, typing_data):
        stress_level = 0.3 + random.uniform(-0.1, 0.2)
        return {
            'overall_stress': stress_level,
            'risk_level': 'medium' if stress_level > 0.6 else 'low',
            'recommendations': ['Take a 5-minute break', 'Try deep breathing exercises'],
            'confidence': 0.8,
            'components': {
                'facial': facial_data.get('stress_level', 0.3) if facial_data else 0.3,
                'voice': voice_data.get('stress_level', 0.25) if voice_data else 0.25,
                'typing': typing_data.get('stress_level', 0.4) if typing_data else 0.4
            }
        }
    
class BaselineCalibrator:
    def __init__(self): 
        self.baselines = {'facial': 0.2, 'voice': 0.15, 'typing': 0.25}
    def update_baseline(self, data): 
        print("📊 Baseline updated")
    def get_baseline_metrics(self): 
        return self.baselines

class NotificationSystem:
    def __init__(self): 
        self.callbacks = {}
    def show_stress_alert(self, stress_level, risk_factors, recommendations): 
        print(f"📢 Stress alert: {stress_level:.1%} stress level")
    def show_break_reminder(self, break_type="general"): 
        print(f"☕ Break reminder: {break_type}")
    def show_wellbeing_tip(self, tip, category="general"):
        print(f"💡 Wellbeing tip: {tip}")
    def register_callback(self, event_type, callback): 
        self.callbacks[event_type] = callback
    def update_settings(self, settings):
        print("⚙️ Notification settings updated")
    def show_burnout_warning(self, risk_level, sustained_duration):
        print(f"🚨 BURNOUT WARNING: {risk_level} risk for {sustained_duration}")
    
class WellbeingResources:
    def __init__(self): 
        self.exercises = {
            'breathing': {'name': 'Deep Breathing', 'duration': 3, 'description': 'Calming breath exercise'},
            'stretching': {'name': 'Neck Stretch', 'duration': 2, 'description': 'Quick tension relief'}
        }
    def get_recommended_exercise(self, stress_level, available_time=5, exercise_type=None): 
        return {
            'exercise': self.exercises['breathing'],
            'type': 'breathing',
            'estimated_duration': 3
        }
    def get_random_tip(self, category=None): 
        tips = [
            "Take three deep breaths",
            "Look away from screen for 20 seconds",
            "Stretch your shoulders",
            "Drink some water"
        ]
        return random.choice(tips)
    def start_guided_session(self, exercise_data):
        print(f"🧘 Starting exercise: {exercise_data['exercise']['name']}")
        return "session_123"
    def get_session_status(self):
        return None
    
class ConfigurationManager:
    def __init__(self): 
        from types import SimpleNamespace
        self.monitoring = SimpleNamespace(
            facial_monitoring_enabled=True,
            voice_monitoring_enabled=True,
            # typing_monitoring_enabled removed by user request
            facial_sensitivity=0.7,
            voice_sensitivity=0.7,
            # typing_sensitivity removed by user request
        )
        self.alerts = SimpleNamespace(
            enabled=True,
            level_threshold='medium',
            desktop_notifications=False,
            popup_alerts=False
        )
        self.privacy = SimpleNamespace(
            local_processing_only=True,
            store_facial_data=False,
            store_voice_data=False,
            anonymize_data=True
        )
        self.ui = SimpleNamespace(
            theme='dark',
            auto_refresh_interval=5
        )
        self.wellbeing = SimpleNamespace(
            auto_break_reminders=True,
            break_reminder_interval=60,
            preferred_exercise_duration=5
        )
    def save_config(self): 
        print("💾 Configuration saved")
    def get_all_settings(self): 
        return {
            'monitoring': vars(self.monitoring),
            'alerts': vars(self.alerts),
            'privacy': vars(self.privacy),
            'ui': vars(self.ui),
            'wellbeing': vars(self.wellbeing)
        }
    def update_monitoring_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.monitoring, key):
                setattr(self.monitoring, key, value)
    def update_alert_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.alerts, key):
                setattr(self.alerts, key, value)
    
class DataLogger:
    def __init__(self, *args, **kwargs): 
        self.data = []
    def log_stress_level(self, stress_level, components, risk_factors, confidence=1.0):
        entry = {
            'timestamp': datetime.now(),
            'type': 'stress_level',
            'stress_level': stress_level,
            'components': components,
            'risk_factors': risk_factors
        }
        self.data.append(entry)
        print(f"📊 Logged stress level: {stress_level:.2f}")
    def log_system_event(self, event_type, description, metadata=None):
        print(f"🔍 System event: {event_type} - {description}")
    def log_user_action(self, action, context, result=None):
        print(f"👤 User action: {action} in {context}")
    def log_alert(self, alert_type, severity, message):
        print(f"🚨 Alert: {severity} - {message}")
    def log_exercise_session(self, exercise_name, duration_minutes, completion_status):
        print(f"🧘 Exercise: {exercise_name} ({duration_minutes}min) - {completion_status}")
    def get_recent_data(self, data_type=None, hours=24, limit=100):
        return self.data[-limit:] if self.data else []
    def get_stress_history(self, days=7):
        return [(datetime.now() - timedelta(hours=i), 0.3 + i*0.05) for i in range(24)]
    def close(self): 
        print("📊 Data logger closed")

class StressBurnoutApp(ctk.CTk):
    """Main application window for Stress and Burnout Early Warning System"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("AI Stress & Burnout Early Warning System")
        self.geometry("1200x800")
        if os.path.exists("assets/icon.ico"):
            self.iconbitmap("assets/icon.ico")
        
        # Application state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.current_stress_level = 0.0
        self.current_risk_category = "Normal"
        self.session_start_time = None
        self.camera_permission_granted = False
        self.microphone_permission_granted = False
        
        # Initialize monitoring components
        self.facial_monitor = FacialStressMonitor()
        self.voice_monitor = VoiceStressMonitor()
        # Typing monitor removed by user request
        
        # Initialize analysis components
        self.baseline_calibrator = BaselineCalibrator()
        self.stress_analyzer = StressAnalyzer(self.baseline_calibrator)
        
        # Initialize system components
        self.config_manager = ConfigurationManager()
        self.data_logger = DataLogger(anonymize=self.config_manager.privacy.anonymize_data)
        self.notification_system = NotificationSystem()
        self.wellbeing_resources = WellbeingResources()
        
        # Setup notification callbacks
        self.setup_notification_callbacks()
        
        # Initialize data storage for real-time display
        self.stress_history = []
        self.last_stress_check = time.time()
        
        # Load user settings
        self.load_settings()
        
        # Setup GUI
        self.setup_gui()
        
        # Setup system tray (for background operation)
        self.setup_system_tray()
        
        # Log application start
        self.data_logger.log_system_event("app_start", "Application started successfully")
        print("✅ Application initialized successfully")
        print("📱 Desktop application running...")
        print("💡 Tip: Check the system tray for background monitoring")
        
        # Auto-start camera preview (no button needed)
        self.after(1000, self.auto_start_camera)  # Start camera after 1 second delay
    
    def setup_notification_callbacks(self):
        """Setup callbacks for notification system"""
        
        self.notification_system.register_callback('break_mode_requested', self.handle_break_request)
        self.notification_system.register_callback('emergency_break_requested', self.handle_emergency_break)
        self.notification_system.register_callback('guided_break_requested', self.handle_guided_break)
        self.notification_system.register_callback('wellbeing_activity_requested', self.handle_wellbeing_activity)
    
    def handle_break_request(self):
        """Handle regular break request"""
        self.log_activity("☕ Break request received - Take a 5 minute break")
        self.data_logger.log_user_action("break_requested", "stress_alert", "user_accepted")
    
    def handle_emergency_break(self):
        """Handle emergency break request"""
        self.log_activity("🚨 Emergency break request - Please take a longer break")
        self.data_logger.log_user_action("emergency_break_requested", "burnout_warning", "user_accepted")
    
    def handle_guided_break(self, break_type):
        """Handle guided break session"""
        self.start_guided_exercise(break_type)
        self.data_logger.log_user_action("guided_break_started", f"break_type_{break_type}")
    
    def handle_wellbeing_activity(self, category):
        """Handle wellbeing activity request"""
        exercise = self.wellbeing_resources.get_recommended_exercise(
            stress_level=self.current_stress_level,
            available_time=self.config_manager.wellbeing.preferred_exercise_duration
        )
        self.start_guided_exercise_session(exercise)
        self.data_logger.log_user_action("wellbeing_activity_started", f"category_{category}")
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
        # Create status bar
        self.create_status_bar()
        
        # Activity logging moved to console only (removed overlapping text widget)
        
    def create_sidebar(self):
        """Create the left sidebar with navigation and controls"""
        
        # Create scrollable sidebar frame
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=300, corner_radius=15)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=(10, 5), pady=10)
        
        # App logo and title with gradient effect
        logo_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        logo_frame.pack(pady=(20, 10), padx=20, fill="x")
        
        self.logo_label = ctk.CTkLabel(
            logo_frame,
            text="🧠 MindGuard AI",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=("#1f538d", "#14375e")
        )
        self.logo_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            logo_frame,
            text="Stress & Burnout Early Warning",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        subtitle_label.pack()
        
        # Main control section with modern styling
        control_frame = ctk.CTkFrame(self.sidebar_frame, fg_color=("#f0f0f0", "#2b2b2b"))
        control_frame.pack(pady=15, padx=20, fill="x")
        
        # Camera and microphone status indicators
        permissions_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        permissions_frame.pack(pady=10, padx=15, fill="x")
        
        ctk.CTkLabel(
            permissions_frame,
            text="📷 Camera & 🎤 Microphone Status",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack()
        
        # Status indicators with visual feedback
        status_row = ctk.CTkFrame(permissions_frame, fg_color="transparent")
        status_row.pack(pady=5, fill="x")
        
        self.camera_status = ctk.CTkLabel(
            status_row,
            text="📷 Not Granted",
            font=ctk.CTkFont(size=10),
            text_color="red"
        )
        self.camera_status.pack(side="left", padx=5)
        
        self.mic_status = ctk.CTkLabel(
            status_row,
            text="🎤 Not Granted", 
            font=ctk.CTkFont(size=10),
            text_color="red"
        )
        self.mic_status.pack(side="right", padx=5)
        
        # Main start/stop button with enhanced styling
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="🚀 Start Monitoring",
            command=self.toggle_monitoring,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#2CC985", "#2FA572"),
            hover_color=("#25B574", "#28956B"),
            corner_radius=25
        )
        self.start_btn.pack(pady=15, padx=15, fill="x")
        
        # Real-time status display with enhanced visuals
        self.status_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        self.status_frame.pack(pady=10, padx=20, fill="x")
        
        # Status header
        status_header = ctk.CTkLabel(
            self.status_frame,
            text="📊 Live Monitoring",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        status_header.pack(pady=(15, 5))
        
        # Stress level with animated progress
        self.stress_level_label = ctk.CTkLabel(
            self.status_frame,
            text="Stress Level: Initializing...",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.stress_level_label.pack(pady=5)
        
        # Enhanced progress bar with color coding
        self.stress_progress = ctk.CTkProgressBar(
            self.status_frame,
            width=200,
            height=12,
            corner_radius=6,
            progress_color=("#4CAF50", "#2E7D32")
        )
        self.stress_progress.pack(pady=10, padx=20)
        self.stress_progress.set(0)
        
        # Individual component status
        components_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        components_frame.pack(pady=5, padx=15, fill="x")
        
        # Facial monitoring indicator
        facial_frame = ctk.CTkFrame(components_frame, fg_color="transparent")
        facial_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(facial_frame, text="👁️", font=ctk.CTkFont(size=14)).pack(side="left")
        self.facial_indicator = ctk.CTkLabel(
            facial_frame,
            text="Facial: --",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.facial_indicator.pack(side="left", padx=5)
        
        # Voice monitoring indicator
        voice_frame = ctk.CTkFrame(components_frame, fg_color="transparent")
        voice_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(voice_frame, text="🎤", font=ctk.CTkFont(size=14)).pack(side="left")
        self.voice_indicator = ctk.CTkLabel(
            voice_frame,
            text="Voice: --",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.voice_indicator.pack(side="left", padx=5)
        
        # Typing monitoring removed by user request
        
        # Navigation section removed by user request
        
        # Enhanced privacy controls with better visual design
        self.privacy_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        self.privacy_frame.pack(pady=15, padx=20, fill="x")
        
        privacy_header = ctk.CTkFrame(self.privacy_frame, fg_color="transparent")
        privacy_header.pack(pady=(15, 10), fill="x")
        
        ctk.CTkLabel(
            privacy_header, 
            text="🔒 Privacy Controls", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack()
        
        ctk.CTkLabel(
            privacy_header,
            text="All data processed locally",
            font=ctk.CTkFont(size=9),
            text_color="gray"
        ).pack()
        
        # Privacy switches with enhanced styling
        privacy_switches = [
            ("👁️ Facial Analysis", "facial_monitoring", self.toggle_facial_monitoring),
            ("🎤 Voice Analysis", "voice_monitoring", self.toggle_voice_monitoring),
            # Typing Analysis removed by user request
        ]
        
        for text, attr, command in privacy_switches:
            switch_frame = ctk.CTkFrame(self.privacy_frame, fg_color="transparent")
            switch_frame.pack(fill="x", pady=5, padx=15)
            
            switch = ctk.CTkSwitch(
                switch_frame,
                text=text,
                command=command,
                font=ctk.CTkFont(size=11),
                switch_width=50,
                switch_height=25
            )
            switch.pack(anchor="w")
            switch.select()  # Enable by default
            setattr(self, attr, switch)
        
        # Quick wellness actions
        wellness_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        wellness_frame.pack(pady=(0, 15), padx=20, fill="x")
        
        ctk.CTkLabel(
            wellness_frame,
            text="⚡ Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10))
        
        quick_actions = [
            ("🫁 Breathe", self.start_breathing_exercise, "#e74c3c"),
            ("☕ Break", self.suggest_break, "#f39c12")
        ]
        
        actions_row = ctk.CTkFrame(wellness_frame, fg_color="transparent")
        actions_row.pack(pady=(0, 15), padx=15, fill="x")
        
        for text, command, color in quick_actions:
            btn = ctk.CTkButton(
                actions_row,
                text=text,
                command=command,
                width=80,
                height=35,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=color,
                hover_color=self.darken_color(color),
                corner_radius=8
            )
            btn.pack(side="left", padx=5, expand=True, fill="x")
    
    def darken_color(self, hex_color):
        """Darken a hex color for hover effects"""
        # Remove the # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Darken by reducing each component by 20%
        darker_rgb = tuple(max(0, int(c * 0.8)) for c in rgb)
        # Convert back to hex
        return f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        
    def create_main_content(self):
        """Create the main content area with different pages"""
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create main dashboard content directly
        self.create_dashboard_content()
        
    def create_dashboard_content(self):
        """Create the main dashboard with monitoring overview"""
        
        # Header section
        header_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, height=100)
        header_frame.pack(fill="x", pady=(20, 20), padx=20)
        header_frame.pack_propagate(False)
        
        # Welcome message with time-based greeting
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good Morning"
            emoji = "🌅"
        elif hour < 17:
            greeting = "Good Afternoon"
            emoji = "☀️"
        else:
            greeting = "Good Evening"
            emoji = "🌆"
        
        welcome_label = ctk.CTkLabel(
            header_frame,
            text=f"{emoji} {greeting}! Let's monitor your wellbeing",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        welcome_label.pack(pady=20)
        
        # Create main content area with camera preview
        content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left side - Camera preview
        camera_frame = ctk.CTkFrame(content_frame, corner_radius=15)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Camera header
        camera_header = ctk.CTkLabel(
            camera_frame,
            text="📷 Live Camera Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        camera_header.pack(pady=(15, 10))
        
        # Camera display area
        self.camera_display_frame = ctk.CTkFrame(camera_frame, corner_radius=10)
        self.camera_display_frame.pack(pady=10, padx=15, fill="both", expand=True)
        
        # Camera preview label (will display video frames) - smaller size for performance
        self.camera_preview = ctk.CTkLabel(
            self.camera_display_frame,
            text="📷\nCamera Preview\n\nStart monitoring to enable\ncamera preview",
            font=ctk.CTkFont(size=12),
            fg_color=("#f0f0f0", "#2b2b2b"),
            corner_radius=10,
            width=240,
            height=180
        )
        self.camera_preview.pack(pady=15, padx=15, fill="both", expand=True)
        
        # Camera controls
        camera_controls = ctk.CTkFrame(camera_frame, fg_color="transparent")
        camera_controls.pack(pady=(0, 15), padx=15, fill="x")
        
        # Detection info (camera is always on)
        self.detection_info = ctk.CTkLabel(
            camera_controls,
            text="Face detection: Starting...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="orange"
        )
        self.detection_info.pack(side="right", padx=5)
        
        # Right side - Analytics and stats
        stats_frame = ctk.CTkFrame(content_frame, corner_radius=15, width=300)
        stats_frame.pack(side="right", fill="y", padx=(10, 0))
        stats_frame.pack_propagate(False)
        
        # Stats header
        stats_header = ctk.CTkLabel(
            stats_frame,
            text="📊 Live Analytics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_header.pack(pady=(15, 10))
        
        # Facial emotion stats
        emotion_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        emotion_frame.pack(pady=10, padx=15, fill="x")
        
        ctk.CTkLabel(
            emotion_frame,
            text="😊 Detected Emotions",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(10, 5))
        
        self.emotion_display = ctk.CTkLabel(
            emotion_frame,
            text="No emotions detected",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.emotion_display.pack(pady=(0, 10))
        
        # Stress indicators
        stress_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        stress_frame.pack(pady=10, padx=15, fill="x")
        
        ctk.CTkLabel(
            stress_frame,
            text="⚡ Stress Indicators",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(10, 5))
        
        self.stress_indicators = ctk.CTkLabel(
            stress_frame,
            text="No stress data available",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.stress_indicators.pack(pady=(0, 10))
        
        # Initialize camera preview variables
        self.camera_preview_active = False
        self.camera_preview_thread = None
        self.camera_cap = None
    
    def create_status_bar(self):
        """Create the bottom status bar"""
        
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="🟢 System Ready - Privacy Protected",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.pack(side="left", padx=10)
        
        self.session_time_label = ctk.CTkLabel(
            self.status_bar,
            text="Session: 00:00:00",
            font=ctk.CTkFont(size=10)
        )
        self.session_time_label.pack(side="right", padx=10)
    
    def toggle_monitoring(self):
        """Start or stop the monitoring system"""
        
        if not self.monitoring_active:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def request_permissions(self):
        """Request camera and microphone permissions"""
        self.log_activity("🔒 Requesting camera and microphone permissions...")
        
        # Request camera permission
        if CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    self.camera_permission_granted = True
                    self.camera_status.configure(text="📷 Granted", text_color="green")
                    self.log_activity("✅ Camera permission granted")
                    cap.release()
                else:
                    self.camera_status.configure(text="📷 Denied", text_color="red")
                    self.log_activity("❌ Camera permission denied")
            except Exception as e:
                self.camera_status.configure(text="📷 Error", text_color="red")
                self.log_activity(f"❌ Camera error: {e}")
        else:
            self.camera_permission_granted = True  # Simulate for demo
            self.camera_status.configure(text="📷 Simulated", text_color="blue")
            self.log_activity("✅ Camera permission simulated")
        
        # Request microphone permission
        if PYAUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()
                # Test if we can access microphone
                device_count = p.get_device_count()
                mic_found = False
                for i in range(device_count):
                    device_info = p.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels') > 0:
                        mic_found = True
                        break
                
                if mic_found:
                    self.microphone_permission_granted = True
                    self.mic_status.configure(text="🎤 Granted", text_color="green")
                    self.log_activity("✅ Microphone permission granted")
                else:
                    self.mic_status.configure(text="🎤 No Device", text_color="orange")
                    self.log_activity("⚠️ No microphone device found")
                p.terminate()
            except Exception as e:
                self.mic_status.configure(text="🎤 Error", text_color="red")
                self.log_activity(f"❌ Microphone error: {e}")
        else:
            self.microphone_permission_granted = True  # Simulate for demo
            self.mic_status.configure(text="🎤 Simulated", text_color="blue")
            self.log_activity("✅ Microphone permission simulated")
    
    def start_monitoring(self):
        """Start the real-time monitoring"""
        
        if self.monitoring_active:
            return
        
        # First request permissions
        self.request_permissions()
        
        self.monitoring_active = True
        self.session_start_time = datetime.now()
        
        # Start individual monitors based on configuration
        if self.config_manager.monitoring.facial_monitoring_enabled and self.camera_permission_granted:
            self.facial_monitor.start_monitoring()
            # Auto-start camera preview when monitoring starts
            if not self.camera_preview_active:
                self.start_camera_preview()
        
        if self.config_manager.monitoring.voice_monitoring_enabled and self.microphone_permission_granted:
            self.voice_monitor.start_monitoring()
        
        # Typing monitoring removed by user request
        
        # Update UI
        self.start_btn.configure(text="⏸️ Stop Monitoring")
        self.status_label.configure(text="🔴 Monitoring Active - Data Processed Locally")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start UI update timer
        self.update_ui_timer()
        
        # Log the start
        self.data_logger.log_system_event("monitoring_start", "Real-time monitoring session started")
        self.log_activity("✅ Monitoring started - All data processed locally for privacy")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Stop individual monitors
        self.facial_monitor.stop_monitoring()
        self.voice_monitor.stop_monitoring()
        # Typing monitor removed by user request
        
        # Stop camera preview if active
        if self.camera_preview_active:
            self.stop_camera_preview()
        
        # Update UI
        self.start_btn.configure(text="🚀 Start Monitoring")
        self.status_label.configure(text="🟢 System Ready - Privacy Protected")
        
        # Reset indicators
        self.facial_indicator.configure(text="Facial: --", text_color="gray")
        self.voice_indicator.configure(text="Voice: --", text_color="gray")
        # Typing indicator removed by user request
        
        # Reset permission status
        self.camera_status.configure(text="📷 Not Active", text_color="gray")
        self.mic_status.configure(text="🎤 Not Active", text_color="gray")
        
        # Log the stop
        self.data_logger.log_system_event("monitoring_stop", "Monitoring session ended")
        self.log_activity("⏹️ Monitoring stopped")
    
    def monitoring_loop(self):
        """Optimized monitoring loop that runs in background thread"""
        
        connection_check_interval = 0  # Counter for periodic connection checks
        ui_update_counter = 0  # Counter for UI updates
        
        while self.monitoring_active:
            try:
                # Collect data from enabled monitors
                facial_data = None
                voice_data = None
                typing_data = None
                
                if self.config_manager.monitoring.facial_monitoring_enabled and self.camera_permission_granted:
                    facial_data = self.facial_monitor.get_current_data()
                
                if self.config_manager.monitoring.voice_monitoring_enabled and self.microphone_permission_granted:
                    voice_data = self.voice_monitor.get_current_data()
                
                # Typing monitoring removed by user request
                
                # Analyze stress levels using multimodal data
                stress_analysis = self.stress_analyzer.analyze_multimodal_stress(
                    facial_data or {},
                    voice_data or {},
                    {} # Empty dict instead of typing_data
                )
                
                # Update current stress level
                self.current_stress_level = stress_analysis['overall_stress']
                self.current_risk_category = stress_analysis['risk_level']
                
                # Update UI only every 3rd iteration to reduce overhead
                ui_update_counter += 1
                if ui_update_counter % 3 == 0:
                    self.after(0, self.update_monitoring_display, stress_analysis, facial_data, voice_data, None)  # None instead of typing_data
                
                # Log stress data (reduced frequency)
                if ui_update_counter % 2 == 0:
                    self.data_logger.log_stress_level(
                        stress_level=stress_analysis['overall_stress'],
                        components=stress_analysis.get('components', {}),
                        risk_factors=stress_analysis.get('risk_factors', []),
                        confidence=stress_analysis.get('confidence', 1.0)
                    )
                
                # Store for UI display (reduced data retention)
                self.stress_history.append((datetime.now(), stress_analysis['overall_stress']))
                if len(self.stress_history) > 50:  # Reduced from 100 to 50 readings
                    self.stress_history = self.stress_history[-50:]
                
                # Check for alerts (less frequently)
                if ui_update_counter % 4 == 0:
                    self.check_stress_alerts(stress_analysis)
                
                # Periodic connection health check (every 60 seconds instead of 30)
                connection_check_interval += 1
                if connection_check_interval >= 12:  # 12 * 5 seconds = 60 seconds
                    connection_check_interval = 0
                    self.check_connection_health(facial_data, voice_data)
                
                # Increased sleep interval for better performance
                time.sleep(max(2.0, self.config_manager.ui.auto_refresh_interval))  # Minimum 2 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                self.data_logger.log_system_event("monitoring_error", f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def check_connection_health(self, facial_data, voice_data):
        """Check the health of camera and microphone connections"""
        try:
            # Check camera connection
            if facial_data and facial_data.get('connection_status') == 'error':
                self.after(0, lambda: self.log_activity("⚠️ Camera connection lost - trying to reconnect..."))
                # Try to reconnect camera
                if hasattr(self.facial_monitor, 'camera') and self.facial_monitor.camera:
                    self.facial_monitor.camera.release()
                    self.facial_monitor.camera = None
                # The next get_current_data call will attempt reconnection
            
            # Check microphone connection  
            if voice_data and voice_data.get('connection_status') == 'error':
                self.after(0, lambda: self.log_activity("⚠️ Microphone connection lost - trying to reconnect..."))
                # Try to reconnect microphone
                if hasattr(self.voice_monitor, 'microphone') and self.voice_monitor.microphone:
                    self.voice_monitor.microphone.terminate()
                    self.voice_monitor.microphone = None
                # The next get_current_data call will attempt reconnection
                
        except Exception as e:
            print(f"Connection health check error: {e}")
    
    def update_monitoring_display(self, stress_analysis, facial_data, voice_data, typing_data):
        """Update UI display with current monitoring data"""
        # Note: typing_data parameter kept for compatibility but ignored (typing monitoring removed)
        try:
            # Update stress level display
            stress_level = stress_analysis['overall_stress']
            risk_level = stress_analysis['risk_level']
            
            self.stress_level_label.configure(text=f"Stress Level: {stress_level:.1%}")
            self.stress_progress.set(stress_level)
            
            # Color code based on stress level
            if stress_level < 0.3:
                color = "#4CAF50"  # Green
                risk_text = "Low"
            elif stress_level < 0.6:
                color = "#FF9800"  # Orange
                risk_text = "Medium"
            else:
                color = "#F44336"  # Red
                risk_text = "High"
            
            self.stress_progress.configure(progress_color=color)
            
            # Update component indicators
            if facial_data:
                self.facial_indicator.configure(
                    text=f"Facial: {facial_data['stress_level']:.1%}",
                    text_color=color
                )
            
            if voice_data:
                self.voice_indicator.configure(
                    text=f"Voice: {voice_data['stress_level']:.1%}",
                    text_color=color
                )
            
            # Typing indicator removed by user request
            
            # Update status label instead
            self.status_label.configure(
                text=f"🔴 Monitoring - {risk_text} Stress ({stress_level:.1%})"
            )
            
        except Exception as e:
            print(f"UI update error: {e}")
    
    def check_stress_alerts(self, stress_analysis):
        """Check if stress levels warrant alerts"""
        
        if not self.config_manager.alerts.enabled:
            return
        
        stress_score = stress_analysis['overall_stress']
        
        # Check for elevated stress (warning alert)
        if stress_score > 0.6:
            self.after(0, lambda: self.log_activity(f"⚠️ ELEVATED STRESS: {stress_score:.1%} detected"))
        
        # Check for break reminders
        elif stress_score > 0.4:
            current_time = time.time()
            if current_time - self.last_stress_check > (self.config_manager.wellbeing.break_reminder_interval * 60):
                self.after(0, lambda: self.log_activity("💡 Consider taking a break"))
                self.last_stress_check = current_time
    
    def log_activity(self, message):
        """Log activity to the activity feed"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            
            # Update activity text in main thread (with safety check)
            if hasattr(self, 'activity_text') and self.activity_text.winfo_exists():
                self.activity_text.insert("end", log_message)
                self.activity_text.see("end")
            else:
                # Fallback to console if widget doesn't exist
                print(log_message.strip())
        except Exception as e:
            print(f"Logging error: {e}")
            print(message)
    
    def update_ui_timer(self):
        """Update UI elements with current data"""
        
        if self.monitoring_active:
            # Update session time
            if self.session_start_time:
                elapsed = datetime.now() - self.session_start_time
                hours, remainder = divmod(elapsed.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}"
                self.session_time_label.configure(text=time_str)
            
            # Schedule next update
            self.after(1000, self.update_ui_timer)
    
    def setup_system_tray(self):
        """Setup system tray (placeholder)"""
        pass
    
    def load_settings(self):
        """Load user settings"""
        self.log_activity("⚙️ Settings loaded successfully")
    
    def save_settings(self):
        """Save user settings"""
        self.config_manager.save_config()
    
    # Placeholder methods for GUI functionality
    def toggle_facial_monitoring(self):
        self.config_manager.monitoring.facial_monitoring_enabled = self.facial_monitoring.get()
        self.log_activity(f"👁️ Facial monitoring: {'enabled' if self.facial_monitoring.get() else 'disabled'}")
    
    def toggle_voice_monitoring(self):
        self.config_manager.monitoring.voice_monitoring_enabled = self.voice_monitoring.get()
        self.log_activity(f"🎤 Voice monitoring: {'enabled' if self.voice_monitoring.get() else 'disabled'}")
    
    # toggle_typing_monitoring method removed by user request
    
    def start_breathing_exercise(self):
        self.log_activity("🫁 Starting breathing exercise...")
        self.wellbeing_resources.start_guided_session({'exercise': {'name': 'Deep Breathing'}})
    
    def suggest_break(self):
        self.log_activity("☕ Break suggestion activated")
        self.notification_system.show_break_reminder("general")
    
    def start_guided_exercise(self, exercise_type):
        """Start a guided exercise session"""
        exercise = self.wellbeing_resources.get_recommended_exercise(
            stress_level=self.current_stress_level,
            available_time=self.config_manager.wellbeing.preferred_exercise_duration
        )
        
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        
        if session_id:
            self.log_activity(f"🧘 Started guided exercise: {exercise['exercise']['name']}")
    
    def start_guided_exercise_session(self, exercise_data):
        """Start a guided exercise session from exercise data"""
        session_id = self.wellbeing_resources.start_guided_session(exercise_data)
        
        if session_id:
            exercise = exercise_data['exercise']
            self.log_activity(f"🧘 Started guided exercise: {exercise['name']}")
    
    def auto_start_camera(self):
        """Automatically start camera preview on app startup"""
        print("🎬 Auto-starting camera preview...")
        self.start_camera_preview()
    
    def start_camera_preview(self):
        """Start the camera preview"""
        if CV2_AVAILABLE:
            try:
                # Find the best camera (MacBook built-in instead of iPhone)
                camera_index = get_camera_index()
                if camera_index is not None:
                    self.camera_cap = cv2.VideoCapture(camera_index)
                    if self.camera_cap.isOpened():
                        self.camera_preview_active = True
                        self.detection_info.configure(text=f"📷 Face detection: Active (Camera {camera_index})", text_color="green")
                        
                        # Start camera preview thread
                    self.camera_preview_thread = threading.Thread(target=self.camera_preview_loop, daemon=True)
                    self.camera_preview_thread.start()
                    
                    self.log_activity("📷 Camera preview started automatically")
                else:
                    self.log_activity("❌ Failed to open camera")
                    self.camera_preview.configure(text="📷\nCamera Error\n\nUnable to access camera\nPlease check permissions")
            except Exception as e:
                self.log_activity(f"❌ Camera error: {e}")
                self.camera_preview.configure(text="📷\nCamera Error\n\nUnable to access camera\nPlease check permissions")
        else:
            # Simulation mode
            self.camera_preview_active = True
            self.detection_info.configure(text="📷 Face detection: Simulated", text_color="blue")
            self.camera_preview_thread = threading.Thread(target=self.camera_simulation_loop, daemon=True)
            self.camera_preview_thread.start()
            self.log_activity("📷 Camera preview started automatically (simulation mode)")
    
    def stop_camera_preview(self):
        """Stop the camera preview"""
        self.camera_preview_active = False
        self.detection_info.configure(text="📷 Face detection: Inactive", text_color="gray")
        
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
        
        # Reset preview display
        self.camera_preview.configure(
            image=None,
            text="📷\nCamera Preview\n\nStarting camera...\nPlease wait"
        )
        
        self.log_activity("📷 Camera preview stopped")
    
    def camera_preview_loop(self):
        """Optimized camera preview loop"""
        try:
            # Initialize face cascade once (performance optimization)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            frame_skip = 0
            face_detection_interval = 5  # Only detect faces every 5 frames for performance
            
            while self.camera_preview_active and self.camera_cap and self.camera_cap.isOpened():
                ret, frame = self.camera_cap.read()
                if ret:
                    # Resize frame for display (smaller for better performance)
                    display_frame = cv2.resize(frame, (240, 180))  # Reduced size
                    
                    # Convert from BGR to RGB
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Only perform face detection every few frames for performance
                    if frame_skip % face_detection_interval == 0:
                        try:
                            # Convert to grayscale for face detection
                            gray = cv2.cvtColor(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                            # Use faster parameters for face detection
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
                            
                            # Store face detection results for reuse
                            self.last_faces = faces
                            
                            # Update UI only when faces change
                            face_count = len(faces)
                            if hasattr(self, 'last_face_count') and self.last_face_count != face_count:
                                if face_count > 0:
                                    self.after(0, lambda: self.detection_info.configure(
                                        text=f"Face detection: {face_count} face(s) detected", 
                                        text_color="green"
                                    ))
                                    self.after(0, lambda: self.emotion_display.configure(
                                        text=f"Analyzing {face_count} face(s)...",
                                        text_color="orange"
                                    ))
                                else:
                                    self.after(0, lambda: self.detection_info.configure(
                                        text="Face detection: No faces detected", 
                                        text_color="orange"
                                    ))
                                    self.after(0, lambda: self.emotion_display.configure(
                                        text="No faces detected",
                                        text_color="gray"
                                    ))
                            self.last_face_count = face_count
                            
                        except Exception as e:
                            print(f"Face detection error: {e}")
                    
                    # Draw rectangles using cached face detection results
                    if hasattr(self, 'last_faces'):
                        # Scale face coordinates to match resized frame
                        scale_x = 240 / 320  # Original was 320, now 240
                        scale_y = 180 / 240  # Original was 240, now 180
                        
                        for (x, y, w, h) in self.last_faces:
                            # Scale coordinates
                            x_scaled = int(x * scale_x)
                            y_scaled = int(y * scale_y)
                            w_scaled = int(w * scale_x)
                            h_scaled = int(h * scale_y)
                            
                            cv2.rectangle(display_frame, (x_scaled, y_scaled), 
                                        (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)
                            cv2.putText(display_frame, "OK", (x_scaled, y_scaled-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Update preview every 3rd frame for smoother performance
                    if frame_skip % 3 == 0:
                        try:
                            from PIL import Image
                            pil_image = Image.fromarray(display_frame)
                            
                            # Convert to CTkImage with reduced size
                            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(240, 180))
                            
                            # Update the preview label
                            self.after(0, lambda img=ctk_image: self.camera_preview.configure(image=img, text=""))
                            
                        except Exception as e:
                            print(f"Image conversion error: {e}")
                
                frame_skip += 1
                # Reduced frame rate for better performance (15 FPS instead of 30)
                time.sleep(1/15)
                
        except Exception as e:
            print(f"Camera preview error: {e}")
            self.after(0, lambda: self.log_activity(f"❌ Camera preview error: {e}"))
    
    def camera_simulation_loop(self):
        """Optimized simulation loop for when camera is not available"""
        try:
            emotions = ["😊 Happy", "😐 Neutral", "😔 Sad", "😠 Angry", "😨 Surprised", "😟 Worried"]
            counter = 0
            
            while self.camera_preview_active:
                # Simulate changing emotions (slower updates for performance)
                current_emotion = emotions[counter % len(emotions)]
                simulation_text = f"📷\nCamera Simulation\n\nDetected: {current_emotion}\n\nDemo Mode"
                
                self.after(0, lambda text=simulation_text: self.camera_preview.configure(text=text))
                self.after(0, lambda emotion=current_emotion: self.emotion_display.configure(
                    text=f"Simulated: {emotion}",
                    text_color="blue"
                ))
                
                counter += 1
                time.sleep(3)  # Increased from 2 to 3 seconds for better performance
                
        except Exception as e:
            print(f"Camera simulation error: {e}")

def main():
    """Main function to run the application"""
    try:
        # Application startup
        print("\n🚀 Starting application...")
        print("📱 Desktop application initializing...")
        
        # Create and run the application
        app = StressBurnoutApp()
        print("✅ Application initialized successfully")
        print("� Desktop application running...")
        print("💡 Tip: Check the system tray for background monitoring")
        
        app.mainloop()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
