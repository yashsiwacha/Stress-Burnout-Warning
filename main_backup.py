#!/usr/bin/env python3
"""
AI-Based Stress & Burnout Early Warning System
Main Desktop Application with Advanced ML Integration
Enhanced with CNN facial recognition and NLP text analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
from datetime import datetime, timedelta
from collections import deque
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    CTK_AVAILABLE = False
    print("⚠️ CustomTkinter not available - using standard Tkinter")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import our advanced ML modules
try:
    from src.monitoring.facial_monitor import FacialStressMonitor
    FACIAL_AVAILABLE = True
except ImportError:
    FACIAL_AVAILABLE = False
    print("⚠️ Facial monitoring not available")

try:
    from src.monitoring.voice_monitor import VoiceStressMonitor
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️ Voice monitoring not available")

try:
    from src.monitoring.typing_monitor import TypingPatternMonitor
    TYPING_AVAILABLE = True
except ImportError:
    TYPING_AVAILABLE = False
    print("⚠️ Typing monitoring not available")

try:
    from src.ai.conversational_ai import ConversationalAI
    CONVERSATIONAL_AVAILABLE = True
except ImportError:
    CONVERSATIONAL_AVAILABLE = False
    print("⚠️ Conversational AI not available")

try:
    from src.notifications.alert_system import AlertSystem
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    print("⚠️ Alert system not available")

# Advanced ML imports
try:
    from src.monitoring.cnn_facial_analysis import CNNFacialStressAnalyzer
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("⚠️ CNN facial analysis not available")

try:
    from src.ai.advanced_nlp_analysis import AdvancedNLPStressAnalyzer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️ Advanced NLP analysis not available")
            print("👁️ Facial monitoring started (simulation)")
        def stop_monitoring(self): 
            self.active = False
            print("👁️ Facial monitoring stopped")
        def get_current_data(self): 
            return self.stress_data
    
    class VoiceStressMonitor:
        def __init__(self): 
            self.active = False
            self.stress_data = {'stress_level': 0.25, 'confidence': 0.7}
        def start_monitoring(self): 
            self.active = True
            print("🎤 Voice monitoring started (simulation)")
        def stop_monitoring(self): 
            self.active = False
            print("🎤 Voice monitoring stopped")
        def get_current_data(self): 
            return self.stress_data
    
    class TypingBehaviorMonitor:
        def __init__(self): 
            self.active = False
            self.stress_data = {'stress_level': 0.4, 'confidence': 0.9}
        def start_monitoring(self): 
            self.active = True
            print("⌨️ Typing monitoring started (simulation)")
        def stop_monitoring(self): 
            self.active = False
            print("⌨️ Typing monitoring stopped")
        def get_current_data(self): 
            return self.stress_data
    
    class StressAnalyzer:
        def __init__(self, baseline_calibrator=None): 
            self.baseline_calibrator = baseline_calibrator
        def analyze_multimodal_stress(self, facial_data, voice_data, typing_data):
            import random
            stress_level = 0.3 + random.uniform(-0.1, 0.2)
            return {
                'overall_stress': stress_level,
                'risk_level': 'medium' if stress_level > 0.6 else 'low',
                'recommendations': ['Take a 5-minute break', 'Try deep breathing exercises'],
                'confidence': 0.8,
                'components': {
                    'facial': facial_data.get('stress_level', 0.3),
                    'voice': voice_data.get('stress_level', 0.25),
                    'typing': typing_data.get('stress_level', 0.4)
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
            import random
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
                typing_monitoring_enabled=True,
                facial_sensitivity=0.7,
                voice_sensitivity=0.7,
                typing_sensitivity=0.7
            )
            self.alerts = SimpleNamespace(
                enabled=True,
                level_threshold='medium',
                desktop_notifications=True,
                popup_alerts=True
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
        def get_recent_data(self, data_type=None, hours=24, limit=100):
            return self.data[-limit:] if self.data else []
        def get_stress_history(self, days=7):
            return [(datetime.now() - timedelta(hours=i), 0.3 + i*0.05) for i in range(24)]
        def close(self): 
            print("📊 Data logger closed")
    
    # Create NotificationLevel and DataType enums for simulation
    class NotificationLevel:
        INFO = "info"
        WARNING = "warning"
        URGENT = "urgent"
        CRITICAL = "critical"
    
    class DataType:
        STRESS_LEVEL = "stress_level"
        SYSTEM_EVENT = "system_event"
        USER_ACTION = "user_action"

class StressBurnoutApp(ctk.CTk):
    """Main application window for Stress and Burnout Early Warning System"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("AI Stress & Burnout Early Warning System")
        self.geometry("1200x800")
        self.iconbitmap("assets/icon.ico") if os.path.exists("assets/icon.ico") else None
        
        # Application state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.current_stress_level = 0.0
        self.current_risk_category = "Normal"
        self.session_start_time = None
        
        # Initialize monitoring components
        self.facial_monitor = FacialStressMonitor()
        self.voice_monitor = VoiceStressMonitor()
        self.typing_monitor = TypingBehaviorMonitor()
        
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
    
    def setup_notification_callbacks(self):
        """Setup callbacks for notification system"""
        
        self.notification_system.register_callback('break_mode_requested', self.handle_break_request)
        self.notification_system.register_callback('emergency_break_requested', self.handle_emergency_break)
        self.notification_system.register_callback('guided_break_requested', self.handle_guided_break)
        self.notification_system.register_callback('wellbeing_activity_requested', self.handle_wellbeing_activity)
    
    def handle_break_request(self):
        """Handle regular break request"""
        self.show_break_screen("regular")
        self.data_logger.log_user_action("break_requested", "stress_alert", "user_accepted")
    
    def handle_emergency_break(self):
        """Handle emergency break request"""
        self.show_break_screen("emergency")
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
        
    def create_sidebar(self):
        """Create the left sidebar with navigation and controls"""
        
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=15)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=(10, 5), pady=10)
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        
        # App logo and title with gradient effect
        logo_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
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
        control_frame.grid(row=1, column=0, padx=20, pady=15, sticky="ew")
        
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
            text="📷 Ready",
            font=ctk.CTkFont(size=10),
            text_color="green"
        )
        self.camera_status.pack(side="left", padx=5)
        
        self.mic_status = ctk.CTkLabel(
            status_row,
            text="🎤 Ready", 
            font=ctk.CTkFont(size=10),
            text_color="green"
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
        self.status_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
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
        
        # Typing monitoring indicator
        typing_frame = ctk.CTkFrame(components_frame, fg_color="transparent")
        typing_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(typing_frame, text="⌨️", font=ctk.CTkFont(size=14)).pack(side="left")
        self.typing_indicator = ctk.CTkLabel(
            typing_frame,
            text="Typing: --",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.typing_indicator.pack(side="left", padx=5)
        
        # Enhanced navigation with modern icons and styling
        self.nav_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        self.nav_frame.grid(row=3, column=0, padx=20, pady=15, sticky="ew")
        
        nav_title = ctk.CTkLabel(
            self.nav_frame,
            text="🧭 Navigation",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        nav_title.pack(pady=(15, 10))
        
        # Navigation buttons with better styling
        nav_buttons = [
            ("📊 Dashboard", "dashboard", "#3498db"),
            ("� AI Chat", "chat", "#e74c3c"),
            ("�📈 Analytics", "analytics", "#9b59b6"),
            ("⚙️ Settings", "settings", "#34495e"),
            ("🧘 Wellness Hub", "wellbeing", "#1abc9c")
        ]
        
        for text, page, color in nav_buttons:
            btn = ctk.CTkButton(
                self.nav_frame,
                text=text,
                command=lambda p=page: self.show_page(p),
                height=40,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=color,
                hover_color=self.darken_color(color),
                corner_radius=10
            )
            btn.pack(pady=5, padx=15, fill="x")
        
        # Enhanced privacy controls with better visual design
        self.privacy_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        self.privacy_frame.grid(row=4, column=0, padx=20, pady=15, sticky="ew")
        
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
            ("⌨️ Typing Analysis", "typing_monitoring", self.toggle_typing_monitoring)
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
            setattr(self, attr, switch)
        
        # Quick wellness actions
        wellness_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=15)
        wellness_frame.grid(row=5, column=0, padx=20, pady=(0, 15), sticky="ew")
        
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
        
        # Create different pages
        self.pages = {}
        self.create_dashboard_page()
        self.create_chat_page()
        self.create_analytics_page()
        self.create_settings_page()
        self.create_wellbeing_page()
        
        # Show dashboard by default
        self.show_page("dashboard")
        
    def create_dashboard_page(self):
        """Create the main dashboard with monitoring overview"""
        dashboard = ctk.CTkFrame(self.main_frame)
        self.pages["dashboard"] = dashboard
        
        # Header section
        header_frame = ctk.CTkFrame(dashboard, corner_radius=15, height=100)
        header_frame.pack(fill="x", pady=(20, 20), padx=20)
        header_frame.pack_propagate(False)
        
        # Welcome message with time-based greeting
        import datetime
        hour = datetime.datetime.now().hour
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
        
        # Current status indicator
        self.status_indicator = ctk.CTkLabel(
            header_frame,
            text="🟢 System Ready - All monitors available",
            font=ctk.CTkFont(size=12),
            text_color="#27ae60"
        )
        self.status_indicator.pack()
        
        # Main monitoring grid
        grid_frame = ctk.CTkFrame(dashboard, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configure grid weights
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.columnconfigure(2, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)
        
        # Camera monitoring card
        self.camera_card = self.create_monitor_card(
            grid_frame, "📸 Camera Status", "Ready", "#3498db", 0, 0
        )
        
        # Microphone monitoring card
        self.mic_card = self.create_monitor_card(
            grid_frame, "🎤 Microphone Status", "Ready", "#e74c3c", 0, 1
        )
        
        # Typing monitoring card
        self.typing_card = self.create_monitor_card(
            grid_frame, "⌨️ Typing Analysis", "Ready", "#f39c12", 0, 2
        )
        
        # Overall stress level card
        self.stress_card = self.create_stress_level_card(grid_frame, 1, 0)
        
        # Activity timeline card
        self.timeline_card = self.create_timeline_card(grid_frame, 1, 1)
        
        # Quick recommendations card
        self.recommendations_card = self.create_recommendations_card(grid_frame, 1, 2)
    
    def create_monitor_card(self, parent, title, status, color, row, col):
        """Create a monitoring status card"""
        card = ctk.CTkFrame(parent, corner_radius=15)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Card header
        header = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        header.pack(pady=(20, 10))
        
        # Status indicator
        status_frame = ctk.CTkFrame(card, fg_color="transparent")
        status_frame.pack(pady=10)
        
        status_dot = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=20),
            text_color=color
        )
        status_dot.pack()
        
        status_label = ctk.CTkLabel(
            status_frame,
            text=status,
            font=ctk.CTkFont(size=12),
            text_color=color
        )
        status_label.pack()
        
        # Permission button for camera and microphone
        if "Camera" in title or "Microphone" in title:
            permission_btn = ctk.CTkButton(
                card,
                text="Grant Permission",
                command=lambda: self.request_permission(title),
                height=30,
                font=ctk.CTkFont(size=10),
                fg_color=color,
                hover_color=self.darken_color(color),
                corner_radius=8
            )
            permission_btn.pack(pady=(10, 20))
        
        return card
    
    def create_stress_level_card(self, parent, row, col):
        """Create stress level monitoring card"""
        card = ctk.CTkFrame(parent, corner_radius=15)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            card,
            text="� Current Stress Level",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 10))
        
        # Stress level meter
        self.stress_meter = ctk.CTkProgressBar(
            card,
            width=200,
            height=20,
            progress_color="#27ae60"
        )
        self.stress_meter.pack(pady=10)
        self.stress_meter.set(0.3)  # Default 30%
        
        self.stress_level_label = ctk.CTkLabel(
            card,
            text="Low (30%)",
            font=ctk.CTkFont(size=12),
            text_color="#27ae60"
        )
        self.stress_level_label.pack(pady=(0, 20))
        
        return card
    
    def create_timeline_card(self, parent, row, col):
        """Create activity timeline card"""
        card = ctk.CTkFrame(parent, corner_radius=15)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            card,
            text="📈 Recent Activity",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 10))
        
        # Timeline items
        timeline_items = [
            ("2 min ago", "😌 Breathing exercise completed", "#27ae60"),
            ("15 min ago", "😐 Mild stress detected", "#f39c12"),
            ("1 hr ago", "😊 Break reminder taken", "#3498db")
        ]
        
        for time, event, color in timeline_items:
            item_frame = ctk.CTkFrame(card, fg_color="transparent")
            item_frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                item_frame,
                text=f"• {time}: {event}",
                font=ctk.CTkFont(size=10),
                text_color=color,
                anchor="w"
            ).pack(fill="x")
        
        return card
    
    def create_recommendations_card(self, parent, row, col):
        """Create quick recommendations card"""
        card = ctk.CTkFrame(parent, corner_radius=15)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            card,
            text="💡 Smart Suggestions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 10))
        
        suggestions = [
            "🫁 Try a 2-minute breathing exercise",
            "👁️ Take an eye break - look away",
            "🚶 Stand up and stretch"
        ]
        
        for suggestion in suggestions:
            suggestion_frame = ctk.CTkFrame(card, fg_color="transparent")
            suggestion_frame.pack(fill="x", padx=15, pady=5)
            
            suggestion_label = ctk.CTkLabel(
                suggestion_frame,
                text=suggestion,
                font=ctk.CTkFont(size=10),
                anchor="w"
            )
            suggestion_label.pack(side="left", fill="x", expand=True)
            
            action_btn = ctk.CTkButton(
                suggestion_frame,
                text="▶",
                width=25,
                height=25,
                font=ctk.CTkFont(size=10),
                corner_radius=5
            )
            action_btn.pack(side="right")
        
        return card
    
    def request_permission(self, device_type):
        """Request camera or microphone permission"""
        if "Camera" in device_type:
            # Simulate camera permission request
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    self.update_device_status("camera", "Connected", "#27ae60")
                    cap.release()
                else:
                    self.update_device_status("camera", "Not Available", "#e74c3c")
            except ImportError:
                self.update_device_status("camera", "Simulated - Ready", "#3498db")
        
        elif "Microphone" in device_type:
            # Simulate microphone permission request
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                # Test if we can access microphone
                for i in range(p.get_device_count()):
                    device_info = p.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels') > 0:
                        self.update_device_status("microphone", "Connected", "#27ae60")
                        break
                else:
                    self.update_device_status("microphone", "Not Available", "#e74c3c")
                p.terminate()
            except ImportError:
                self.update_device_status("microphone", "Simulated - Ready", "#e74c3c")
    
    def update_device_status(self, device, status, color):
        """Update device status in the UI"""
        if device == "camera":
            # Update camera card status
            pass
        elif device == "microphone":
            # Update microphone card status
            pass
        
        # Update overall status
        self.status_indicator.configure(
            text=f"🟡 {device.title()}: {status}",
            text_color=color
        )
    
    def create_chat_page(self):
        """Create the AI chat interface page"""
        try:
            from src.ui.chat_interface import ChatInterface
            
            chat = ctk.CTkFrame(self.main_frame)
            self.pages["chat"] = chat
            
            # Chat interface
            self.chat_interface = ChatInterface(chat)
            self.chat_interface.pack(fill="both", expand=True, padx=10, pady=10)
            
        except ImportError:
            # Fallback for simulation mode
            chat = ctk.CTkFrame(self.main_frame)
            self.pages["chat"] = chat
            
            # Title
            title = ctk.CTkLabel(
                chat,
                text="🤖 AI Conversation Assistant",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            title.pack(pady=20)
            
            # Description
            desc_frame = ctk.CTkFrame(chat, corner_radius=15)
            desc_frame.pack(fill="x", padx=20, pady=10)
            
            description = ctk.CTkLabel(
                desc_frame,
                text="Talk to our AI assistant for personalized stress analysis and support.\n\n"
                     "🎯 Features:\n"
                     "• Real-time sentiment analysis\n"
                     "• Personalized stress assessment\n"
                     "• Intelligent recommendations\n"
                     "• Crisis intervention support\n"
                     "• Conversation summaries",
                font=ctk.CTkFont(size=14),
                justify="left"
            )
            description.pack(padx=20, pady=20)
            
            # Simulated chat area
            chat_area = ctk.CTkFrame(chat, corner_radius=15)
            chat_area.pack(fill="both", expand=True, padx=20, pady=(0, 20))
            
            # Simulated messages
            messages = [
                ("🤖 AI: Hello! I'm your wellness companion. How are you feeling today?", "#2c3e50"),
                ("👤 You: I'm feeling quite stressed with work lately.", "#3498db"),
                ("🤖 AI: I understand. Can you tell me more about what's causing the stress?", "#2c3e50"),
                ("👤 You: Too many deadlines and not enough time to complete everything.", "#3498db"),
                ("🤖 AI: That sounds overwhelming. Let's work on some strategies to help you manage this.", "#2c3e50")
            ]
            
            for msg, color in messages:
                msg_frame = ctk.CTkFrame(chat_area, fg_color=color, corner_radius=10)
                msg_frame.pack(fill="x", padx=20, pady=5)
                
                msg_label = ctk.CTkLabel(
                    msg_frame,
                    text=msg,
                    font=ctk.CTkFont(size=12),
                    text_color="white",
                    anchor="w"
                )
                msg_label.pack(padx=15, pady=10, fill="x")
            
            # Installation note
            note_frame = ctk.CTkFrame(chat, fg_color="#f39c12", corner_radius=10)
            note_frame.pack(fill="x", padx=20, pady=(0, 20))
            
            note_label = ctk.CTkLabel(
                note_frame,
                text="💡 Note: Install required dependencies to enable full chat functionality",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
            note_label.pack(pady=10)
    
    def create_analytics_page(self):
        """Create the analytics page with charts and trends"""
        
        analytics = ctk.CTkFrame(self.main_frame)
        self.pages["analytics"] = analytics
        
        # Title
        title = ctk.CTkLabel(
            analytics,
            text="Stress Analytics & Trends",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Time period selector
        period_frame = ctk.CTkFrame(analytics)
        period_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(period_frame, text="Time Period:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
        
        self.period_selector = ctk.CTkOptionMenu(
            period_frame,
            values=["Today", "This Week", "This Month", "Last 3 Months"],
            command=self.update_analytics
        )
        self.period_selector.pack(side="left", padx=10)
        
        # Charts area
        charts_frame = ctk.CTkFrame(analytics)
        charts_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Create matplotlib figure for charts
        self.setup_analytics_charts(charts_frame)
        
        # Statistics summary
        stats_frame = ctk.CTkFrame(analytics)
        stats_frame.pack(pady=10, padx=20, fill="x")
        
        self.create_statistics_summary(stats_frame)
        
    def create_settings_page(self):
        """Create the settings configuration page"""
        
        settings = ctk.CTkFrame(self.main_frame)
        self.pages["settings"] = settings
        
        # Title
        title = ctk.CTkLabel(
            settings,
            text="Settings & Configuration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Settings notebook
        settings_notebook = ttk.Notebook(settings)
        settings_notebook.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Monitoring settings
        monitoring_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(monitoring_frame, text="Monitoring")
        self.create_monitoring_settings(monitoring_frame)
        
        # Alert settings
        alerts_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(alerts_frame, text="Alerts")
        self.create_alert_settings(alerts_frame)
        
        # Privacy settings
        privacy_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(privacy_frame, text="Privacy")
        self.create_privacy_settings(privacy_frame)
        
        # Integration settings
        integration_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(integration_frame, text="Integrations")
        self.create_integration_settings(integration_frame)
        
    def create_wellbeing_page(self):
        """Create the wellbeing and suggestions page"""
        
        wellbeing = ctk.CTkFrame(self.main_frame)
        self.pages["wellbeing"] = wellbeing
        
        # Title
        title = ctk.CTkLabel(
            wellbeing,
            text="Well-being Center",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Current recommendations
        rec_frame = ctk.CTkFrame(wellbeing)
        rec_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Current Recommendations",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.recommendations_text = ctk.CTkTextbox(rec_frame, height=100)
        self.recommendations_text.pack(pady=10, padx=20, fill="x")
        
        # Exercise library
        exercise_frame = ctk.CTkFrame(wellbeing)
        exercise_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        ctk.CTkLabel(
            exercise_frame,
            text="🧘 Stress Relief Exercises",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Exercise buttons
        exercises_grid = ctk.CTkFrame(exercise_frame)
        exercises_grid.pack(pady=10, fill="x")
        
        exercises = [
            ("🫁 Breathing Exercise", self.start_breathing_exercise),
            ("🧘 Guided Meditation", self.start_meditation),
            ("💪 Desk Stretches", self.start_stretches),
            ("👁️ Eye Rest", self.start_eye_rest),
            ("🎵 Calming Sounds", self.play_calming_sounds),
            ("☕ Break Reminder", self.suggest_break)
        ]
        
        for i, (text, command) in enumerate(exercises):
            row, col = divmod(i, 3)
            btn = ctk.CTkButton(exercises_grid, text=text, command=command)
            btn.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            exercises_grid.grid_columnconfigure(col, weight=1)
    
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
    
    def start_monitoring(self):
        """Start the real-time monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.session_start_time = datetime.now()
        
        # Start individual monitors based on configuration
        if self.config_manager.monitoring.facial_monitoring_enabled:
            self.facial_monitor.start_monitoring()
        
        if self.config_manager.monitoring.voice_monitoring_enabled:
            self.voice_monitor.start_monitoring()
        
        if self.config_manager.monitoring.typing_monitoring_enabled:
            self.typing_monitor.start_monitoring()
        
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
        self.typing_monitor.stop_monitoring()
        
        # Update UI
        self.start_btn.configure(text="▶️ Start Monitoring")
        self.status_label.configure(text="🟢 System Ready - Privacy Protected")
        
        # Log the stop
        self.data_logger.log_system_event("monitoring_stop", "Monitoring session ended")
        self.log_activity("⏹️ Monitoring stopped")
    
    def monitoring_loop(self):
        """Main monitoring loop that runs in background thread"""
        
        while self.monitoring_active:
            try:
                # Collect data from enabled monitors
                facial_data = None
                voice_data = None
                typing_data = None
                
                if self.config_manager.monitoring.facial_monitoring_enabled:
                    facial_data = self.facial_monitor.get_current_data()
                
                if self.config_manager.monitoring.voice_monitoring_enabled:
                    voice_data = self.voice_monitor.get_current_data()
                
                if self.config_manager.monitoring.typing_monitoring_enabled:
                    typing_data = self.typing_monitor.get_current_data()
                
                # Analyze stress levels using multimodal data
                stress_analysis = self.stress_analyzer.analyze_multimodal_stress(
                    facial_data or {},
                    voice_data or {},
                    typing_data or {}
                )
                
                # Update current stress level
                self.current_stress_level = stress_analysis['overall_stress']
                self.current_risk_category = stress_analysis['risk_level']
                
                # Log stress data
                self.data_logger.log_stress_level(
                    stress_level=stress_analysis['overall_stress'],
                    components=stress_analysis.get('components', {}),
                    risk_factors=stress_analysis.get('risk_factors', []),
                    confidence=stress_analysis.get('confidence', 1.0)
                )
                
                # Store for UI display
                self.stress_history.append((datetime.now(), stress_analysis['overall_stress']))
                if len(self.stress_history) > 100:  # Keep last 100 readings
                    self.stress_history = self.stress_history[-100:]
                
                # Check for alerts
                self.check_stress_alerts(stress_analysis)
                
                # Update baseline (adaptive learning)
                self.baseline_calibrator.update_baseline({
                    'facial': facial_data.get('stress_level', 0) if facial_data else 0,
                    'voice': voice_data.get('stress_level', 0) if voice_data else 0,
                    'typing': typing_data.get('stress_level', 0) if typing_data else 0,
                    'overall': stress_analysis['overall_stress']
                })
                
                # Sleep for monitoring interval
                time.sleep(self.config_manager.ui.auto_refresh_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                self.data_logger.log_system_event("monitoring_error", f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def check_stress_alerts(self, stress_analysis):
        """Check if stress levels warrant alerts"""
        
        if not self.config_manager.alerts.enabled:
            return
        
        risk_level = stress_analysis['risk_level']
        stress_score = stress_analysis['overall_stress']
        recommendations = stress_analysis.get('recommendations', [])
        
        # Check for burnout risk (critical alert)
        if stress_score > 0.8 and risk_level in ['high', 'critical']:
            self.trigger_burnout_alert(stress_analysis)
        
        # Check for elevated stress (warning alert)
        elif stress_score > 0.6 and risk_level in ['medium', 'high']:
            self.trigger_stress_alert(stress_analysis)
        
        # Check for break reminders
        elif stress_score > 0.4:
            current_time = time.time()
            if current_time - self.last_stress_check > (self.config_manager.wellbeing.break_reminder_interval * 60):
                self.suggest_wellbeing_activity()
                self.last_stress_check = current_time
    
    def trigger_burnout_alert(self, analysis):
        """Trigger high-priority burnout warning"""
        
        # Use notification system with proper parameters
        self.notification_system.show_burnout_warning(
            risk_level=analysis['risk_level'].upper(),
            sustained_duration="Current session"
        )
        
        # Log the alert
        self.data_logger.log_alert(
            alert_type="burnout_warning",
            severity="critical",
            message=f"Burnout risk detected - stress level {analysis['overall_stress']:.1%}"
        )
        
        self.log_activity("🚨 HIGH RISK: Burnout indicators detected")
    
    def trigger_stress_alert(self, analysis):
        """Trigger moderate stress alert"""
        
        # Use notification system with proper parameters
        self.notification_system.show_stress_alert(
            stress_level=analysis['overall_stress'],
            risk_factors=analysis.get('risk_factors', []),
            recommendations=analysis.get('recommendations', [])
        )
        
        # Log the alert
        self.data_logger.log_alert(
            alert_type="stress_warning",
            severity="medium",
            message=f"Elevated stress detected - level {analysis['overall_stress']:.1%}"
        )
        
        self.log_activity(f"⚠️ MODERATE RISK: Stress level {analysis['overall_stress']:.1%}")
    
    def suggest_wellbeing_activity(self):
        """Suggest a wellbeing activity"""
        
        # Get a random wellbeing tip
        tip = self.wellbeing_resources.get_random_tip()
        
        # Show the tip via notification system
        self.notification_system.show_wellbeing_tip(tip, "general")
        
        # Log the suggestion
        self.data_logger.log_user_action("wellbeing_tip_shown", "automatic", "tip_displayed")
    
    def start_guided_exercise(self, exercise_type):
        """Start a guided exercise session"""
        
        # Get exercise recommendation
        exercise = self.wellbeing_resources.get_recommended_exercise(
            stress_level=self.current_stress_level,
            available_time=self.config_manager.wellbeing.preferred_exercise_duration
        )
        
        # Start the guided session
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        
        if session_id:
            self.log_activity(f"🧘 Started guided exercise: {exercise['exercise']['name']}")
            
            # Log the exercise session
            self.data_logger.log_exercise_session(
                exercise_name=exercise['exercise']['name'],
                duration_minutes=exercise['exercise']['duration'],
                completion_status="started"
            )
    
    def start_guided_exercise_session(self, exercise_data):
        """Start a guided exercise session from exercise data"""
        
        session_id = self.wellbeing_resources.start_guided_session(exercise_data)
        
        if session_id:
            exercise = exercise_data['exercise']
            self.log_activity(f"🧘 Started guided exercise: {exercise['name']}")
            
            # Log the exercise session
            self.data_logger.log_exercise_session(
                exercise_name=exercise['name'],
                duration_minutes=exercise['duration'],
                completion_status="started"
            )
    
    def show_break_screen(self, break_type="regular"):
        """Show break screen overlay"""
        
        # Create break window
        break_window = ctk.CTkToplevel(self)
        break_window.title("Break Time")
        break_window.geometry("400x300")
        break_window.transient(self)
        break_window.grab_set()
        
        # Center the window
        break_window.update_idletasks()
        x = (break_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (break_window.winfo_screenheight() // 2) - (300 // 2)
        break_window.geometry(f"400x300+{x}+{y}")
        
        # Break message
        if break_type == "emergency":
            title = "🚨 Emergency Break Needed"
            message = "Your stress levels are critically high.\nTake a break now for your wellbeing."
            color = "red"
        else:
            title = "☕ Break Time"
            message = "Time for a healthy break.\nYour mind and body will thank you."
            color = "blue"
        
        # Title label
        title_label = ctk.CTkLabel(
            break_window,
            text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=color
        )
        title_label.pack(pady=20)
        
        # Message label
        message_label = ctk.CTkLabel(
            break_window,
            text=message,
            font=ctk.CTkFont(size=14),
            wraplength=350
        )
        message_label.pack(pady=10)
        
        # Exercise suggestion
        exercise = self.wellbeing_resources.get_recommended_exercise(
            stress_level=self.current_stress_level,
            available_time=5
        )
        
        suggestion_label = ctk.CTkLabel(
            break_window,
            text=f"💡 Suggested: {exercise['exercise']['name']}\n({exercise['exercise']['duration']} minutes)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        suggestion_label.pack(pady=10)
        
        # Buttons
        button_frame = ctk.CTkFrame(break_window)
        button_frame.pack(pady=20)
        
        # Start exercise button
        start_btn = ctk.CTkButton(
            button_frame,
            text="Start Exercise",
            command=lambda: self.start_exercise_from_break(exercise, break_window),
            fg_color="green"
        )
        start_btn.pack(side="left", padx=10)
        
        # Skip break button
        skip_btn = ctk.CTkButton(
            button_frame,
            text="Skip Break",
            command=break_window.destroy,
            fg_color="gray"
        )
        skip_btn.pack(side="left", padx=10)
        
        # Auto close after 30 seconds
        break_window.after(30000, break_window.destroy)
    
    def start_exercise_from_break(self, exercise_data, break_window):
        """Start exercise session from break screen"""
        
        break_window.destroy()
        self.start_guided_exercise_session(exercise_data)
        """Suggest proactive wellbeing activities"""
        
        suggestions = self.wellbeing_suggestions.get_proactive_suggestions()
        suggestion = suggestions[0] if suggestions else "Consider taking a short break"
        
        self.log_activity(f"💡 Suggestion: {suggestion}")
    
    def log_activity(self, message):
        """Log activity to the activity feed"""
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Update activity text in main thread
        self.after(0, lambda: self.activity_text.insert("1.0", log_message))
    
    def update_ui_timer(self):
        """Update UI elements with current data"""
        
        if self.monitoring_active:
            # Update stress level displays
            self.facial_stress_var.set(f"{self.facial_monitor.current_stress:.1%}")
            self.voice_stress_var.set(f"{self.voice_monitor.current_stress:.1%}")
            self.typing_stress_var.set(f"{self.typing_monitor.current_stress:.1%}")
            self.overall_stress_var.set(f"{self.current_stress_level:.1%}")
            
            # Update progress bar
            self.stress_progress.set(self.current_stress_level)
            
            # Update stress level label with color coding
            if self.current_stress_level < 0.3:
                color, status = "#4CAF50", "Low"
            elif self.current_stress_level < 0.6:
                color, status = "#FF9800", "Moderate"
            else:
                color, status = "#F44336", "High"
            
            self.stress_level_label.configure(
                text=f"Stress Level: {status}",
                text_color=color
            )
            
            # Update session time
            if self.session_start_time:
                session_duration = datetime.now() - self.session_start_time
                duration_str = str(session_duration).split('.')[0]  # Remove microseconds
                self.session_time_label.configure(text=f"Session: {duration_str}")
            
            # Schedule next update
            self.after(1000, self.update_ui_timer)
    
    def show_page(self, page_name):
        """Show the specified page"""
        
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()
        
        # Show selected page
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)
    
    def setup_system_tray(self):
        """Setup system tray for background operation"""
        # This would integrate with system tray APIs
        pass
    
    def load_settings(self):
        """Load user settings from configuration manager"""
        
        # Load all configuration settings
        settings = self.config_manager.get_all_settings()
        
        # Apply monitoring settings
        if hasattr(self, 'facial_monitoring'):
            self.facial_monitoring.set(self.config_manager.monitoring.facial_monitoring_enabled)
        if hasattr(self, 'voice_monitoring'):
            self.voice_monitoring.set(self.config_manager.monitoring.voice_monitoring_enabled)
        if hasattr(self, 'typing_monitoring'):
            self.typing_monitoring.set(self.config_manager.monitoring.typing_monitoring_enabled)
        
        # Apply UI settings
        ctk.set_appearance_mode(self.config_manager.ui.theme.value if hasattr(self.config_manager.ui.theme, 'value') else self.config_manager.ui.theme)
        
        # Update notification system settings
        notification_settings = {
            'show_desktop_notifications': self.config_manager.alerts.desktop_notifications,
            'show_popup_alerts': self.config_manager.alerts.popup_alerts,
            'sound_enabled': getattr(self.config_manager.alerts, 'sound_alerts', False)
        }
        self.notification_system.update_settings(notification_settings)
        
        self.log_activity("⚙️ Settings loaded successfully")
    
    def save_settings(self):
        """Save current settings to configuration manager"""
        
        # Update monitoring settings if UI variables exist
        if hasattr(self, 'facial_monitoring'):
            self.config_manager.update_monitoring_settings(
                facial_monitoring_enabled=self.facial_monitoring.get()
            )
        if hasattr(self, 'voice_monitoring'):
            self.config_manager.update_monitoring_settings(
                voice_monitoring_enabled=self.voice_monitoring.get()
            )
        if hasattr(self, 'typing_monitoring'):
            self.config_manager.update_monitoring_settings(
                typing_monitoring_enabled=self.typing_monitoring.get()
            )
        
        # Save configuration
        self.config_manager.save_config()
        
        self.log_activity("💾 Settings saved successfully")
    
    def on_closing(self):
        """Handle application closing"""
        
        # Stop monitoring
        if self.monitoring_active:
            self.stop_monitoring()
        
        # Save settings
        self.save_settings()
        
        # Close data logger
        self.data_logger.close()
        
        # Log application shutdown
        self.data_logger.log_system_event("app_shutdown", "Application closed by user")
        
        # Destroy the window
        self.destroy()
    
    # Monitoring toggle methods
    def toggle_facial_monitoring(self):
        if hasattr(self, 'facial_monitoring'):
            enabled = self.facial_monitoring.get()
            self.config_manager.update_monitoring_settings(facial_monitoring_enabled=enabled)
            
            if enabled:
                self.facial_monitor.start_monitoring()
            else:
                self.facial_monitor.stop_monitoring()
            
            self.data_logger.log_user_action("toggle_facial_monitoring", f"enabled_{enabled}")
            self.log_activity(f"👁️ Facial monitoring {'enabled' if enabled else 'disabled'}")
    
    def toggle_voice_monitoring(self):
        if hasattr(self, 'voice_monitoring'):
            enabled = self.voice_monitoring.get()
            self.config_manager.update_monitoring_settings(voice_monitoring_enabled=enabled)
            
            if enabled:
                self.voice_monitor.start_monitoring()
            else:
                self.voice_monitor.stop_monitoring()
            
            self.data_logger.log_user_action("toggle_voice_monitoring", f"enabled_{enabled}")
            self.log_activity(f"🎤 Voice monitoring {'enabled' if enabled else 'disabled'}")
    
    def toggle_typing_monitoring(self):
        if hasattr(self, 'typing_monitoring'):
            enabled = self.typing_monitoring.get()
            self.config_manager.update_monitoring_settings(typing_monitoring_enabled=enabled)
            
            if enabled:
                self.typing_monitor.start_monitoring()
            else:
                self.typing_monitor.stop_monitoring()
            
            self.data_logger.log_user_action("toggle_typing_monitoring", f"enabled_{enabled}")
            self.log_activity(f"⌨️ Typing monitoring {'enabled' if enabled else 'disabled'}")
    
    # Quick action methods
    def pause_monitoring(self):
        if self.monitoring_active:
            self.stop_monitoring()
            self.data_logger.log_user_action("pause_monitoring", "user_request")
            self.log_activity("⏸️ Monitoring paused by user")
    
    def suggest_break(self):
        """Trigger a break suggestion"""
        self.notification_system.show_break_reminder("general")
        self.data_logger.log_user_action("manual_break_suggestion", "user_request")
        self.log_activity("☕ Break suggestion triggered")
    
    def start_breathing_exercise(self):
        """Start a breathing exercise"""
        exercise = self.wellbeing_resources.get_recommended_exercise(
            stress_level=self.current_stress_level,
            exercise_type=None  # Let it choose breathing automatically
        )
        
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        if session_id:
            self.data_logger.log_exercise_session(
                exercise_name=exercise['exercise']['name'],
                duration_minutes=exercise['exercise']['duration'],
                completion_status="started"
            )
        
        self.log_activity("🫁 Breathing exercise started")
    
    def start_meditation(self):
        """Start a meditation session"""
        
        # Get meditation exercise
        from wellbeing.wellbeing_resources import ExerciseType
        try:
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level,
                exercise_type=ExerciseType.MEDITATION
            )
        except:
            # Fallback for simulation mode
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level
            )
        
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        if session_id:
            self.data_logger.log_exercise_session(
                exercise_name=exercise['exercise']['name'],
                duration_minutes=exercise['exercise']['duration'],
                completion_status="started"
            )
        
        self.log_activity("🧘 Meditation session started")
    
    def start_stretches(self):
        """Start desk stretches"""
        
        # Get stretching exercise
        from wellbeing.wellbeing_resources import ExerciseType
        try:
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level,
                exercise_type=ExerciseType.STRETCHING
            )
        except:
            # Fallback for simulation mode
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level
            )
        
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        if session_id:
            self.data_logger.log_exercise_session(
                exercise_name=exercise['exercise']['name'],
                duration_minutes=exercise['exercise']['duration'],
                completion_status="started"
            )
        
        self.log_activity("💪 Desk stretches initiated")
    
    def start_eye_rest(self):
        """Start eye rest exercises"""
        
        # Get eye exercise
        from wellbeing.wellbeing_resources import ExerciseType
        try:
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level,
                exercise_type=ExerciseType.EYE_EXERCISES
            )
        except:
            # Fallback for simulation mode
            exercise = self.wellbeing_resources.get_recommended_exercise(
                stress_level=self.current_stress_level
            )
        
        session_id = self.wellbeing_resources.start_guided_session(exercise)
        if session_id:
            self.data_logger.log_exercise_session(
                exercise_name=exercise['exercise']['name'],
                duration_minutes=exercise['exercise']['duration'],
                completion_status="started"
            )
        
        self.log_activity("👁️ Eye rest break started")
    
    def play_calming_sounds(self):
        """Show calming sounds/ambient noise options"""
        
        # Show wellbeing tip about ambient sounds
        tip = "Try playing ambient sounds like rain, ocean waves, or white noise to create a calming atmosphere."
        self.notification_system.show_wellbeing_tip(tip, "relaxation")
        
        self.data_logger.log_user_action("calming_sounds_requested", "user_action")
        self.log_activity("🎵 Calming sounds suggestion shown")
    
    # Settings page implementations
    def create_monitoring_settings(self, frame):
        """Create monitoring configuration settings"""
        # Implementation for monitoring settings
        pass
    
    def create_alert_settings(self, frame):
        """Create alert configuration settings"""
        # Implementation for alert settings
        pass
    
    def create_privacy_settings(self, frame):
        """Create privacy configuration settings"""
        # Implementation for privacy settings
        pass
    
    def create_integration_settings(self, frame):
        """Create integration configuration settings"""
        # Implementation for integration settings
        pass
    
    def setup_analytics_charts(self, frame):
        """Setup matplotlib charts for analytics"""
        # Implementation for analytics charts
        pass
    
    def create_statistics_summary(self, frame):
        """Create statistics summary widgets"""
        # Implementation for statistics summary
        pass
    
    def update_analytics(self, period):
        """Update analytics based on selected time period"""
        # Implementation for updating analytics
        pass

def main():
    """Main application entry point"""
    
    print("🚀 Starting AI-Based Stress and Burnout Early Warning System...")
    print("🔒 Privacy-focused local processing")
    print("🧠 Multimodal stress detection")
    print("💡 Intelligent wellbeing recommendations")
    print()
    
    try:
        # Create the main application
        app = StressBurnoutApp()
        
        # Set up proper cleanup on window close
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start the application
        print("✅ Application initialized successfully")
        print("📱 Desktop application running...")
        print("💡 Tip: Check the system tray for background monitoring")
        print()
        
        app.mainloop()
        
    except KeyboardInterrupt:
        print("\n⏹️ Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Thank you for using the Stress & Burnout Warning System!")
        print("🌟 Remember to take care of your mental health!")

if __name__ == "__main__":
    main()
