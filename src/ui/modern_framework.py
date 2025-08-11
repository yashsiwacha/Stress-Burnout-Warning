"""
Modern UI Framework for Stress Monitor V2.0
Built on CustomTkinter with advanced features and responsive design
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import json

from ..core import get_system_core, SystemEvent, EventType

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Default to dark mode
ctk.set_default_color_theme("blue")  # Blue color theme

class ThemeManager:
    """Manages application themes and styling"""
    
    def __init__(self):
        self.current_theme = "dark"
        self.themes = {
            "dark": {
                "bg_color": "#1a1a1a",
                "fg_color": "#ffffff",
                "accent_color": "#00a8ff",
                "success_color": "#00d862",
                "warning_color": "#ff9500",
                "error_color": "#ff453a",
                "card_color": "#2d2d2d",
                "hover_color": "#3d3d3d"
            },
            "light": {
                "bg_color": "#ffffff", 
                "fg_color": "#000000",
                "accent_color": "#007aff",
                "success_color": "#30d158",
                "warning_color": "#ff9500",
                "error_color": "#ff3b30",
                "card_color": "#f2f2f7",
                "hover_color": "#e5e5ea"
            }
        }
        self.callbacks: List[Callable] = []
    
    def set_theme(self, theme_name: str):
        """Set the application theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            ctk.set_appearance_mode(theme_name)
            
            # Notify all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(theme_name)
                except Exception as e:
                    print(f"Theme callback error: {e}")
    
    def get_color(self, color_key: str) -> str:
        """Get a color value for the current theme"""
        return self.themes[self.current_theme].get(color_key, "#000000")
    
    def register_callback(self, callback: Callable):
        """Register a callback for theme changes"""
        self.callbacks.append(callback)

class Widget(ABC):
    """Base class for all UI widgets"""
    
    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.widget = None
        self.theme_manager = ThemeManager()
        self.visible = True
        self.enabled = True
        
    @abstractmethod
    def create_widget(self) -> ctk.CTkBaseClass:
        """Create the actual widget"""
        pass
    
    def show(self):
        """Show the widget"""
        if self.widget:
            self.widget.grid()
            self.visible = True
    
    def hide(self):
        """Hide the widget"""
        if self.widget:
            self.widget.grid_remove()
            self.visible = False
    
    def enable(self):
        """Enable the widget"""
        if self.widget:
            self.widget.configure(state="normal")
            self.enabled = True
    
    def disable(self):
        """Disable the widget"""
        if self.widget:
            self.widget.configure(state="disabled")
            self.enabled = False

class StressIndicator(Widget):
    """Circular stress level indicator widget"""
    
    def __init__(self, parent, size=100, **kwargs):
        super().__init__(parent, **kwargs)
        self.size = size
        self.stress_level = 0.0
        self.widget = self.create_widget()
    
    def create_widget(self) -> ctk.CTkProgressBar:
        """Create circular progress indicator"""
        # For now using progress bar, will upgrade to custom circular widget
        progress = ctk.CTkProgressBar(
            self.parent,
            width=self.size,
            height=20,
            progress_color=self.theme_manager.get_color("accent_color")
        )
        return progress
    
    def update_stress_level(self, level: float, risk_level: str = "low"):
        """Update the stress level display"""
        self.stress_level = max(0.0, min(1.0, level))
        
        # Update color based on risk level
        color_map = {
            "low": self.theme_manager.get_color("success_color"),
            "medium": self.theme_manager.get_color("warning_color"),
            "high": self.theme_manager.get_color("error_color"),
            "critical": "#ff0000"
        }
        
        color = color_map.get(risk_level, self.theme_manager.get_color("accent_color"))
        
        if self.widget:
            self.widget.configure(progress_color=color)
            self.widget.set(self.stress_level)

class MetricCard(Widget):
    """Card widget for displaying metrics"""
    
    def __init__(self, parent, title: str, value: str = "0", unit: str = "", **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.value = value
        self.unit = unit
        self.widget = self.create_widget()
    
    def create_widget(self) -> ctk.CTkFrame:
        """Create metric card widget"""
        frame = ctk.CTkFrame(
            self.parent,
            fg_color=self.theme_manager.get_color("card_color"),
            corner_radius=10
        )
        
        # Title label
        title_label = ctk.CTkLabel(
            frame,
            text=self.title,
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=self.theme_manager.get_color("fg_color")
        )
        title_label.pack(pady=(10, 5))
        
        # Value label
        self.value_label = ctk.CTkLabel(
            frame,
            text=f"{self.value} {self.unit}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.theme_manager.get_color("accent_color")
        )
        self.value_label.pack(pady=(0, 10))
        
        return frame
    
    def update_value(self, value: str, unit: str = None):
        """Update the metric value"""
        self.value = value
        if unit is not None:
            self.unit = unit
        
        if hasattr(self, 'value_label'):
            self.value_label.configure(text=f"{self.value} {self.unit}")

class AlertPanel(Widget):
    """Panel for displaying alerts and notifications"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.alerts: List[Dict[str, Any]] = []
        self.widget = self.create_widget()
    
    def create_widget(self) -> ctk.CTkScrollableFrame:
        """Create scrollable alert panel"""
        frame = ctk.CTkScrollableFrame(
            self.parent,
            fg_color=self.theme_manager.get_color("card_color"),
            corner_radius=10
        )
        
        # Header
        header = ctk.CTkLabel(
            frame,
            text="🚨 Alerts & Notifications",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.theme_manager.get_color("fg_color")
        )
        header.pack(pady=10, padx=20, anchor="w")
        
        return frame
    
    def add_alert(self, alert_type: str, message: str, severity: str = "low"):
        """Add a new alert to the panel"""
        alert_data = {
            "id": f"alert_{len(self.alerts)}",
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now()
        }
        
        self.alerts.append(alert_data)
        self._create_alert_widget(alert_data)
    
    def _create_alert_widget(self, alert_data: Dict[str, Any]):
        """Create widget for individual alert"""
        severity_colors = {
            "low": self.theme_manager.get_color("accent_color"),
            "medium": self.theme_manager.get_color("warning_color"),
            "high": self.theme_manager.get_color("error_color"),
            "critical": "#ff0000"
        }
        
        alert_frame = ctk.CTkFrame(
            self.widget,
            fg_color=severity_colors.get(alert_data["severity"], "#333333"),
            corner_radius=5
        )
        alert_frame.pack(fill="x", padx=10, pady=5)
        
        # Alert message
        message_label = ctk.CTkLabel(
            alert_frame,
            text=alert_data["message"],
            font=ctk.CTkFont(size=12),
            text_color="white",
            wraplength=300
        )
        message_label.pack(pady=10, padx=10, anchor="w")
        
        # Timestamp
        time_label = ctk.CTkLabel(
            alert_frame,
            text=alert_data["timestamp"].strftime("%H:%M:%S"),
            font=ctk.CTkFont(size=10),
            text_color="white"
        )
        time_label.pack(pady=(0, 10), padx=10, anchor="e")

class ModernDashboard(ctk.CTkFrame):
    """Main dashboard interface with modern design"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.theme_manager = ThemeManager()
        self.system_core = get_system_core()
        
        # Dashboard components
        self.stress_indicator = None
        self.metric_cards: Dict[str, MetricCard] = {}
        self.alert_panel = None
        
        self.setup_dashboard()
        self.setup_event_handlers()
    
    def setup_dashboard(self):
        """Set up the dashboard layout"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Header section
        self.create_header()
        
        # Main stress indicator
        self.create_stress_section()
        
        # Metrics section
        self.create_metrics_section()
        
        # Alerts section
        self.create_alerts_section()
    
    def create_header(self):
        """Create dashboard header"""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=20, pady=(20, 10))
        
        # Title
        title = ctk.CTkLabel(
            header_frame,
            text="🧠 Stress Monitor V2.0 Dashboard",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=self.theme_manager.get_color("fg_color")
        )
        title.pack(side="left")
        
        # Theme toggle button
        theme_btn = ctk.CTkButton(
            header_frame,
            text="🌙/☀️",
            width=50,
            command=self.toggle_theme,
            font=ctk.CTkFont(size=16)
        )
        theme_btn.pack(side="right", padx=10)
    
    def create_stress_section(self):
        """Create main stress indicator section"""
        stress_frame = ctk.CTkFrame(self, corner_radius=15)
        stress_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20, pady=10)
        
        # Stress level label
        stress_label = ctk.CTkLabel(
            stress_frame,
            text="Current Stress Level",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        stress_label.pack(pady=(20, 10))
        
        # Stress indicator
        self.stress_indicator = StressIndicator(stress_frame, size=200)
        self.stress_indicator.widget.pack(pady=10)
        
        # Stress level text
        self.stress_text = ctk.CTkLabel(
            stress_frame,
            text="0% - Relaxed",
            font=ctk.CTkFont(size=16)
        )
        self.stress_text.pack(pady=(10, 20))
    
    def create_metrics_section(self):
        """Create metrics cards section"""
        metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        metrics_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=10)
        
        # Configure grid for metrics
        for i in range(4):
            metrics_frame.grid_columnconfigure(i, weight=1)
        
        # Create metric cards
        metrics = [
            ("Heart Rate", "72", "BPM"),
            ("Session Time", "0", "min"),
            ("Facial Analysis", "Active", ""),
            ("Voice Analysis", "Active", "")
        ]
        
        for i, (title, value, unit) in enumerate(metrics):
            card = MetricCard(metrics_frame, title, value, unit)
            card.widget.grid(row=0, column=i, padx=10, pady=10, sticky="ew")
            self.metric_cards[title] = card
    
    def create_alerts_section(self):
        """Create alerts and notifications section"""
        self.alert_panel = AlertPanel(self)
        self.alert_panel.widget.grid(row=3, column=0, columnspan=3, sticky="ew", padx=20, pady=10)
    
    def setup_event_handlers(self):
        """Set up event handlers for system events"""
        # This would connect to the event bus in a real implementation
        pass
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        current_theme = self.theme_manager.current_theme
        new_theme = "light" if current_theme == "dark" else "dark"
        self.theme_manager.set_theme(new_theme)
    
    def update_stress_level(self, level: float, risk_level: str = "low"):
        """Update the main stress indicator"""
        if self.stress_indicator:
            self.stress_indicator.update_stress_level(level, risk_level)
            
            # Update text display
            percentage = int(level * 100)
            risk_text = {
                "low": "Relaxed",
                "medium": "Moderate",
                "high": "High Stress",
                "critical": "Critical!"
            }.get(risk_level, "Unknown")
            
            self.stress_text.configure(text=f"{percentage}% - {risk_text}")
    
    def update_metric(self, metric_name: str, value: str, unit: str = None):
        """Update a metric card value"""
        if metric_name in self.metric_cards:
            self.metric_cards[metric_name].update_value(value, unit)
    
    def add_alert(self, alert_type: str, message: str, severity: str = "low"):
        """Add a new alert to the dashboard"""
        if self.alert_panel:
            self.alert_panel.add_alert(alert_type, message, severity)

class UIManager:
    """Manages the overall UI framework"""
    
    def __init__(self):
        self.root = None
        self.dashboard = None
        self.theme_manager = ThemeManager()
        self.running = False
        
    def initialize(self, title: str = "Stress Monitor V2.0", geometry: str = "1200x800"):
        """Initialize the UI framework"""
        self.root = ctk.CTk()
        self.root.title(title)
        self.root.geometry(geometry)
        
        # Configure root grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main dashboard
        self.dashboard = ModernDashboard(self.root)
        self.dashboard.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        print("✅ Modern UI Framework initialized!")
    
    def run(self):
        """Start the UI main loop"""
        if self.root:
            self.running = True
            self.root.mainloop()
    
    def stop(self):
        """Stop the UI"""
        self.running = False
        if self.root:
            self.root.quit()
    
    def get_dashboard(self) -> ModernDashboard:
        """Get the main dashboard"""
        return self.dashboard

# Global UI manager instance
ui_manager = UIManager()

def get_ui_manager() -> UIManager:
    """Get the global UI manager instance"""
    return ui_manager
