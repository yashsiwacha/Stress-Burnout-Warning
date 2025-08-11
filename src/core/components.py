"""
Core Components for Stress Monitor V2.0
Implementation of key system components using the new architecture
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

from .architecture import (
    IComponent, ComponentConfig, ComponentStatus, 
    SystemEvent, EventType, get_system_core
)
from ..database import get_database_manager, StressReading

class UserManagerComponent(IComponent):
    """Manages user sessions and authentication"""
    
    def __init__(self):
        config = ComponentConfig(
            component_id="user_manager",
            name="User Manager",
            enabled=True,
            config={
                "session_timeout_minutes": 60,
                "max_concurrent_sessions": 5
            }
        )
        super().__init__(config)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.db_manager = get_database_manager()
    
    async def initialize(self) -> bool:
        """Initialize user manager"""
        try:
            # Subscribe to relevant events
            self.event_bus.subscribe(EventType.USER_LOGIN, self.handle_user_login)
            self.event_bus.subscribe(EventType.SESSION_END, self.handle_session_end)
            return True
        except Exception as e:
            print(f"❌ User Manager initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start user manager"""
        # Start session cleanup task
        asyncio.create_task(self._session_cleanup_task())
        return True
    
    async def stop(self) -> bool:
        """Stop user manager"""
        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)
        return True
    
    async def process_event(self, event: SystemEvent) -> Optional[SystemEvent]:
        """Process system events"""
        if event.event_type == EventType.STRESS_READING:
            # Update session activity
            session_id = event.session_id
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = datetime.now()
        
        return None
    
    async def handle_user_login(self, event: SystemEvent):
        """Handle user login event"""
        user_id = event.user_id
        if user_id:
            self.db_manager.update_user_login(user_id)
            print(f"👤 User {user_id} logged in")
    
    async def handle_session_end(self, event: SystemEvent):
        """Handle session end event"""
        session_id = event.session_id
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            print(f"📊 Session {session_id} ended")
    
    async def create_session(self, user_id: str, session_type: str = "monitoring") -> str:
        """Create a new user session"""
        session_id = self.db_manager.create_session(user_id, session_type)
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "session_type": session_type,
            "start_time": datetime.now(),
            "last_activity": datetime.now()
        }
        
        # Emit session start event
        event = SystemEvent(
            event_type=EventType.SESSION_START,
            source_component=self.config.component_id,
            user_id=user_id,
            session_id=session_id,
            data={"session_type": session_type}
        )
        self.emit_event(event)
        
        return session_id
    
    async def end_session(self, session_id: str, notes: str = None):
        """End a user session"""
        if session_id not in self.active_sessions:
            return
        
        session_data = self.active_sessions[session_id]
        self.db_manager.end_session(session_id, notes)
        
        # Emit session end event
        event = SystemEvent(
            event_type=EventType.SESSION_END,
            source_component=self.config.component_id,
            user_id=session_data["user_id"],
            session_id=session_id,
            data={"notes": notes}
        )
        self.emit_event(event)
        
        del self.active_sessions[session_id]
    
    async def _session_cleanup_task(self):
        """Cleanup expired sessions"""
        while self.get_status() == ComponentStatus.ACTIVE:
            try:
                timeout_minutes = self.config.config["session_timeout_minutes"]
                cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
                
                expired_sessions = []
                for session_id, session_data in self.active_sessions.items():
                    if session_data["last_activity"] < cutoff_time:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.end_session(session_id, "Session expired")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Session cleanup error: {e}")
                await asyncio.sleep(60)

class StressAnalysisComponent(IComponent):
    """Analyzes stress data and generates insights"""
    
    def __init__(self):
        config = ComponentConfig(
            component_id="stress_analyzer",
            name="Stress Analysis Engine",
            enabled=True,
            config={
                "analysis_interval_seconds": 30,
                "stress_threshold_high": 0.7,
                "stress_threshold_critical": 0.9,
                "trend_window_minutes": 10
            },
            dependencies=["user_manager"]
        )
        super().__init__(config)
        self.db_manager = get_database_manager()
        self.analysis_task = None
    
    async def initialize(self) -> bool:
        """Initialize stress analyzer"""
        try:
            # Subscribe to stress reading events
            self.event_bus.subscribe(EventType.STRESS_READING, self.handle_stress_reading)
            return True
        except Exception as e:
            print(f"❌ Stress Analyzer initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start stress analyzer"""
        # Start continuous analysis task
        self.analysis_task = asyncio.create_task(self._continuous_analysis_task())
        return True
    
    async def stop(self) -> bool:
        """Stop stress analyzer"""
        if self.analysis_task:
            self.analysis_task.cancel()
        return True
    
    async def process_event(self, event: SystemEvent) -> Optional[SystemEvent]:
        """Process system events"""
        return None
    
    async def handle_stress_reading(self, event: SystemEvent):
        """Handle new stress reading"""
        try:
            stress_level = event.data.get("overall_stress_level", 0.0)
            user_id = event.user_id
            session_id = event.session_id
            
            # Check for high stress alerts
            await self._check_stress_alerts(user_id, session_id, stress_level)
            
            # Update trend analysis
            await self._update_stress_trends(user_id, stress_level)
            
        except Exception as e:
            print(f"❌ Error handling stress reading: {e}")
    
    async def _check_stress_alerts(self, user_id: str, session_id: str, stress_level: float):
        """Check if stress level warrants an alert"""
        high_threshold = self.config.config["stress_threshold_high"]
        critical_threshold = self.config.config["stress_threshold_critical"]
        
        alert_type = None
        severity = None
        
        if stress_level >= critical_threshold:
            alert_type = "stress_critical"
            severity = "critical"
        elif stress_level >= high_threshold:
            alert_type = "stress_high"
            severity = "high"
        
        if alert_type:
            # Create alert in database
            alert_id = self.db_manager.create_alert(
                user_id=user_id,
                alert_type=alert_type,
                severity=severity,
                title=f"{'Critical' if severity == 'critical' else 'High'} Stress Detected",
                message=f"Your stress level is {stress_level:.1%}. Consider taking a break.",
                session_id=session_id,
                context_data={"stress_level": stress_level}
            )
            
            # Emit alert event
            event = SystemEvent(
                event_type=EventType.ALERT_TRIGGERED,
                source_component=self.config.component_id,
                user_id=user_id,
                session_id=session_id,
                data={
                    "alert_id": alert_id,
                    "alert_type": alert_type,
                    "severity": severity,
                    "stress_level": stress_level
                },
                priority=5 if severity == "critical" else 3
            )
            self.emit_event(event)
    
    async def _update_stress_trends(self, user_id: str, stress_level: float):
        """Update stress trend analysis"""
        # This would implement trend analysis logic
        # For now, just log the trend update
        pass
    
    async def _continuous_analysis_task(self):
        """Continuous analysis of stress patterns"""
        while self.get_status() == ComponentStatus.ACTIVE:
            try:
                # Perform periodic analysis
                await self._analyze_all_users()
                
                interval = self.config.config["analysis_interval_seconds"]
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"❌ Continuous analysis error: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_all_users(self):
        """Analyze stress patterns for all active users"""
        # This would implement comprehensive analysis
        # For now, just placeholder
        pass

class AlertManagerComponent(IComponent):
    """Manages system alerts and notifications"""
    
    def __init__(self):
        config = ComponentConfig(
            component_id="alert_manager",
            name="Alert Manager",
            enabled=True,
            config={
                "max_alerts_per_hour": 10,
                "alert_cooldown_minutes": 5
            },
            dependencies=["user_manager"]
        )
        super().__init__(config)
        self.db_manager = get_database_manager()
        self.recent_alerts: Dict[str, List[datetime]] = {}
    
    async def initialize(self) -> bool:
        """Initialize alert manager"""
        try:
            # Subscribe to alert events
            self.event_bus.subscribe(EventType.ALERT_TRIGGERED, self.handle_alert_triggered)
            return True
        except Exception as e:
            print(f"❌ Alert Manager initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start alert manager"""
        return True
    
    async def stop(self) -> bool:
        """Stop alert manager"""
        return True
    
    async def process_event(self, event: SystemEvent) -> Optional[SystemEvent]:
        """Process system events"""
        return None
    
    async def handle_alert_triggered(self, event: SystemEvent):
        """Handle triggered alerts"""
        try:
            user_id = event.user_id
            alert_type = event.data.get("alert_type")
            
            # Check rate limiting
            if self._should_suppress_alert(user_id, alert_type):
                print(f"⏭️  Alert suppressed for user {user_id}: rate limited")
                return
            
            # Record alert
            self._record_alert(user_id, alert_type)
            
            # Process the alert (send notifications, etc.)
            await self._process_alert(event)
            
        except Exception as e:
            print(f"❌ Error handling alert: {e}")
    
    def _should_suppress_alert(self, user_id: str, alert_type: str) -> bool:
        """Check if alert should be suppressed due to rate limiting"""
        key = f"{user_id}:{alert_type}"
        cooldown_minutes = self.config.config["alert_cooldown_minutes"]
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        if key not in self.recent_alerts:
            return False
        
        # Check if we have recent alerts within cooldown period
        recent = [t for t in self.recent_alerts[key] if t > cutoff_time]
        return len(recent) > 0
    
    def _record_alert(self, user_id: str, alert_type: str):
        """Record alert for rate limiting"""
        key = f"{user_id}:{alert_type}"
        if key not in self.recent_alerts:
            self.recent_alerts[key] = []
        
        self.recent_alerts[key].append(datetime.now())
        
        # Clean up old entries
        max_alerts = self.config.config["max_alerts_per_hour"]
        if len(self.recent_alerts[key]) > max_alerts:
            self.recent_alerts[key] = self.recent_alerts[key][-max_alerts:]
    
    async def _process_alert(self, event: SystemEvent):
        """Process and route the alert"""
        alert_data = event.data
        severity = alert_data.get("severity", "low")
        
        print(f"🚨 Alert processed: {alert_data.get('alert_type')} "
              f"(severity: {severity}) for user {event.user_id}")
        
        # Here you would implement actual notification logic
        # (desktop notifications, email, SMS, etc.)

# Initialize all core components
def create_core_components() -> List[IComponent]:
    """Create and return all core components"""
    return [
        UserManagerComponent(),
        StressAnalysisComponent(),
        AlertManagerComponent()
    ]
