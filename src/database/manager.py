"""
Database Manager for Stress & Burnout Warning System V2.0
Handles all database operations with connection pooling and transaction management
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import threading
from dataclasses import dataclass

@dataclass
class StressReading:
    """Data class for stress readings"""
    reading_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    overall_stress_level: float
    confidence_score: float
    facial_stress_component: Optional[float] = None
    voice_stress_component: Optional[float] = None
    behavioral_stress_component: Optional[float] = None
    risk_level: str = "low"
    contributing_factors: List[str] = None
    recommendations: List[str] = None
    raw_data: Dict[str, Any] = None

@dataclass
class UserSession:
    """Data class for user sessions"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    session_type: str = "monitoring"
    status: str = "active"
    device_info: Dict[str, Any] = None
    environment_info: Dict[str, Any] = None
    notes: Optional[str] = None

class DatabaseManager:
    """Advanced database manager with connection pooling and transaction support"""
    
    def __init__(self, db_path="data/stress_monitor_v2.db", pool_size=5):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Ensure database exists
        if not self.db_path.exists():
            from .schema import DatabaseSchema
            schema_manager = DatabaseSchema(str(self.db_path))
            schema_manager.create_schema()
            schema_manager.create_sample_data()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager"""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
    
    # User Management
    def create_user(self, username: str, email: str = None, full_name: str = None) -> str:
        """Create a new user and return user_id"""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (user_id, username, email, full_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, email, full_name, datetime.now()))
        
        return user_id
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by user_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_user_login(self, user_id: str):
        """Update user's last login timestamp"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE users SET last_login = ? WHERE user_id = ?
            """, (datetime.now(), user_id))
    
    # Session Management
    def create_session(self, user_id: str, session_type: str = "monitoring", 
                      device_info: Dict[str, Any] = None) -> str:
        """Create a new monitoring session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO sessions (session_id, user_id, start_time, session_type, 
                                status, device_info)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, user_id, datetime.now(), session_type,
                "active", json.dumps(device_info) if device_info else None
            ))
        
        return session_id
    
    def end_session(self, session_id: str, notes: str = None):
        """End a monitoring session"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Get session start time
            cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Session {session_id} not found")
            
            start_time = datetime.fromisoformat(row['start_time'])
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds() / 60)
            
            cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, duration_minutes = ?, status = ?, notes = ?
            WHERE session_id = ?
            """, (end_time, duration, "completed", notes, session_id))
    
    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM sessions 
            WHERE user_id = ? AND status = 'active'
            ORDER BY start_time DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Stress Data Management
    def save_stress_reading(self, reading: StressReading) -> str:
        """Save a stress reading to the database"""
        if not reading.reading_id:
            reading.reading_id = f"reading_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO stress_readings (
                reading_id, session_id, user_id, timestamp, overall_stress_level,
                confidence_score, facial_stress_component, voice_stress_component,
                behavioral_stress_component, risk_level, contributing_factors,
                recommendations, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading.reading_id, reading.session_id, reading.user_id,
                reading.timestamp, reading.overall_stress_level, reading.confidence_score,
                reading.facial_stress_component, reading.voice_stress_component,
                reading.behavioral_stress_component, reading.risk_level,
                json.dumps(reading.contributing_factors or []),
                json.dumps(reading.recommendations or []),
                json.dumps(reading.raw_data or {})
            ))
        
        return reading.reading_id
    
    def get_stress_readings(self, user_id: str, start_date: datetime = None, 
                           end_date: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get stress readings for a user within a date range"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM stress_readings WHERE user_id = ?"
            params = [user_id]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stress_statistics(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get stress statistics for a user over the last N days"""
        start_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_readings,
                AVG(overall_stress_level) as avg_stress,
                MIN(overall_stress_level) as min_stress,
                MAX(overall_stress_level) as max_stress,
                AVG(confidence_score) as avg_confidence,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) as critical_risk_count
            FROM stress_readings 
            WHERE user_id = ? AND timestamp >= ?
            """, (user_id, start_date))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    # Facial Data Management
    def save_facial_data(self, reading_id: str, facial_data: Dict[str, Any]) -> str:
        """Save facial analysis data"""
        facial_id = f"facial_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO facial_data (
                facial_id, reading_id, timestamp, detected_emotions,
                facial_landmarks, eye_gaze_data, blink_rate, head_pose,
                micro_expressions, stress_indicators, quality_score,
                processing_time_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                facial_id, reading_id, datetime.now(),
                json.dumps(facial_data.get('emotions', {})),
                json.dumps(facial_data.get('landmarks', {})),
                json.dumps(facial_data.get('gaze_data', {})),
                facial_data.get('blink_rate'),
                json.dumps(facial_data.get('head_pose', {})),
                json.dumps(facial_data.get('micro_expressions', {})),
                json.dumps(facial_data.get('stress_indicators', {})),
                facial_data.get('quality_score', 0.0),
                facial_data.get('processing_time_ms', 0),
                facial_data.get('model_version', 'v1.0')
            ))
        
        return facial_id
    
    # Voice Data Management
    def save_voice_data(self, reading_id: str, voice_data: Dict[str, Any]) -> str:
        """Save voice analysis data"""
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO voice_data (
                voice_id, reading_id, timestamp, fundamental_frequency,
                pitch_variation, speech_rate, volume_level, spectral_features,
                prosodic_features, voice_quality_metrics, stress_indicators,
                emotion_classification, audio_quality_score, processing_time_ms,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                voice_id, reading_id, datetime.now(),
                voice_data.get('fundamental_frequency'),
                voice_data.get('pitch_variation'),
                voice_data.get('speech_rate'),
                voice_data.get('volume_level'),
                json.dumps(voice_data.get('spectral_features', {})),
                json.dumps(voice_data.get('prosodic_features', {})),
                json.dumps(voice_data.get('quality_metrics', {})),
                json.dumps(voice_data.get('stress_indicators', {})),
                json.dumps(voice_data.get('emotion_classification', {})),
                voice_data.get('audio_quality_score', 0.0),
                voice_data.get('processing_time_ms', 0),
                voice_data.get('model_version', 'v1.0')
            ))
        
        return voice_id
    
    # Alert Management
    def create_alert(self, user_id: str, alert_type: str, severity: str, 
                    title: str, message: str, session_id: str = None,
                    context_data: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO alerts (
                alert_id, user_id, session_id, timestamp, alert_type,
                severity, title, message, context_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id, user_id, session_id, datetime.now(),
                alert_type, severity, title, message,
                json.dumps(context_data) if context_data else None
            ))
        
        return alert_id
    
    def get_active_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active alerts for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM alerts 
            WHERE user_id = ? AND dismissed_at IS NULL
            ORDER BY timestamp DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def dismiss_alert(self, alert_id: str, effectiveness_rating: int = None):
        """Dismiss an alert with optional effectiveness rating"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE alerts 
            SET dismissed_at = ?, effectiveness_rating = ?
            WHERE alert_id = ?
            """, (datetime.now(), effectiveness_rating, alert_id))
    
    # User Preferences
    def set_user_preference(self, user_id: str, category: str, 
                           setting_key: str, setting_value: str):
        """Set a user preference"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO user_preferences 
            (preference_id, user_id, category, setting_key, setting_value, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"pref_{uuid.uuid4().hex[:8]}", user_id, category,
                setting_key, setting_value, datetime.now()
            ))
    
    def get_user_preferences(self, user_id: str, category: str = None) -> Dict[str, Any]:
        """Get user preferences by category"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if category:
                cursor.execute("""
                SELECT setting_key, setting_value FROM user_preferences 
                WHERE user_id = ? AND category = ?
                """, (user_id, category))
            else:
                cursor.execute("""
                SELECT category, setting_key, setting_value FROM user_preferences 
                WHERE user_id = ?
                """, (user_id,))
            
            if category:
                return {row['setting_key']: row['setting_value'] for row in cursor.fetchall()}
            else:
                preferences = {}
                for row in cursor.fetchall():
                    if row['category'] not in preferences:
                        preferences[row['category']] = {}
                    preferences[row['category']][row['setting_key']] = row['setting_value']
                return preferences
    
    # Analytics and Insights
    def save_analytics(self, user_id: str, analysis_date: datetime, 
                      analysis_type: str, analytics_data: Dict[str, Any]):
        """Save analytics and insights data"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO analytics (
                analytics_id, user_id, analysis_date, analysis_type,
                stress_summary, pattern_analysis, trend_analysis,
                insights, recommendations, wellness_score,
                improvement_areas, achievements
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"analytics_{uuid.uuid4().hex[:8]}", user_id, analysis_date, analysis_type,
                json.dumps(analytics_data.get('stress_summary', {})),
                json.dumps(analytics_data.get('pattern_analysis', {})),
                json.dumps(analytics_data.get('trend_analysis', {})),
                json.dumps(analytics_data.get('insights', [])),
                json.dumps(analytics_data.get('recommendations', [])),
                analytics_data.get('wellness_score', 0.0),
                json.dumps(analytics_data.get('improvement_areas', [])),
                json.dumps(analytics_data.get('achievements', []))
            ))
    
    def get_analytics(self, user_id: str, analysis_type: str = None, 
                     days: int = 30) -> List[Dict[str, Any]]:
        """Get analytics data for a user"""
        start_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM analytics WHERE user_id = ? AND analysis_date >= ?"
            params = [user_id, start_date.date()]
            
            if analysis_type:
                query += " AND analysis_type = ?"
                params.append(analysis_type)
            
            query += " ORDER BY analysis_date DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # Wellness Activities
    def start_wellness_activity(self, user_id: str, activity_type: str, 
                               activity_name: str, stress_before: float = None) -> str:
        """Start a wellness activity"""
        activity_id = f"wellness_{uuid.uuid4().hex[:8]}"
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO wellness_activities (
                activity_id, user_id, activity_type, activity_name,
                start_time, stress_before, completion_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                activity_id, user_id, activity_type, activity_name,
                datetime.now(), stress_before, "in_progress"
            ))
        
        return activity_id
    
    def complete_wellness_activity(self, activity_id: str, effectiveness_rating: int = None,
                                  stress_after: float = None, notes: str = None):
        """Complete a wellness activity"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Get start time to calculate duration
            cursor.execute("SELECT start_time, stress_before FROM wellness_activities WHERE activity_id = ?", 
                          (activity_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Activity {activity_id} not found")
            
            start_time = datetime.fromisoformat(row['start_time'])
            stress_before = row['stress_before']
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds() / 60)
            
            improvement_score = None
            if stress_before is not None and stress_after is not None:
                improvement_score = max(0, stress_before - stress_after)
            
            cursor.execute("""
            UPDATE wellness_activities 
            SET end_time = ?, duration_minutes = ?, completion_status = ?,
                effectiveness_rating = ?, stress_after = ?, improvement_score = ?,
                notes = ?
            WHERE activity_id = ?
            """, (
                end_time, duration, "completed", effectiveness_rating,
                stress_after, improvement_score, notes, activity_id
            ))
    
    def get_wellness_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get wellness activity history"""
        start_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM wellness_activities 
            WHERE user_id = ? AND start_time >= ?
            ORDER BY start_time DESC
            """, (user_id, start_date))
            return [dict(row) for row in cursor.fetchall()]

# Global database manager instance
db_manager = DatabaseManager()

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager
