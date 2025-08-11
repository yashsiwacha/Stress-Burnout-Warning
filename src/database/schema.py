"""
Database Schema for Stress & Burnout Warning System V2.0
Comprehensive data model for user management, stress monitoring, and analytics
"""

import sqlite3
from datetime import datetime
from pathlib import Path
import json

class DatabaseSchema:
    """Database schema manager for V2.0"""
    
    def __init__(self, db_path="data/stress_monitor_v2.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
    def create_schema(self):
        """Create all database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create all tables
        self._create_users_table(cursor)
        self._create_sessions_table(cursor)
        self._create_stress_readings_table(cursor)
        self._create_facial_data_table(cursor)
        self._create_voice_data_table(cursor)
        self._create_predictions_table(cursor)
        self._create_alerts_table(cursor)
        self._create_wellness_activities_table(cursor)
        self._create_user_preferences_table(cursor)
        self._create_analytics_table(cursor)
        
        # Create indexes for better performance
        self._create_indexes(cursor)
        
        conn.commit()
        conn.close()
        
        print("✅ Database schema created successfully!")
    
    def _create_users_table(self, cursor):
        """Create users table for multi-user support"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            profile_settings TEXT,  -- JSON blob
            privacy_settings TEXT   -- JSON blob
        )
        """)
    
    def _create_sessions_table(self, cursor):
        """Create monitoring sessions table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_minutes INTEGER,
            session_type TEXT DEFAULT 'monitoring',  -- monitoring, wellness, demo
            status TEXT DEFAULT 'active',  -- active, completed, paused, terminated
            device_info TEXT,  -- JSON blob with device details
            environment_info TEXT,  -- JSON blob with environment context
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
    
    def _create_stress_readings_table(self, cursor):
        """Create main stress readings table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stress_readings (
            reading_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            overall_stress_level REAL NOT NULL,
            confidence_score REAL NOT NULL,
            facial_stress_component REAL,
            voice_stress_component REAL,
            behavioral_stress_component REAL,
            risk_level TEXT,  -- low, medium, high, critical
            contributing_factors TEXT,  -- JSON array
            recommendations TEXT,  -- JSON array
            raw_data TEXT,  -- JSON blob with all raw sensor data
            FOREIGN KEY (session_id) REFERENCES sessions(session_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
    
    def _create_facial_data_table(self, cursor):
        """Create facial analysis data table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS facial_data (
            facial_id TEXT PRIMARY KEY,
            reading_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            detected_emotions TEXT,  -- JSON with emotion probabilities
            facial_landmarks TEXT,  -- JSON with landmark coordinates
            eye_gaze_data TEXT,      -- JSON with gaze tracking
            blink_rate REAL,
            head_pose TEXT,          -- JSON with head orientation
            micro_expressions TEXT,  -- JSON with micro-expression data
            stress_indicators TEXT,  -- JSON with specific stress markers
            quality_score REAL,      -- Image quality for reliability
            processing_time_ms INTEGER,
            model_version TEXT,
            FOREIGN KEY (reading_id) REFERENCES stress_readings(reading_id)
        )
        """)
    
    def _create_voice_data_table(self, cursor):
        """Create voice analysis data table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_data (
            voice_id TEXT PRIMARY KEY,
            reading_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            fundamental_frequency REAL,
            pitch_variation REAL,
            speech_rate REAL,
            volume_level REAL,
            spectral_features TEXT,  -- JSON with spectral analysis
            prosodic_features TEXT,  -- JSON with prosodic features
            voice_quality_metrics TEXT,  -- JSON with voice quality data
            stress_indicators TEXT,  -- JSON with voice stress markers
            emotion_classification TEXT,  -- JSON with emotion probabilities
            audio_quality_score REAL,
            processing_time_ms INTEGER,
            model_version TEXT,
            FOREIGN KEY (reading_id) REFERENCES stress_readings(reading_id)
        )
        """)
    
    def _create_predictions_table(self, cursor):
        """Create stress predictions table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            prediction_type TEXT NOT NULL,  -- short_term, daily, weekly
            predicted_stress_level REAL,
            prediction_confidence REAL,
            time_horizon_minutes INTEGER,
            contributing_patterns TEXT,  -- JSON with pattern analysis
            recommended_actions TEXT,    -- JSON with action recommendations
            actual_outcome REAL,         -- For model validation
            accuracy_score REAL,
            model_version TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
    
    def _create_alerts_table(self, cursor):
        """Create alerts and notifications table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            timestamp TIMESTAMP NOT NULL,
            alert_type TEXT NOT NULL,  -- stress_high, break_reminder, wellness_tip
            severity TEXT NOT NULL,    -- low, medium, high, critical
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            action_required BOOLEAN DEFAULT 0,
            action_taken TEXT,
            dismissed_at TIMESTAMP,
            effectiveness_rating INTEGER,  -- 1-5 user feedback
            context_data TEXT,  -- JSON with context information
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
        """)
    
    def _create_wellness_activities_table(self, cursor):
        """Create wellness activities and interventions table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS wellness_activities (
            activity_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            activity_type TEXT NOT NULL,  -- breathing, meditation, break, exercise
            activity_name TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_minutes INTEGER,
            completion_status TEXT,  -- completed, partial, skipped
            effectiveness_rating INTEGER,  -- 1-5 user feedback
            stress_before REAL,
            stress_after REAL,
            improvement_score REAL,
            activity_data TEXT,  -- JSON with activity-specific data
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
    
    def _create_user_preferences_table(self, cursor):
        """Create user preferences and settings table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            preference_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,  -- ui, notifications, privacy, monitoring
            setting_key TEXT NOT NULL,
            setting_value TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, category, setting_key)
        )
        """)
    
    def _create_analytics_table(self, cursor):
        """Create analytics and insights table"""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            analytics_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            analysis_date DATE NOT NULL,
            analysis_type TEXT NOT NULL,  -- daily, weekly, monthly
            stress_summary TEXT,      -- JSON with stress statistics
            pattern_analysis TEXT,    -- JSON with detected patterns
            trend_analysis TEXT,      -- JSON with trend data
            insights TEXT,            -- JSON with AI-generated insights
            recommendations TEXT,     -- JSON with personalized recommendations
            wellness_score REAL,
            improvement_areas TEXT,   -- JSON with areas for improvement
            achievements TEXT,        -- JSON with user achievements
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, analysis_date, analysis_type)
        )
        """)
    
    def _create_indexes(self, cursor):
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON sessions(user_id, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_stress_readings_session_time ON stress_readings(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_stress_readings_user_time ON stress_readings(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_facial_data_reading ON facial_data(reading_id)",
            "CREATE INDEX IF NOT EXISTS idx_voice_data_reading ON voice_data(reading_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_user_time ON predictions(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_user_time ON alerts(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_wellness_user_time ON wellness_activities(user_id, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_user_date ON analytics(user_id, analysis_date)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def create_sample_data(self):
        """Create sample data for testing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sample user
        cursor.execute("""
        INSERT OR IGNORE INTO users (user_id, username, email, full_name, profile_settings, privacy_settings)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "user_001",
            "demo_user",
            "demo@stressmonitor.com",
            "Demo User",
            json.dumps({"theme": "dark", "language": "en"}),
            json.dumps({"data_retention_days": 30, "share_analytics": False})
        ))
        
        # Sample session
        session_time = datetime.now()
        cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id, user_id, start_time, session_type, status, device_info)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "session_001",
            "user_001",
            session_time,
            "monitoring",
            "active",
            json.dumps({"camera": "Built-in", "microphone": "Built-in", "os": "macOS"})
        ))
        
        # Sample preferences
        preferences = [
            ("pref_001", "user_001", "ui", "theme", "dark"),
            ("pref_002", "user_001", "notifications", "stress_alerts", "enabled"),
            ("pref_003", "user_001", "monitoring", "facial_enabled", "true"),
            ("pref_004", "user_001", "monitoring", "voice_enabled", "true"),
            ("pref_005", "user_001", "privacy", "data_retention", "30")
        ]
        
        cursor.executemany("""
        INSERT OR IGNORE INTO user_preferences (preference_id, user_id, category, setting_key, setting_value)
        VALUES (?, ?, ?, ?, ?)
        """, preferences)
        
        conn.commit()
        conn.close()
        
        print("✅ Sample data created successfully!")

def main():
    """Initialize the database schema"""
    print("🗄️ Initializing Database Schema for V2.0...")
    
    db_manager = DatabaseSchema()
    db_manager.create_schema()
    db_manager.create_sample_data()
    
    print(f"📊 Database created at: {db_manager.db_path}")
    print("🚀 Ready for V2.0 development!")

if __name__ == "__main__":
    main()
