"""
Database package initialization for V2.0
Provides easy access to database functionality
"""

from .schema import DatabaseSchema
from .manager import DatabaseManager, get_database_manager, StressReading, UserSession

__all__ = [
    'DatabaseSchema',
    'DatabaseManager', 
    'get_database_manager',
    'StressReading',
    'UserSession'
]

# Initialize database on import
def initialize_database():
    """Initialize database if it doesn't exist"""
    try:
        db_manager = get_database_manager()
        print("✅ Database V2.0 initialized successfully!")
        return db_manager
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

# Auto-initialize when package is imported
_db_manager = initialize_database()
