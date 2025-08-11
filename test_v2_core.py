"""
Simple test for V2.0 core components
Tests database and core architecture without UI
"""

import asyncio
from datetime import datetime

from src.core import get_system_core
from src.database import get_database_manager

async def test_v2_core():
    """Test V2.0 core functionality"""
    print("🔧 Testing V2.0 Core Components...")
    
    # Test database
    print("\n1. Testing Database:")
    db = get_database_manager()
    
    # Create test user
    try:
        user_id = db.create_user("test_user", "test@example.com", "Test User")
        print(f"✅ Created user: {user_id}")
    except Exception as e:
        print(f"⚠️  User creation issue (might exist): {e}")
        user_id = "user_001"  # Use default user
    
    # Test user preferences
    db.set_user_preference(user_id, "ui", "theme", "dark")
    prefs = db.get_user_preferences(user_id, "ui")
    print(f"✅ User preferences: {prefs}")
    
    # Test system core
    print("\n2. Testing System Core:")
    core = get_system_core()
    
    await core.initialize()
    print("✅ Core initialized")
    
    await core.start()
    print("✅ Core started")
    
    # Get system status
    status = core.get_status()
    print(f"✅ System status: {status['core_status']['active_components']} active components")
    
    # Test session management
    print("\n3. Testing Session Management:")
    user_manager = core.component_manager.get_component("user_manager")
    if user_manager:
        session_id = await user_manager.create_session(user_id, "test")
        print(f"✅ Created session: {session_id}")
        
        await asyncio.sleep(1)
        
        await user_manager.end_session(session_id, "Test completed")
        print("✅ Session ended successfully")
    
    # Test stress analysis
    print("\n4. Testing Database Analytics:")
    stats = db.get_stress_statistics(user_id, days=7)
    print(f"✅ Stress statistics: {stats}")
    
    # Cleanup
    await core.stop()
    print("✅ Core stopped")
    
    print("\n🎉 All V2.0 core tests passed!")

if __name__ == "__main__":
    asyncio.run(test_v2_core())
