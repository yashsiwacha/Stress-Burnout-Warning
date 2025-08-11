"""
Demo script for Stress Monitor V2.0 Architecture
Tests the new modular system with database integration and modern UI
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
import random

# Import V2.0 components
from src.core import get_system_core, SystemEvent, EventType
from src.database import get_database_manager, StressReading
from src.ui.modern_framework import get_ui_manager

class V2Demo:
    """Demonstration of V2.0 capabilities"""
    
    def __init__(self):
        self.system_core = get_system_core()
        self.db_manager = get_database_manager()
        self.ui_manager = get_ui_manager()
        self.demo_user_id = "demo_user_v2"
        self.demo_session_id = None
        self.running = False
        
    async def initialize_demo(self):
        """Initialize the demo environment"""
        print("🚀 Initializing Stress Monitor V2.0 Demo...")
        
        # Initialize system core
        await self.system_core.initialize()
        await self.system_core.start()
        
        # Create demo user if not exists
        existing_user = self.db_manager.get_user(self.demo_user_id)
        if not existing_user:
            try:
                self.demo_user_id = self.db_manager.create_user(
                    username="demo_user_v2",
                    email="demo_v2@stressmonitor.com",
                    full_name="V2.0 Demo User"
                )
            except Exception as e:
                print(f"User might already exist, using existing: {e}")
                # Use existing demo user
                pass
        
        # Create demo session
        user_manager = self.system_core.component_manager.get_component("user_manager")
        if user_manager:
            self.demo_session_id = await user_manager.create_session(
                self.demo_user_id, "demo"
            )
        
        print("✅ Demo environment initialized!")
        
    def start_ui_demo(self):
        """Start the UI demonstration in a separate thread"""
        def ui_thread():
            self.ui_manager.initialize("Stress Monitor V2.0 - Demo", "1400x900")
            self.ui_manager.run()
        
        ui_thread = threading.Thread(target=ui_thread, daemon=True)
        ui_thread.start()
        
        # Give UI time to initialize
        time.sleep(2)
        
    async def simulate_stress_monitoring(self):
        """Simulate stress monitoring with realistic data"""
        print("📊 Starting stress monitoring simulation...")
        
        dashboard = self.ui_manager.get_dashboard()
        session_start = datetime.now()
        
        while self.running:
            try:
                # Generate realistic stress data
                current_time = datetime.now()
                session_minutes = (current_time - session_start).total_seconds() / 60
                
                # Simulate stress pattern (increases over time, then drops)
                base_stress = 0.3 + 0.4 * (session_minutes / 30)  # Gradual increase
                noise = random.uniform(-0.1, 0.1)  # Random variation
                stress_level = max(0.0, min(1.0, base_stress + noise))
                
                # Determine risk level
                if stress_level >= 0.9:
                    risk_level = "critical"
                elif stress_level >= 0.7:
                    risk_level = "high"
                elif stress_level >= 0.5:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                
                # Create stress reading
                reading = StressReading(
                    reading_id="",
                    session_id=self.demo_session_id,
                    user_id=self.demo_user_id,
                    timestamp=current_time,
                    overall_stress_level=stress_level,
                    confidence_score=random.uniform(0.8, 0.95),
                    facial_stress_component=stress_level + random.uniform(-0.1, 0.1),
                    voice_stress_component=stress_level + random.uniform(-0.1, 0.1),
                    behavioral_stress_component=stress_level + random.uniform(-0.05, 0.05),
                    risk_level=risk_level,
                    contributing_factors=["Extended screen time", "High workload"],
                    recommendations=["Take a 5-minute break", "Practice deep breathing"]
                )
                
                # Save to database
                reading_id = self.db_manager.save_stress_reading(reading)
                
                # Emit system event
                event = SystemEvent(
                    event_type=EventType.STRESS_READING,
                    source_component="demo_monitor",
                    user_id=self.demo_user_id,
                    session_id=self.demo_session_id,
                    data={
                        "reading_id": reading_id,
                        "overall_stress_level": stress_level,
                        "risk_level": risk_level,
                        "confidence_score": reading.confidence_score
                    }
                )
                self.system_core.emit_event(event)
                
                # Update UI if available
                if dashboard:
                    dashboard.update_stress_level(stress_level, risk_level)
                    dashboard.update_metric("Session Time", f"{int(session_minutes)}", "min")
                    dashboard.update_metric("Heart Rate", f"{random.randint(60, 100)}", "BPM")
                    
                    # Add alerts for high stress
                    if stress_level >= 0.8 and random.random() < 0.3:
                        severity = "critical" if stress_level >= 0.9 else "high"
                        dashboard.add_alert(
                            "stress_alert",
                            f"High stress detected: {stress_level:.1%}. Consider taking a break.",
                            severity
                        )
                
                print(f"📈 Stress Level: {stress_level:.2%} ({risk_level}) - Session: {session_minutes:.1f}min")
                
                await asyncio.sleep(3)  # Update every 3 seconds for demo
                
            except Exception as e:
                print(f"❌ Error in stress monitoring: {e}")
                await asyncio.sleep(5)
    
    async def demonstrate_features(self):
        """Demonstrate various V2.0 features"""
        print("\n🎯 Demonstrating V2.0 Features:")
        
        # 1. Database functionality
        print("\n1. 📊 Database Statistics:")
        stats = self.db_manager.get_stress_statistics(self.demo_user_id, days=1)
        print(f"   Total readings: {stats.get('total_readings', 0)}")
        print(f"   Average stress: {stats.get('avg_stress', 0):.2%}")
        print(f"   High risk events: {stats.get('high_risk_count', 0)}")
        
        # 2. System status
        print("\n2. 🔧 System Status:")
        status = self.system_core.get_status()
        print(f"   System running: {status['system_running']}")
        print(f"   Active components: {status['core_status']['active_components']}")
        
        # 3. User preferences
        print("\n3. ⚙️  User Preferences:")
        self.db_manager.set_user_preference(self.demo_user_id, "ui", "theme", "dark")
        self.db_manager.set_user_preference(self.demo_user_id, "monitoring", "facial_enabled", "true")
        preferences = self.db_manager.get_user_preferences(self.demo_user_id)
        print(f"   Preferences set: {preferences}")
        
        # 4. Alert system
        print("\n4. 🚨 Alert System:")
        alert_id = self.db_manager.create_alert(
            user_id=self.demo_user_id,
            alert_type="demo_alert",
            severity="medium",
            title="Demo Alert",
            message="This is a demonstration alert showing V2.0 capabilities"
        )
        print(f"   Demo alert created: {alert_id}")
        
        # 5. Wellness activity
        print("\n5. 🧘 Wellness Features:")
        activity_id = self.db_manager.start_wellness_activity(
            user_id=self.demo_user_id,
            activity_type="breathing",
            activity_name="5-Minute Breathing Exercise",
            stress_before=0.75
        )
        
        # Simulate activity completion
        await asyncio.sleep(2)
        self.db_manager.complete_wellness_activity(
            activity_id=activity_id,
            effectiveness_rating=4,
            stress_after=0.45,
            notes="Demo breathing exercise completed successfully"
        )
        print(f"   Wellness activity completed: {activity_id}")
    
    async def run_demo(self, duration_minutes: int = 5):
        """Run the complete demo"""
        try:
            await self.initialize_demo()
            
            print(f"\n🎮 Starting {duration_minutes}-minute demo...")
            print("Press Ctrl+C to stop the demo early\n")
            
            # Start UI in background
            self.start_ui_demo()
            
            # Start monitoring simulation
            self.running = True
            
            # Run demo tasks concurrently
            demo_tasks = [
                self.simulate_stress_monitoring(),
                self.demonstrate_features()
            ]
            
            # Run for specified duration
            await asyncio.wait_for(
                asyncio.gather(*demo_tasks, return_exceptions=True),
                timeout=duration_minutes * 60
            )
            
        except asyncio.TimeoutError:
            print(f"\n⏰ Demo completed after {duration_minutes} minutes")
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            await self.cleanup_demo()
    
    async def cleanup_demo(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo...")
        
        self.running = False
        
        # End demo session
        if self.demo_session_id:
            user_manager = self.system_core.component_manager.get_component("user_manager")
            if user_manager:
                await user_manager.end_session(self.demo_session_id, "Demo completed")
        
        # Stop system
        await self.system_core.stop()
        
        # Stop UI
        self.ui_manager.stop()
        
        print("✅ Demo cleanup completed!")

async def main():
    """Main demo function"""
    print("🧠 Stress & Burnout Warning System V2.0 Demo")
    print("=" * 50)
    
    demo = V2Demo()
    await demo.run_demo(duration_minutes=3)  # 3-minute demo
    
    print("\n🎉 Thank you for trying Stress Monitor V2.0!")
    print("📈 Check the generated database at: data/stress_monitor_v2.db")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
