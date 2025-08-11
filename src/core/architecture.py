"""
Core Architecture for Stress & Burnout Warning System V2.0
Modular, extensible architecture with plugin support and async processing
"""

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from enum import Enum

class ComponentStatus(Enum):
    """Component status enumeration"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    STOPPING = "stopping"

class EventType(Enum):
    """System event types"""
    STRESS_READING = "stress_reading"
    USER_LOGIN = "user_login"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ALERT_TRIGGERED = "alert_triggered"
    WELLNESS_ACTIVITY = "wellness_activity"
    SYSTEM_ERROR = "system_error"

@dataclass
class SystemEvent:
    """System event data structure"""
    event_id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:8]}")
    event_type: EventType = EventType.STRESS_READING
    timestamp: datetime = field(default_factory=datetime.now)
    source_component: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=critical

@dataclass
class ComponentConfig:
    """Configuration for system components"""
    component_id: str
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

class IComponent(ABC):
    """Interface for all system components"""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.status = ComponentStatus.INACTIVE
        self.event_bus = None
        self._lock = threading.RLock()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component"""
        pass
    
    @abstractmethod
    async def process_event(self, event: SystemEvent) -> Optional[SystemEvent]:
        """Process a system event"""
        pass
    
    def get_status(self) -> ComponentStatus:
        """Get component status"""
        with self._lock:
            return self.status
    
    def set_status(self, status: ComponentStatus):
        """Set component status"""
        with self._lock:
            self.status = status
    
    def emit_event(self, event: SystemEvent):
        """Emit an event to the system"""
        if self.event_bus:
            self.event_bus.emit(event)

class EventBus:
    """Central event bus for component communication"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(handler)
    
    async def emit(self, event: SystemEvent):
        """Emit an event to all subscribers"""
        await self.event_queue.put(event)
    
    async def start_processing(self):
        """Start processing events"""
        self.running = True
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    async def _process_event(self, event: SystemEvent):
        """Process a single event"""
        with self._lock:
            handlers = self.subscribers.get(event.event_type, [])
        
        # Process handlers concurrently
        if handlers:
            tasks = [handler(event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop_processing(self):
        """Stop event processing"""
        self.running = False

class ComponentManager:
    """Manages all system components"""
    
    def __init__(self):
        self.components: Dict[str, IComponent] = {}
        self.event_bus = EventBus()
        self.dependency_graph: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def register_component(self, component: IComponent):
        """Register a component with the manager"""
        with self._lock:
            component_id = component.config.component_id
            self.components[component_id] = component
            component.event_bus = self.event_bus
            
            # Build dependency graph
            self.dependency_graph[component_id] = component.config.dependencies
            
            print(f"✅ Registered component: {component.config.name}")
    
    def get_component(self, component_id: str) -> Optional[IComponent]:
        """Get a component by ID"""
        return self.components.get(component_id)
    
    async def initialize_all(self) -> bool:
        """Initialize all components in dependency order"""
        initialization_order = self._get_initialization_order()
        
        for component_id in initialization_order:
            component = self.components[component_id]
            if not component.config.enabled:
                print(f"⏭️  Skipping disabled component: {component.config.name}")
                continue
            
            try:
                component.set_status(ComponentStatus.INITIALIZING)
                success = await component.initialize()
                
                if success:
                    component.set_status(ComponentStatus.ACTIVE)
                    print(f"✅ Initialized: {component.config.name}")
                else:
                    component.set_status(ComponentStatus.ERROR)
                    print(f"❌ Failed to initialize: {component.config.name}")
                    return False
                    
            except Exception as e:
                component.set_status(ComponentStatus.ERROR)
                print(f"❌ Error initializing {component.config.name}: {e}")
                return False
        
        return True
    
    async def start_all(self) -> bool:
        """Start all components"""
        # Start event bus
        event_task = asyncio.create_task(self.event_bus.start_processing())
        
        # Start components
        for component in self.components.values():
            if component.config.enabled and component.get_status() == ComponentStatus.ACTIVE:
                try:
                    await component.start()
                    print(f"🚀 Started: {component.config.name}")
                except Exception as e:
                    print(f"❌ Error starting {component.config.name}: {e}")
                    return False
        
        print("🎉 All components started successfully!")
        return True
    
    async def stop_all(self):
        """Stop all components"""
        # Stop components in reverse order
        initialization_order = self._get_initialization_order()
        
        for component_id in reversed(initialization_order):
            component = self.components[component_id]
            if component.get_status() == ComponentStatus.ACTIVE:
                try:
                    component.set_status(ComponentStatus.STOPPING)
                    await component.stop()
                    component.set_status(ComponentStatus.INACTIVE)
                    print(f"🛑 Stopped: {component.config.name}")
                except Exception as e:
                    print(f"❌ Error stopping {component.config.name}: {e}")
        
        # Stop event bus
        self.event_bus.stop_processing()
        print("🏁 All components stopped")
    
    def _get_initialization_order(self) -> List[str]:
        """Get component initialization order based on dependencies"""
        visited = set()
        order = []
        
        def visit(component_id: str):
            if component_id in visited:
                return
            
            visited.add(component_id)
            
            # Visit dependencies first
            for dep_id in self.dependency_graph.get(component_id, []):
                if dep_id in self.components:
                    visit(dep_id)
            
            order.append(component_id)
        
        for component_id in self.components:
            visit(component_id)
        
        return order
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self.components),
            "active_components": 0,
            "inactive_components": 0,
            "error_components": 0,
            "components": {}
        }
        
        for component_id, component in self.components.items():
            comp_status = component.get_status()
            status["components"][component_id] = {
                "name": component.config.name,
                "status": comp_status.value,
                "enabled": component.config.enabled
            }
            
            if comp_status == ComponentStatus.ACTIVE:
                status["active_components"] += 1
            elif comp_status == ComponentStatus.INACTIVE:
                status["inactive_components"] += 1
            elif comp_status == ComponentStatus.ERROR:
                status["error_components"] += 1
        
        return status

class SystemCore:
    """Main system core that orchestrates everything"""
    
    def __init__(self):
        self.component_manager = ComponentManager()
        self.config = {}
        self.running = False
        
    async def initialize(self, config_path: str = None):
        """Initialize the system"""
        print("🔧 Initializing Stress Monitor V2.0 Core...")
        
        # Load configuration
        if config_path:
            await self._load_config(config_path)
        
        # Initialize components
        success = await self.component_manager.initialize_all()
        if not success:
            raise RuntimeError("Failed to initialize system components")
        
        print("✅ System core initialized successfully!")
    
    async def start(self):
        """Start the system"""
        print("🚀 Starting Stress Monitor V2.0...")
        
        success = await self.component_manager.start_all()
        if not success:
            raise RuntimeError("Failed to start system components")
        
        self.running = True
        print("🎉 System started successfully!")
    
    async def stop(self):
        """Stop the system"""
        print("🛑 Stopping Stress Monitor V2.0...")
        
        self.running = False
        await self.component_manager.stop_all()
        
        print("🏁 System stopped successfully!")
    
    async def _load_config(self, config_path: str):
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"📋 Configuration loaded from {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to load config from {config_path}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "system_running": self.running,
            "core_status": self.component_manager.get_system_status()
        }
    
    def register_component(self, component: IComponent):
        """Register a component with the system"""
        self.component_manager.register_component(component)
    
    def emit_event(self, event: SystemEvent):
        """Emit a system event"""
        asyncio.create_task(self.component_manager.event_bus.emit(event))

# Global system core instance
system_core = SystemCore()

def get_system_core() -> SystemCore:
    """Get the global system core instance"""
    return system_core
