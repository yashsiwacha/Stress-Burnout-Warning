"""
Core package initialization for V2.0
Provides the main system architecture and components
"""

from .architecture import (
    SystemCore, ComponentManager, EventBus, IComponent,
    ComponentConfig, ComponentStatus, SystemEvent, EventType,
    get_system_core
)
from .components import (
    UserManagerComponent, StressAnalysisComponent, AlertManagerComponent,
    create_core_components
)

__all__ = [
    'SystemCore',
    'ComponentManager', 
    'EventBus',
    'IComponent',
    'ComponentConfig',
    'ComponentStatus',
    'SystemEvent',
    'EventType',
    'get_system_core',
    'UserManagerComponent',
    'StressAnalysisComponent', 
    'AlertManagerComponent',
    'create_core_components'
]

def initialize_core_system():
    """Initialize the core system with all components"""
    print("🔧 Initializing Core System V2.0...")
    
    # Get system core
    core = get_system_core()
    
    # Register all core components
    components = create_core_components()
    for component in components:
        core.register_component(component)
    
    print("✅ Core system components registered!")
    return core

# Auto-initialize when package is imported
_core_system = initialize_core_system()
