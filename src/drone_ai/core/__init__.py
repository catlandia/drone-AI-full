"""
Core module containing shared components across all layers.

- config: System and drone configuration
- state: Unified drone state representation
"""

from drone_ai.core.config import DroneConfig, SystemConfig, DroneSpecs
from drone_ai.core.state import DroneState, Obstacle

__all__ = [
    "DroneConfig",
    "SystemConfig",
    "DroneSpecs",
    "DroneState",
    "Obstacle",
]
