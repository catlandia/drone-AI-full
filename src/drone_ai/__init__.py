"""
Drone AI Full - Complete 4-Layer Autonomous Drone System

This package integrates four AI layers for autonomous drone operation:

Layer 1 - Manager:     High-level mission planning and task scheduling
Layer 2 - PathFinder:  Path planning with obstacle avoidance (A*, RRT)
Layer 3 - Perception:  Real-time obstacle detection and tracking
Layer 4 - FlyControl:  Low-level flight control using reinforcement learning

Usage:
    from drone_ai import DroneAI

    drone = DroneAI()
    drone.add_delivery(target=[100, 50, 0])
    drone.run()
"""

__version__ = "1.0.0"
__author__ = "Drone AI Team"

# Core components
from drone_ai.core.config import DroneConfig, SystemConfig
from drone_ai.core.state import DroneState

# Layer imports
from drone_ai.layers.manager import MissionPlanner, DeliveryRequest
from drone_ai.layers.pathfinder import PathPlanner
from drone_ai.layers.perception import PerceptionAI
from drone_ai.layers.flycontrol import DroneSimulation, DroneEnv, PPOAgent

# Integration
from drone_ai.integration.full_system import DroneAI

__all__ = [
    # Version
    "__version__",
    # Config
    "DroneConfig",
    "SystemConfig",
    "DroneState",
    # Layer 1: Manager
    "MissionPlanner",
    "DeliveryRequest",
    # Layer 2: PathFinder
    "PathPlanner",
    # Layer 3: Perception
    "PerceptionAI",
    # Layer 4: FlyControl
    "DroneSimulation",
    "DroneEnv",
    "PPOAgent",
    # Full System
    "DroneAI",
]
