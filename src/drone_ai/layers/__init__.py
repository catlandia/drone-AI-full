"""
Drone AI Layers

Four-layer architecture for autonomous drone operation:

Layer 1 - Manager:     High-level mission planning (WHAT to do)
Layer 2 - PathFinder:  Path planning with obstacle avoidance (HOW to get there)
Layer 3 - Perception:  Real-time obstacle detection (WHAT's in the way)
Layer 4 - FlyControl:  Low-level flight control (HOW to fly)

Data Flow:
    Manager -> PathFinder -> FlyControl
                   ^
                   |
              Perception
"""

from drone_ai.layers.manager import MissionPlanner, DeliveryRequest
from drone_ai.layers.pathfinder import PathPlanner
from drone_ai.layers.perception import PerceptionAI
from drone_ai.layers.flycontrol import DroneSimulation, DroneEnv, PPOAgent

__all__ = [
    # Layer 1
    "MissionPlanner",
    "DeliveryRequest",
    # Layer 2
    "PathPlanner",
    # Layer 3
    "PerceptionAI",
    # Layer 4
    "DroneSimulation",
    "DroneEnv",
    "PPOAgent",
]
