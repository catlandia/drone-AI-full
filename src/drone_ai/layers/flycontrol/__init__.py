"""
Layer 4: FlyControl - Low-Level Flight Control

This layer handles:
- Quadrotor physics simulation
- Motor thrust control
- Gymnasium-compatible RL environment
- PPO agent for learning flight control
- Hardware interfaces (simulation-ready)

The FlyControl layer executes paths from PathFinder through motor commands.
"""

from drone_ai.layers.flycontrol.simulation import (
    DroneSimulation,
    Drone,
)
from drone_ai.layers.flycontrol.environment import DroneEnv, TaskType
from drone_ai.layers.flycontrol.agent import PPOAgent, PPOConfig

__all__ = [
    "DroneSimulation",
    "Drone",
    "DroneEnv",
    "TaskType",
    "PPOAgent",
    "PPOConfig",
]
