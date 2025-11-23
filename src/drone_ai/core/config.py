"""
Unified configuration for the Drone AI system.

Contains configuration classes used across all four layers.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class DroneSpecs:
    """
    Physical specifications of the drone.

    Used by Manager for battery/range calculations and FlyControl for physics.
    """
    # Mass properties
    empty_weight: float = 2.5          # Drone weight without payload (kg)
    max_payload: float = 2.0           # Maximum cargo weight (kg)

    # Battery properties
    battery_capacity_wh: float = 100.0  # Battery capacity in Watt-hours
    battery_voltage: float = 22.2       # Nominal voltage (6S LiPo)

    # Motor/propulsion properties
    num_motors: int = 4
    motor_efficiency: float = 0.85
    propeller_efficiency: float = 0.75

    # Flight characteristics
    cruise_speed: float = 10.0          # Cruise speed in m/s
    max_speed: float = 15.0             # Maximum speed in m/s
    climb_rate: float = 3.0             # Vertical climb rate m/s
    descent_rate: float = 2.0           # Vertical descent rate m/s

    # Power consumption (Watts)
    hover_power_base: float = 150.0     # Base hover power (empty drone)
    power_per_kg: float = 50.0          # Additional power per kg payload
    cruise_power_factor: float = 1.2    # Multiplier vs hover for forward flight
    climb_power_factor: float = 1.8     # Multiplier vs hover for climbing

    # Safety margins
    reserve_battery: float = 0.15       # Reserve battery (don't use last 15%)

    def hover_power(self, payload_kg: float = 0.0) -> float:
        """Calculate hover power consumption in Watts."""
        return self.hover_power_base + (payload_kg * self.power_per_kg)

    def cruise_power(self, payload_kg: float = 0.0) -> float:
        """Calculate cruise flight power consumption in Watts."""
        return self.hover_power(payload_kg) * self.cruise_power_factor

    def climb_power(self, payload_kg: float = 0.0) -> float:
        """Calculate climbing power consumption in Watts."""
        return self.hover_power(payload_kg) * self.climb_power_factor


@dataclass
class DroneConfig:
    """
    Drone physics configuration for simulation.

    Used by FlyControl layer for accurate physics simulation.
    """
    # Physical properties
    mass: float = 1.0                   # Total mass (kg)
    arm_length: float = 0.25            # Motor arm length (m)

    # Inertia tensor components
    ixx: float = 0.01                   # Roll inertia (kg*m^2)
    iyy: float = 0.01                   # Pitch inertia (kg*m^2)
    izz: float = 0.02                   # Yaw inertia (kg*m^2)

    # Motor properties
    motor_constant: float = 8.54858e-6  # Thrust coefficient
    moment_constant: float = 1.6e-2     # Torque coefficient
    max_thrust: float = 15.0            # Max thrust per motor (N)
    motor_time_constant: float = 0.02   # Motor response time (s)

    # Aerodynamics
    drag_coeff_xy: float = 0.1          # Horizontal drag coefficient
    drag_coeff_z: float = 0.15          # Vertical drag coefficient

    # Environment
    gravity: float = 9.81               # Gravitational acceleration (m/s^2)
    air_density: float = 1.225          # Air density (kg/m^3)

    # Simulation
    dt: float = 0.02                    # Timestep (s) - 50 Hz


@dataclass
class PerceptionConfig:
    """
    Configuration for the Perception layer.
    """
    # Camera parameters
    image_width: int = 640
    image_height: int = 480
    camera_fov: float = 60.0            # Field of view (degrees)
    focal_length_x: float = 554.0
    focal_length_y: float = 554.0
    principal_point_x: float = 320.0
    principal_point_y: float = 240.0

    # Detection parameters
    min_detection_range: float = 5.0    # Minimum detection range (m)
    max_detection_range: float = 100.0  # Maximum detection range (m)
    confidence_threshold: float = 0.5   # Minimum confidence for detection

    # Tracking parameters
    tracker_max_age: int = 30           # Max frames without detection
    tracker_min_hits: int = 3           # Min detections to confirm track

    # Device
    device: str = "auto"                # "auto", "cuda", "cpu"


@dataclass
class PathFinderConfig:
    """
    Configuration for the PathFinder layer.
    """
    # Grid resolution
    grid_resolution: float = 0.5        # A* grid cell size (m)

    # World bounds
    x_min: float = -100.0
    x_max: float = 100.0
    y_min: float = -100.0
    y_max: float = 100.0
    z_min: float = 0.0
    z_max: float = 50.0

    # Safety
    safety_margin: float = 0.5          # Safety margin around obstacles (m)

    # Algorithm parameters
    max_iterations: int = 5000          # Max iterations for path planning
    rrt_step_size: float = 1.0          # RRT extension step size (m)
    rrt_goal_bias: float = 0.1          # Probability of sampling goal

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get bounds as tuple."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)


@dataclass
class ManagerConfig:
    """
    Configuration for the Manager layer.
    """
    # Battery thresholds
    battery_critical: float = 0.15      # Must return to base
    battery_low: float = 0.30           # Consider returning
    battery_safe: float = 0.50          # Safe for new deliveries

    # Distance cost factors
    distance_cost_factor: float = 0.001  # Battery cost per meter
    delivery_overhead: float = 0.05      # Fixed cost per delivery

    # Scheduling
    max_deliveries_per_episode: int = 10
    priority_levels: int = 3            # 1=normal, 2=urgent, 3=critical


@dataclass
class SystemConfig:
    """
    Complete system configuration combining all layers.
    """
    # Drone specifications
    drone_specs: DroneSpecs = field(default_factory=DroneSpecs)
    drone_config: DroneConfig = field(default_factory=DroneConfig)

    # Layer configurations
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    pathfinder: PathFinderConfig = field(default_factory=PathFinderConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)

    # Base position
    base_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Rendering
    render_mode: Optional[str] = None   # "human", "rgb_array", None

    # Training
    difficulty: float = 0.5             # 0.0 to 1.0
    domain_randomization: bool = False

    def __post_init__(self):
        """Ensure base_position is numpy array."""
        if not isinstance(self.base_position, np.ndarray):
            self.base_position = np.array(self.base_position, dtype=np.float32)
