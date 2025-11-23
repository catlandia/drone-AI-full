"""
Unified state representations shared across all layers.

Provides common data structures for drone state and obstacles.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class ObstacleType(Enum):
    """Types of obstacles in the environment."""
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    BOX = "box"
    TREE = "tree"
    BUILDING = "building"
    POLE = "pole"
    UNKNOWN = "unknown"


@dataclass
class Obstacle:
    """
    Represents an obstacle in the environment.

    Used by PathFinder for collision avoidance and Perception for detection.

    Attributes:
        position: Center position [x, y, z] in meters
        size: Dimensions depending on type:
            - SPHERE: [radius]
            - CYLINDER: [radius, height]
            - BOX: [width, depth, height]
        obstacle_type: Type of obstacle geometry
        velocity: Optional velocity for moving obstacles [vx, vy, vz]
        confidence: Detection confidence (0.0 to 1.0) from Perception
        track_id: Tracking ID from Perception layer
    """
    position: np.ndarray
    size: np.ndarray
    obstacle_type: ObstacleType = ObstacleType.SPHERE
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 1.0
    track_id: Optional[int] = None

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)
        self.size = np.array(self.size, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)

    def get_radius(self) -> float:
        """Get the effective bounding radius of the obstacle."""
        if self.obstacle_type in [ObstacleType.SPHERE]:
            return float(self.size[0])
        elif self.obstacle_type in [ObstacleType.CYLINDER, ObstacleType.TREE, ObstacleType.POLE]:
            return float(self.size[0])
        else:  # BOX, BUILDING
            return float(np.linalg.norm(self.size[:2]) / 2)

    def get_height(self) -> float:
        """Get the height of the obstacle."""
        if self.obstacle_type == ObstacleType.SPHERE:
            return float(self.size[0] * 2)
        elif len(self.size) >= 2:
            return float(self.size[1] if self.obstacle_type == ObstacleType.CYLINDER else self.size[-1])
        else:
            return float(self.size[0] * 2)

    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a point is inside the obstacle (with optional safety margin)."""
        point = np.array(point)

        if self.obstacle_type == ObstacleType.SPHERE:
            dist = np.linalg.norm(point - self.position)
            return dist < (self.size[0] + margin)

        elif self.obstacle_type in [ObstacleType.CYLINDER, ObstacleType.TREE, ObstacleType.POLE]:
            # Check horizontal distance
            horiz_dist = np.linalg.norm(point[:2] - self.position[:2])
            # Check vertical bounds
            radius = self.size[0]
            height = self.size[1] if len(self.size) > 1 else self.size[0] * 4
            bottom = self.position[2]
            top = self.position[2] + height
            in_vertical = bottom - margin <= point[2] <= top + margin
            return horiz_dist < (radius + margin) and in_vertical

        else:  # BOX, BUILDING
            half_size = self.size / 2 + margin
            rel_pos = np.abs(point - self.position)
            return np.all(rel_pos < half_size)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle surface."""
        point = np.array(point)

        if self.obstacle_type == ObstacleType.SPHERE:
            dist = np.linalg.norm(point - self.position) - self.size[0]
            return max(0.0, dist)

        elif self.obstacle_type in [ObstacleType.CYLINDER, ObstacleType.TREE, ObstacleType.POLE]:
            horiz_dist = np.linalg.norm(point[:2] - self.position[:2])
            radius = self.size[0]
            height = self.size[1] if len(self.size) > 1 else self.size[0] * 4
            horiz_penetration = horiz_dist - radius

            bottom = self.position[2]
            top = self.position[2] + height
            if point[2] < bottom:
                vert_dist = bottom - point[2]
            elif point[2] > top:
                vert_dist = point[2] - top
            else:
                vert_dist = 0.0

            if horiz_penetration > 0 and vert_dist > 0:
                return np.sqrt(horiz_penetration**2 + vert_dist**2)
            else:
                return max(horiz_penetration, vert_dist, 0.0)

        else:  # BOX, BUILDING
            half_size = self.size / 2
            rel_pos = point - self.position
            closest = np.clip(rel_pos, -half_size, half_size)
            return float(np.linalg.norm(rel_pos - closest))

    def update(self, dt: float):
        """Update obstacle position for moving obstacles."""
        self.position = self.position + self.velocity * dt


@dataclass
class DroneState:
    """
    Complete state of the drone used across all layers.

    Provides a unified representation of drone state that can be
    passed between Manager, PathFinder, and FlyControl layers.
    """
    # Position and velocity
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Orientation (euler angles: roll, pitch, yaw)
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Quaternion representation (for 3D graphics and precise calculations)
    quaternion: Optional[np.ndarray] = None

    # Battery
    battery_level: float = 1.0          # 0.0 to 1.0

    # Payload
    has_package: bool = False
    package_weight: float = 0.0         # kg

    # Status
    crashed: bool = False
    timestamp: float = 0.0              # Simulation time

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.array(self.position, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)
        self.orientation = np.array(self.orientation, dtype=np.float32)
        self.angular_velocity = np.array(self.angular_velocity, dtype=np.float32)

        if self.quaternion is not None:
            self.quaternion = np.array(self.quaternion, dtype=np.float32)

    def copy(self) -> 'DroneState':
        """Create a deep copy of the state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            quaternion=self.quaternion.copy() if self.quaternion is not None else None,
            battery_level=self.battery_level,
            has_package=self.has_package,
            package_weight=self.package_weight,
            crashed=self.crashed,
            timestamp=self.timestamp,
        )

    def get_euler_angles(self) -> np.ndarray:
        """Get orientation as euler angles [roll, pitch, yaw]."""
        return self.orientation

    def get_speed(self) -> float:
        """Get the scalar speed."""
        return float(np.linalg.norm(self.velocity))

    def get_altitude(self) -> float:
        """Get the altitude (z-coordinate)."""
        return float(self.position[2])

    def distance_to(self, target: np.ndarray) -> float:
        """Calculate distance to a target position."""
        return float(np.linalg.norm(np.array(target) - self.position))

    def to_observation(self) -> np.ndarray:
        """Convert state to observation vector for RL."""
        obs = np.concatenate([
            self.position / 50.0,           # Normalized position
            self.velocity / 10.0,           # Normalized velocity
            self.orientation / np.pi,       # Normalized orientation
            self.angular_velocity / np.pi,  # Normalized angular velocity
            [self.battery_level],           # Battery
            [1.0 if self.has_package else 0.0],  # Package status
        ])
        return obs.astype(np.float32)


def generate_random_obstacles(
    num_obstacles: int,
    bounds: Tuple[float, float, float, float, float, float],
    min_size: float = 0.5,
    max_size: float = 2.0,
    seed: Optional[int] = None
) -> List[Obstacle]:
    """
    Generate random obstacles within bounds.

    Args:
        num_obstacles: Number of obstacles to generate
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size
        seed: Random seed for reproducibility

    Returns:
        List of Obstacle objects
    """
    if seed is not None:
        np.random.seed(seed)

    obstacles = []
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    obstacle_types = [ObstacleType.SPHERE, ObstacleType.CYLINDER, ObstacleType.BOX,
                      ObstacleType.TREE, ObstacleType.BUILDING]

    for _ in range(num_obstacles):
        position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ])

        # Use random index to select obstacle type (avoids np.random.choice with objects)
        obstacle_type = obstacle_types[np.random.randint(len(obstacle_types))]

        if obstacle_type == ObstacleType.SPHERE:
            size = np.array([np.random.uniform(min_size, max_size)])
        elif obstacle_type in [ObstacleType.CYLINDER, ObstacleType.TREE, ObstacleType.POLE]:
            radius = np.random.uniform(min_size, max_size)
            height = np.random.uniform(min_size * 2, max_size * 3)
            size = np.array([radius, height])
        else:  # BOX, BUILDING
            size = np.random.uniform(min_size, max_size, size=3)

        obstacles.append(Obstacle(position, size, obstacle_type))

    return obstacles
