"""
Drone physics simulation.

This module provides the core simulation components:
- Drone: Quadrotor dynamics with realistic physics
- DroneSimulation: Complete simulation environment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from drone_ai.core.state import DroneState, Obstacle
from drone_ai.core.config import DroneConfig


class Drone:
    """
    Simulated quadrotor drone with realistic physics.

    Uses a simplified quadrotor model with:
    - Mass and inertia properties
    - Thrust and torque from 4 rotors
    - Aerodynamic drag
    - Gravity
    """

    GRAVITY = 9.81  # m/s^2

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        config: Optional[DroneConfig] = None
    ):
        """Initialize drone at given position."""
        self.config = config or DroneConfig()
        self.position = np.array(
            position if position is not None else [0, 0, 1],
            dtype=np.float32
        )
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Motor speeds (normalized 0-1)
        self.motor_speeds = np.ones(4) * 0.5

        # Collision state
        self.crashed = False

    def get_state(self) -> DroneState:
        """Get current drone state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            crashed=self.crashed
        )

    def set_state(self, state: DroneState):
        """Set drone state."""
        self.position = state.position.copy()
        self.velocity = state.velocity.copy()
        self.orientation = state.orientation.copy()
        self.angular_velocity = state.angular_velocity.copy()
        self.crashed = state.crashed

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """Reset drone to initial state."""
        self.position = np.array(
            position if position is not None else [0, 0, 1],
            dtype=np.float32
        )
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.array(
            orientation if orientation is not None else [0, 0, 0],
            dtype=np.float32
        )
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.motor_speeds = np.ones(4) * 0.5
        self.crashed = False

    def _rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix from orientation (roll, pitch, yaw)."""
        roll, pitch, yaw = self.orientation

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def apply_action(self, action: np.ndarray, dt: float):
        """
        Apply control action and simulate one timestep.

        Args:
            action: [thrust, roll_rate, pitch_rate, yaw_rate] normalized to [-1, 1]
                   or [motor1, motor2, motor3, motor4] as motor thrusts
            dt: Timestep in seconds
        """
        if self.crashed:
            return

        action = np.clip(action, -1, 1)

        if len(action) == 4:
            # Interpret as motor commands [m1, m2, m3, m4]
            motor_action = (action + 1) / 2  # Map [-1,1] to [0,1]
            thrust_cmd = np.mean(motor_action)
            roll_rate_cmd = (motor_action[0] + motor_action[2] - motor_action[1] - motor_action[3]) * 0.5
            pitch_rate_cmd = (motor_action[0] + motor_action[1] - motor_action[2] - motor_action[3]) * 0.5
            yaw_rate_cmd = (motor_action[0] + motor_action[3] - motor_action[1] - motor_action[2]) * 0.25
        else:
            thrust_cmd = (action[0] + 1) / 2
            roll_rate_cmd = action[1] if len(action) > 1 else 0
            pitch_rate_cmd = action[2] if len(action) > 2 else 0
            yaw_rate_cmd = action[3] if len(action) > 3 else 0

        # Compute thrust in body frame (upward)
        thrust_magnitude = thrust_cmd * self.config.max_thrust
        thrust_body = np.array([0, 0, thrust_magnitude])

        # Rotate thrust to world frame
        R = self._rotation_matrix()
        thrust_world = R @ thrust_body

        # Compute acceleration
        gravity = np.array([0, 0, -self.GRAVITY * self.config.mass])
        drag = -self.config.drag_coeff_xy * self.velocity * np.abs(self.velocity)

        acceleration = (thrust_world + gravity + drag) / self.config.mass

        # Update velocity and position
        damping = 0.98
        self.velocity = self.velocity * damping + acceleration * dt
        self.position = self.position + self.velocity * dt

        # Ground collision
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = 0.0
            if np.linalg.norm(self.velocity) > 2.0:
                self.crashed = True

        # Update angular velocity with simple rate control
        target_angular_vel = np.array([roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd])
        self.angular_velocity = (
            self.angular_velocity +
            (target_angular_vel - self.angular_velocity) * 5.0 * dt
        )

        # Update orientation
        self.orientation = self.orientation + self.angular_velocity * dt

        # Wrap yaw to [-pi, pi]
        self.orientation[2] = np.arctan2(
            np.sin(self.orientation[2]),
            np.cos(self.orientation[2])
        )

        # Clamp roll and pitch
        self.orientation[0] = np.clip(self.orientation[0], -np.pi/4, np.pi/4)
        self.orientation[1] = np.clip(self.orientation[1], -np.pi/4, np.pi/4)

    def check_collision(self, obstacles: List[Obstacle], margin: float = 0.3) -> bool:
        """Check if drone collides with any obstacle."""
        for obstacle in obstacles:
            if obstacle.contains_point(self.position, margin):
                self.crashed = True
                return True
        return False

    def get_nearest_obstacle_distance(
        self,
        obstacles: List[Obstacle]
    ) -> Tuple[float, Optional[Obstacle]]:
        """Find distance to nearest obstacle."""
        min_dist = float('inf')
        nearest = None

        for obstacle in obstacles:
            dist = obstacle.distance_to_point(self.position)
            if dist < min_dist:
                min_dist = dist
                nearest = obstacle

        return min_dist, nearest


class DroneSimulation:
    """
    Complete drone simulation environment.

    Combines drone physics with obstacle handling and package delivery.
    """

    def __init__(self, config: Optional[DroneConfig] = None):
        """Initialize simulation."""
        self.config = config or DroneConfig()
        self.drone = Drone(config=self.config)
        self.obstacles: List[Obstacle] = []
        self.time = 0.0

        # Package state
        self.has_package = False
        self.package_position: Optional[np.ndarray] = None
        self.package_delivered = False

    @property
    def state(self) -> DroneState:
        """Get current drone state."""
        state = self.drone.get_state()
        state.has_package = self.has_package
        state.timestamp = self.time
        return state

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """Reset simulation."""
        self.drone.reset(position, orientation)
        if velocity is not None:
            self.drone.velocity = np.array(velocity, dtype=np.float32)
        self.time = 0.0
        self.has_package = False
        self.package_position = None
        self.package_delivered = False

    def step(self, action: np.ndarray, dt: Optional[float] = None):
        """
        Advance simulation by one timestep.

        Args:
            action: Control action for drone
            dt: Timestep (uses config.dt if not specified)
        """
        if dt is None:
            dt = self.config.dt

        self.drone.apply_action(action, dt)
        self.time += dt

        # Check collisions
        if self.obstacles:
            self.drone.check_collision(self.obstacles)

        # Update package position if attached
        if self.has_package:
            self.package_position = self.drone.position.copy()

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to simulation."""
        self.obstacles.append(obstacle)

    def set_obstacles(self, obstacles: List[Obstacle]):
        """Set all obstacles."""
        self.obstacles = obstacles

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []

    def pickup_package(self) -> bool:
        """Pick up package at current location."""
        if not self.has_package:
            self.has_package = True
            self.package_position = self.drone.position.copy()
            return True
        return False

    def drop_package(self) -> np.ndarray:
        """Drop package at current location. Returns drop position."""
        if self.has_package:
            self.has_package = False
            drop_pos = self.package_position.copy()
            self.package_delivered = True
            return drop_pos
        return self.drone.position.copy()

    def is_crashed(self) -> bool:
        """Check if drone has crashed."""
        return self.drone.crashed

    def get_nearest_obstacle_distance(self) -> float:
        """Get distance to nearest obstacle."""
        dist, _ = self.drone.get_nearest_obstacle_distance(self.obstacles)
        return dist

    def check_obstacle_collision(self) -> bool:
        """Check if currently colliding with obstacle."""
        return self.drone.check_collision(self.obstacles, margin=0.3)

    def compute_hover_action(self) -> np.ndarray:
        """Compute action needed to hover in place."""
        # Approximate hover thrust
        hover_thrust = self.config.mass * Drone.GRAVITY / self.config.max_thrust
        return np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
