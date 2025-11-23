"""
Gymnasium Environment for Drone Flight Training

Provides a Gymnasium-compatible environment for training
reinforcement learning agents to control a quadcopter drone.

Supports multiple task types and integrates with upper layers.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

from drone_ai.layers.flycontrol.simulation import DroneSimulation
from drone_ai.core.config import DroneConfig
from drone_ai.core.state import Obstacle, generate_random_obstacles


class TaskType(Enum):
    """Available training tasks."""
    HOVER = "hover"
    WAYPOINT = "waypoint"
    DELIVERY = "delivery"
    DELIVERY_ROUTE = "delivery_route"


class DroneEnv(gym.Env):
    """
    Gymnasium environment for drone flight control.

    Integrates with PathFinder for waypoints and Manager for deliveries.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Observation dimensions
    OBS_DIM = 31

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        difficulty: float = 0.5,
        config: Optional[DroneConfig] = None,
    ):
        """
        Initialize the drone environment.

        Args:
            task: Task type (hover, waypoint, delivery, delivery_route)
            max_steps: Maximum steps per episode
            render_mode: "human", "rgb_array", or None
            difficulty: Difficulty level 0.0-1.0
            config: Drone configuration
        """
        super().__init__()

        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.config = config or DroneConfig()

        # Initialize simulation
        self.sim = DroneSimulation(self.config)

        # Action space: 4 motor thrusts (0 to 1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        # Target position
        self.target_position = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.base_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Waypoints from PathFinder
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0

        # Delivery state
        self.dropzone_position = np.zeros(3, dtype=np.float32)
        self.route_phase = "outbound"
        self.deliveries_completed = 0
        self.deliveries_successful = 0

        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self._last_distance = float('inf')

        # Renderer
        self._renderer = None

    def set_waypoints(self, waypoints: List[np.ndarray]):
        """
        Set waypoints from PathFinder layer.

        Args:
            waypoints: List of 3D positions to follow
        """
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.current_waypoint_idx = 0
        if self.waypoints:
            self.target_position = self.waypoints[0].copy()

    def set_obstacles(self, obstacles: List[Obstacle]):
        """
        Set obstacles from Perception layer.

        Args:
            obstacles: List of detected obstacles
        """
        self.sim.set_obstacles(obstacles)

    def set_delivery_target(self, dropzone: np.ndarray):
        """
        Set delivery target from Manager layer.

        Args:
            dropzone: Delivery destination position
        """
        self.dropzone_position = np.array(dropzone, dtype=np.float32)
        self.target_position = self.dropzone_position.copy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset simulation
        init_position = self._get_initial_position()
        self.sim.reset(position=init_position)

        # Setup task
        self._setup_task()

        # Reset tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_waypoint_idx = 0
        self._last_distance = np.linalg.norm(
            self.sim.state.position - self.target_position
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.step_count += 1

        # Clamp action to valid range
        action = np.clip(action, 0.0, 1.0).astype(np.float32)

        # Map [0,1] to [-1,1] for simulation
        sim_action = action * 2 - 1

        # Step simulation
        self.sim.step(sim_action)

        # Calculate reward
        reward = self._compute_reward(action)
        self.episode_reward += reward

        # Check waypoint progress
        self._check_waypoint_progress()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _setup_task(self):
        """Setup task-specific parameters."""
        if self.task == TaskType.HOVER:
            range_val = 2.0 + self.difficulty * 3.0
            self.target_position = np.array([
                self.np_random.uniform(-range_val, range_val),
                self.np_random.uniform(-range_val, range_val),
                self.np_random.uniform(1.0, 3.0 + self.difficulty * 2.0)
            ], dtype=np.float32)

        elif self.task == TaskType.WAYPOINT:
            n_waypoints = int(3 + self.difficulty * 5)
            self.waypoints = []
            for _ in range(n_waypoints):
                wp = np.array([
                    self.np_random.uniform(-5, 5),
                    self.np_random.uniform(-5, 5),
                    self.np_random.uniform(1, 5)
                ], dtype=np.float32)
                self.waypoints.append(wp)
            self.current_waypoint_idx = 0
            if self.waypoints:
                self.target_position = self.waypoints[0].copy()

        elif self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
            distance = 20.0 + self.difficulty * 80.0
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.dropzone_position = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                0.0
            ], dtype=np.float32)
            self.target_position = self.dropzone_position.copy()
            self.target_position[2] = 2.0
            self.route_phase = "outbound"

            # Generate obstacles
            if self.difficulty > 0.3:
                n_obstacles = int(self.difficulty * 10)
                bounds = (-distance, distance, -distance, distance, 0, 20)
                self.sim.obstacles = generate_random_obstacles(
                    n_obstacles, bounds, min_size=1.0, max_size=3.0
                )

    def _get_initial_position(self) -> np.ndarray:
        """Get initial drone position."""
        if self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
            return self.base_position.copy() + np.array([0, 0, 0.5])
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute step reward."""
        state = self.sim.state
        reward = 0.0

        # Base alive reward
        reward += 0.1

        # Distance to target
        distance = np.linalg.norm(state.position - self.target_position)

        # Distance improvement
        if hasattr(self, '_last_distance'):
            improvement = self._last_distance - distance
            reward += improvement * 2.0
        self._last_distance = distance

        # Close to target bonus
        if distance < 1.0:
            reward += 1.0 - distance

        # Stability rewards
        velocity = np.linalg.norm(state.velocity)
        if velocity < 1.0:
            reward += 0.1

        tilt = np.sum(np.abs(state.orientation[:2]))
        if tilt < 0.2:
            reward += 0.1

        # Penalties
        if state.crashed:
            reward -= 10.0

        if state.position[2] < 0.1:
            reward -= 1.0

        if self.sim.check_obstacle_collision():
            reward -= 20.0

        # Out of bounds
        if np.any(np.abs(state.position[:2]) > 100) or state.position[2] > 50:
            reward -= 5.0

        return float(reward)

    def _check_waypoint_progress(self):
        """Check and update waypoint progress."""
        if not self.waypoints:
            return

        state = self.sim.state
        distance = np.linalg.norm(state.position - self.target_position)

        if distance < 1.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.waypoints):
                self.target_position = self.waypoints[self.current_waypoint_idx].copy()

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        state = self.sim.state

        # Crashed
        if state.crashed:
            return True

        # Out of bounds
        if state.position[2] < -1.0:
            return True
        if np.any(np.abs(state.position[:2]) > 150):
            return True

        # Task complete
        if self.task == TaskType.WAYPOINT:
            if self.current_waypoint_idx >= len(self.waypoints):
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        state = self.sim.state
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Position (0-2)
        obs[0:3] = state.position / 50.0

        # Velocity (3-5)
        obs[3:6] = state.velocity / 10.0

        # Orientation (6-8)
        obs[6:9] = state.orientation / np.pi

        # Angular velocity (9-11)
        obs[9:12] = state.angular_velocity / np.pi

        # Target relative position (12-14)
        relative = self.target_position - state.position
        obs[12:15] = relative / 50.0

        # Target distance (15)
        obs[15] = np.linalg.norm(relative) / 100.0

        # Waypoint progress (16-19)
        if self.waypoints:
            obs[16] = self.current_waypoint_idx / max(1, len(self.waypoints))
            obs[17] = len(self.waypoints) / 10.0
        obs[18] = 1.0 if state.has_package else 0.0

        # Battery (19)
        obs[19] = state.battery_level

        # Nearest obstacle (20)
        nearest_dist = self.sim.get_nearest_obstacle_distance()
        obs[20] = min(1.0, nearest_dist / 20.0)

        # Deliveries (21-22)
        obs[21] = self.deliveries_completed / 10.0
        obs[22] = self.deliveries_successful / max(1, self.deliveries_completed)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        state = self.sim.state
        return {
            'position': state.position.copy(),
            'velocity': state.velocity.copy(),
            'target_position': self.target_position.copy(),
            'distance_to_target': np.linalg.norm(state.position - self.target_position),
            'waypoint_idx': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'crashed': state.crashed,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        # Visualization would be implemented here
        return None

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer = None


def register_envs():
    """Register drone environments with Gymnasium."""
    gym.register(
        id="DroneHover-v0",
        entry_point="drone_ai.layers.flycontrol.environment:DroneEnv",
        kwargs={"task": TaskType.HOVER, "difficulty": 0.3},
        max_episode_steps=1000
    )

    gym.register(
        id="DroneWaypoint-v0",
        entry_point="drone_ai.layers.flycontrol.environment:DroneEnv",
        kwargs={"task": TaskType.WAYPOINT, "difficulty": 0.5},
        max_episode_steps=2000
    )

    gym.register(
        id="DroneDelivery-v0",
        entry_point="drone_ai.layers.flycontrol.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY, "difficulty": 0.5},
        max_episode_steps=5000
    )

    gym.register(
        id="DroneDeliveryRoute-v0",
        entry_point="drone_ai.layers.flycontrol.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY_ROUTE, "difficulty": 0.5},
        max_episode_steps=10000
    )
