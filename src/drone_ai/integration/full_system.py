"""
Full Drone AI System Integration

Combines all four layers into a unified autonomous drone system:
- Manager:     Mission planning and task scheduling
- PathFinder:  Path planning with obstacle avoidance
- Perception:  Real-time obstacle detection
- FlyControl:  Low-level flight control

Usage:
    drone = DroneAI()
    drone.add_delivery([100, 50, 0])
    drone.run()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from drone_ai.core.config import SystemConfig, DroneSpecs
from drone_ai.core.state import DroneState, Obstacle

from drone_ai.layers.manager import MissionPlanner, DeliveryRequest
from drone_ai.layers.pathfinder import PathPlanner, smooth_path, interpolate_path
from drone_ai.layers.perception import PerceptionAI
from drone_ai.layers.flycontrol import DroneSimulation, DroneEnv, PPOAgent, TaskType


class SystemState(Enum):
    """Overall system state."""
    IDLE = "idle"
    PLANNING = "planning"
    FLYING = "flying"
    DELIVERING = "delivering"
    RETURNING = "returning"
    LANDED = "landed"
    ERROR = "error"


@dataclass
class FlightStatus:
    """Current flight status."""
    state: SystemState = SystemState.IDLE
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery_level: float = 1.0
    current_waypoint: int = 0
    total_waypoints: int = 0
    obstacles_detected: int = 0
    deliveries_completed: int = 0
    deliveries_pending: int = 0
    elapsed_time: float = 0.0
    error_message: str = ""


class DroneAI:
    """
    Complete Drone AI System integrating all four layers.

    This class orchestrates:
    1. Manager: Decides which delivery to do next
    2. PathFinder: Plans route avoiding obstacles
    3. Perception: Detects obstacles in real-time
    4. FlyControl: Executes flight to follow path

    Example:
        # Create system
        drone = DroneAI()

        # Add deliveries
        drone.add_delivery([100, 50, 0], priority=2)
        drone.add_delivery([50, 100, 0], priority=1)

        # Run autonomous operation
        while not drone.is_mission_complete():
            drone.step()
            print(drone.get_status())
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        use_perception: bool = True,
        use_rl_control: bool = False
    ):
        """
        Initialize the Drone AI system.

        Args:
            config: System configuration
            use_perception: Enable real-time perception (vs simulation obstacles)
            use_rl_control: Use trained RL agent (vs mathematical controller)
        """
        self.config = config or SystemConfig()

        # Initialize base position
        self.base_position = self.config.base_position.copy()

        # Layer 1: Manager
        self.manager = MissionPlanner(
            base_position=self.base_position,
            drone_specs=self.config.drone_specs
        )

        # Layer 2: PathFinder
        self.pathfinder = PathPlanner(
            obstacles=[],
            bounds=self.config.pathfinder.bounds,
            grid_resolution=self.config.pathfinder.grid_resolution
        )

        # Layer 3: Perception
        self.perception = PerceptionAI() if use_perception else None

        # Layer 4: FlyControl
        self.simulation = DroneSimulation(self.config.drone_config)
        self.env = DroneEnv(
            task=TaskType.DELIVERY_ROUTE,
            difficulty=self.config.difficulty,
            config=self.config.drone_config
        )

        # RL Agent (optional)
        self.use_rl_control = use_rl_control
        self.agent: Optional[PPOAgent] = None
        if use_rl_control:
            self.agent = PPOAgent(
                obs_dim=DroneEnv.OBS_DIM,
                action_dim=4
            )

        # System state
        self.state = SystemState.IDLE
        self.current_path: List[np.ndarray] = []
        self.current_waypoint_idx = 0
        self.detected_obstacles: List[Obstacle] = []

        # Statistics
        self.total_steps = 0
        self.total_distance = 0.0
        self._last_position = self.base_position.copy()

    def reset(self):
        """Reset the entire system for a new mission."""
        self.manager.reset()
        self.pathfinder.clear_obstacles()
        self.simulation.reset(position=self.base_position.copy())
        self.env.reset()

        self.state = SystemState.IDLE
        self.current_path = []
        self.current_waypoint_idx = 0
        self.detected_obstacles = []
        self.total_steps = 0
        self.total_distance = 0.0
        self._last_position = self.base_position.copy()

    def add_delivery(
        self,
        target: np.ndarray,
        priority: int = 1,
        weight: float = 1.0
    ) -> DeliveryRequest:
        """
        Add a delivery request.

        Args:
            target: Delivery destination [x, y, z]
            priority: 1=normal, 2=urgent, 3=critical
            weight: Package weight in kg

        Returns:
            Created DeliveryRequest
        """
        return self.manager.create_delivery(
            dropzone_position=np.array(target),
            priority=priority,
            weight=weight
        )

    def add_obstacle(self, obstacle: Obstacle):
        """Add a known obstacle to the environment."""
        self.pathfinder.add_obstacle(obstacle)
        self.simulation.add_obstacle(obstacle)

    def set_obstacles(self, obstacles: List[Obstacle]):
        """Set all obstacles in the environment."""
        self.pathfinder.update_obstacles(obstacles)
        self.simulation.set_obstacles(obstacles)
        self.detected_obstacles = obstacles

    def step(self) -> Tuple[DroneState, float, bool, Dict[str, Any]]:
        """
        Execute one step of the autonomous system.

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.total_steps += 1

        # Update perception (if available)
        if self.perception is not None:
            self._update_perception()

        # State machine
        if self.state == SystemState.IDLE:
            self._handle_idle()

        elif self.state == SystemState.PLANNING:
            self._handle_planning()

        elif self.state == SystemState.FLYING:
            reward, done = self._handle_flying()
            if done:
                self.state = SystemState.DELIVERING

        elif self.state == SystemState.DELIVERING:
            self._handle_delivering()

        elif self.state == SystemState.RETURNING:
            reward, done = self._handle_returning()
            if done:
                self.state = SystemState.IDLE

        # Get current state
        drone_state = self.simulation.state

        # Track distance
        distance = np.linalg.norm(drone_state.position - self._last_position)
        self.total_distance += distance
        self._last_position = drone_state.position.copy()

        # Update manager time
        self.manager.update_time(self.config.drone_config.dt)

        # Build info
        info = {
            'system_state': self.state.value,
            'waypoint_idx': self.current_waypoint_idx,
            'total_waypoints': len(self.current_path),
            'deliveries_completed': len(self.manager.state.completed_deliveries),
            'deliveries_pending': len(self.manager.state.pending_deliveries),
            'obstacles_detected': len(self.detected_obstacles),
            'total_distance': self.total_distance,
        }

        done = self.is_mission_complete()
        reward = self._compute_reward()

        return drone_state, reward, done, info

    def _handle_idle(self):
        """Handle IDLE state - select next delivery."""
        delivery = self.manager.select_next_delivery()

        if delivery is not None:
            self.state = SystemState.PLANNING
            self._plan_path_to(delivery.dropzone_position)
        elif self.manager.should_return_to_base():
            # Return to base
            if np.linalg.norm(self.simulation.state.position - self.base_position) > 1.0:
                self.state = SystemState.RETURNING
                self._plan_path_to(self.base_position)
            else:
                self.state = SystemState.LANDED

    def _handle_planning(self):
        """Handle PLANNING state - path is being computed."""
        if self.current_path:
            self.state = SystemState.FLYING
            self.current_waypoint_idx = 0
            self.env.set_waypoints(self.current_path)

    def _handle_flying(self) -> Tuple[float, bool]:
        """Handle FLYING state - follow path to destination."""
        if not self.current_path:
            return 0.0, True

        # Get current target waypoint
        target = self.current_path[self.current_waypoint_idx]

        # Compute action
        action = self._compute_flight_action(target)

        # Step simulation
        self.simulation.step(action)

        # Check if reached waypoint
        distance = np.linalg.norm(self.simulation.state.position - target)
        if distance < 2.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path):
                return 1.0, True  # Reached destination

        return 0.1, False

    def _handle_delivering(self):
        """Handle DELIVERING state - drop package."""
        # Complete delivery
        current = self.manager.state.current_delivery
        if current is not None:
            # Check if close enough to dropzone
            distance = np.linalg.norm(
                self.simulation.state.position[:2] -
                current.dropzone_position[:2]
            )
            success = distance < 5.0
            self.manager.complete_current_delivery(success)

        self.state = SystemState.IDLE

    def _handle_returning(self) -> Tuple[float, bool]:
        """Handle RETURNING state - fly back to base."""
        if not self.current_path:
            return 0.0, True

        target = self.current_path[self.current_waypoint_idx]
        action = self._compute_flight_action(target)
        self.simulation.step(action)

        distance = np.linalg.norm(self.simulation.state.position - target)
        if distance < 2.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path):
                return 1.0, True

        return 0.1, False

    def _plan_path_to(self, target: np.ndarray):
        """Plan path from current position to target."""
        start = self.simulation.state.position.copy()

        # Add altitude for safe flight
        start_elevated = start.copy()
        start_elevated[2] = max(start[2], 5.0)

        target_elevated = target.copy()
        target_elevated[2] = max(target[2], 5.0)

        # Plan path
        path = self.pathfinder.plan_path(start_elevated, target_elevated)

        if path:
            # Smooth and interpolate
            path = smooth_path(path, smoothing_factor=0.3)
            path = interpolate_path(path, max_segment_length=5.0)

            # Add descent to target
            if target[2] < target_elevated[2]:
                path.append(target)

        self.current_path = path
        self.current_waypoint_idx = 0

    def _update_perception(self):
        """Update obstacle detection from perception layer."""
        if self.perception is None:
            return

        # In simulation mode, use simulated detection
        result = self.perception.detect_from_simulation(
            obstacles=self.simulation.obstacles,
            drone_position=self.simulation.state.position,
            max_range=100.0
        )

        # Update pathfinder with detected obstacles
        obstacles = self.perception.get_obstacles_for_pathfinder(result)
        self.detected_obstacles = obstacles
        self.pathfinder.update_obstacles(obstacles)

    def _compute_flight_action(self, target: np.ndarray) -> np.ndarray:
        """
        Compute flight control action.

        Uses either RL agent or mathematical PD controller.
        """
        if self.use_rl_control and self.agent is not None:
            obs = self.simulation.state.to_observation()
            action, _ = self.agent.select_action(obs)
            return action

        # Mathematical PD controller
        state = self.simulation.state
        position_error = target - state.position

        # Proportional gains
        kp_xy = 0.5
        kp_z = 0.8
        kd = 0.3

        # Desired velocity
        desired_vel = position_error * kp_xy
        desired_vel[2] = position_error[2] * kp_z

        # Velocity error
        vel_error = desired_vel - state.velocity

        # Control output
        control = vel_error * kd

        # Convert to motor commands
        # Simplified: thrust + attitude adjustments
        hover_thrust = 0.5  # Approximate hover

        # Thrust
        thrust = hover_thrust + control[2] * 0.1
        thrust = np.clip(thrust, 0.2, 0.8)

        # Roll/Pitch for horizontal movement
        roll_cmd = -control[1] * 0.1
        pitch_cmd = control[0] * 0.1

        # Motor mixing (simplified)
        motor1 = thrust + pitch_cmd + roll_cmd
        motor2 = thrust + pitch_cmd - roll_cmd
        motor3 = thrust - pitch_cmd - roll_cmd
        motor4 = thrust - pitch_cmd + roll_cmd

        action = np.array([motor1, motor2, motor3, motor4], dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        return action

    def _compute_reward(self) -> float:
        """Compute reward for current step."""
        reward = 0.0

        # Progress reward
        if self.current_path and self.current_waypoint_idx > 0:
            reward += 0.1

        # Delivery completion
        if self.state == SystemState.DELIVERING:
            reward += 10.0

        # Collision penalty
        if self.simulation.is_crashed():
            reward -= 50.0

        return reward

    def is_mission_complete(self) -> bool:
        """Check if all deliveries are complete."""
        return (
            len(self.manager.state.pending_deliveries) == 0 and
            self.manager.state.current_delivery is None and
            self.state in [SystemState.IDLE, SystemState.LANDED]
        )

    def get_status(self) -> FlightStatus:
        """Get current system status."""
        state = self.simulation.state
        return FlightStatus(
            state=self.state,
            position=state.position.copy(),
            target=self.current_path[self.current_waypoint_idx].copy() if self.current_path else np.zeros(3),
            battery_level=state.battery_level,
            current_waypoint=self.current_waypoint_idx,
            total_waypoints=len(self.current_path),
            obstacles_detected=len(self.detected_obstacles),
            deliveries_completed=len(self.manager.state.completed_deliveries),
            deliveries_pending=len(self.manager.state.pending_deliveries),
            elapsed_time=self.manager.state.elapsed_time,
        )

    def get_mission_summary(self) -> Dict[str, Any]:
        """Get comprehensive mission summary."""
        return {
            'manager': self.manager.get_mission_summary(),
            'system_state': self.state.value,
            'total_steps': self.total_steps,
            'total_distance_m': self.total_distance,
            'obstacles_count': len(self.detected_obstacles),
            'path_length': len(self.current_path),
            'crashed': self.simulation.is_crashed(),
        }

    def load_agent(self, path: str):
        """Load trained RL agent from checkpoint."""
        if self.agent is None:
            self.agent = PPOAgent(obs_dim=DroneEnv.OBS_DIM, action_dim=4)
        self.agent.load(path)
        self.use_rl_control = True

    def run_episode(self, max_steps: int = 10000, verbose: bool = False) -> Dict[str, Any]:
        """
        Run a complete episode until mission complete or max steps.

        Args:
            max_steps: Maximum steps per episode
            verbose: Print progress updates

        Returns:
            Episode statistics
        """
        self.reset()

        total_reward = 0.0
        for step in range(max_steps):
            state, reward, done, info = self.step()
            total_reward += reward

            if verbose and step % 100 == 0:
                status = self.get_status()
                print(f"Step {step}: {status.state.value}, "
                      f"pos={status.position[:2]}, "
                      f"deliveries={status.deliveries_completed}/{status.deliveries_pending + status.deliveries_completed}")

            if done:
                break

        return {
            'total_reward': total_reward,
            'steps': step + 1,
            'mission_complete': self.is_mission_complete(),
            **self.get_mission_summary()
        }
