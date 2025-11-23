"""
Path planning algorithms for drone navigation.

Implements A* and RRT (Rapidly-exploring Random Trees) path planning
with obstacle avoidance for 3D drone navigation.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

from drone_ai.core.state import Obstacle


@dataclass
class Node:
    """Node for path planning algorithms."""
    position: np.ndarray
    parent: Optional['Node'] = None
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost

    def __hash__(self) -> int:
        return hash(tuple(np.round(self.position, 2)))

    def __eq__(self, other: 'Node') -> bool:
        if not isinstance(other, Node):
            return False
        return np.allclose(self.position, other.position, atol=0.1)


class PathPlanner:
    """
    Path planning using A* and RRT algorithms.

    Provides methods for computing obstacle-free paths in 3D space.
    Integrates with Perception layer for dynamic obstacle updates.
    """

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        grid_resolution: float = 0.5,
        bounds: Tuple[float, float, float, float, float, float] = (-100, 100, -100, 100, 0, 50)
    ):
        """
        Initialize path planner.

        Args:
            obstacles: List of obstacles in the environment
            grid_resolution: Resolution for A* grid in meters
            bounds: (x_min, x_max, y_min, y_max, z_min, z_max) world bounds
        """
        self.obstacles = obstacles or []
        self.grid_resolution = grid_resolution
        self.bounds = bounds
        self.safety_margin = 0.5  # Safety margin around obstacles

        # 3D movement directions (26-connected)
        self._directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        self._directions.append(np.array([dx, dy, dz]) * grid_resolution)

    def update_obstacles(self, obstacles: List[Obstacle]):
        """Update the obstacle list (from Perception layer)."""
        self.obstacles = obstacles

    def add_obstacle(self, obstacle: Obstacle):
        """Add a single obstacle."""
        self.obstacles.append(obstacle)

    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles = []

    def is_point_valid(self, point: np.ndarray) -> bool:
        """
        Check if a point is valid (in bounds and not in obstacle).

        Args:
            point: 3D point to check

        Returns:
            True if point is valid
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        if not (x_min <= point[0] <= x_max and
                y_min <= point[1] <= y_max and
                z_min <= point[2] <= z_max):
            return False

        for obstacle in self.obstacles:
            if obstacle.contains_point(point, self.safety_margin):
                return False

        return True

    def is_path_clear(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        num_samples: int = 10
    ) -> bool:
        """
        Check if direct path between two points is obstacle-free.

        Args:
            point_a: Start point
            point_b: End point
            num_samples: Number of points to sample along path

        Returns:
            True if path is clear
        """
        point_a = np.array(point_a)
        point_b = np.array(point_b)

        for t in np.linspace(0, 1, num_samples):
            point = point_a + t * (point_b - point_a)
            if not self.is_point_valid(point):
                return False

        return True

    def _heuristic(self, point: np.ndarray, goal: np.ndarray) -> float:
        """Euclidean distance heuristic."""
        return float(np.linalg.norm(point - goal))

    def plan_path_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 5000
    ) -> List[np.ndarray]:
        """
        Plan path using A* algorithm.

        Args:
            start: Start position
            goal: Goal position
            max_iterations: Maximum iterations before giving up

        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        if not self.is_point_valid(start):
            start = self._find_nearest_valid_point(start)
            if start is None:
                return []

        if not self.is_point_valid(goal):
            goal = self._find_nearest_valid_point(goal)
            if goal is None:
                return []

        # Check direct path first
        if self.is_path_clear(start, goal):
            return [start, goal]

        # A* search
        start_node = Node(start, None, 0, self._heuristic(start, goal))

        open_set: List[Node] = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            if np.linalg.norm(current.position - goal) < self.grid_resolution * 1.5:
                return self._reconstruct_path(current, goal)

            pos_tuple = tuple(np.round(current.position / self.grid_resolution).astype(int))
            if pos_tuple in closed_set:
                continue
            closed_set.add(pos_tuple)

            for direction in self._directions:
                neighbor_pos = current.position + direction

                if not self.is_point_valid(neighbor_pos):
                    continue

                neighbor_tuple = tuple(np.round(neighbor_pos / self.grid_resolution).astype(int))
                if neighbor_tuple in closed_set:
                    continue

                move_cost = np.linalg.norm(direction)
                g_cost = current.g_cost + move_cost
                h_cost = self._heuristic(neighbor_pos, goal)

                neighbor_node = Node(neighbor_pos, current, g_cost, h_cost)
                heapq.heappush(open_set, neighbor_node)

        # No path found - return direct path anyway
        return [start, goal]

    def _reconstruct_path(self, node: Node, goal: np.ndarray) -> List[np.ndarray]:
        """Reconstruct path from A* result."""
        path = [goal]
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return self._simplify_path(path)

    def _simplify_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Remove unnecessary waypoints from path."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            furthest = i + 1
            for j in range(i + 2, len(path)):
                if self.is_path_clear(path[i], path[j]):
                    furthest = j

            simplified.append(path[furthest])
            i = furthest

        return simplified

    def _find_nearest_valid_point(
        self,
        point: np.ndarray,
        max_distance: float = 5.0
    ) -> Optional[np.ndarray]:
        """Find nearest valid point to given point."""
        if self.is_point_valid(point):
            return point

        for radius in np.linspace(0.5, max_distance, 10):
            for _ in range(50):
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                test_point = point + direction * radius

                if self.is_point_valid(test_point):
                    return test_point

        return None

    def plan_path_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 2000,
        step_size: float = 1.0,
        goal_bias: float = 0.1
    ) -> List[np.ndarray]:
        """
        Plan path using RRT (Rapidly-exploring Random Trees).

        Args:
            start: Start position
            goal: Goal position
            max_iterations: Maximum iterations
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling goal directly

        Returns:
            List of waypoints from start to goal
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        if not self.is_point_valid(start):
            start = self._find_nearest_valid_point(start)
            if start is None:
                return []

        if not self.is_point_valid(goal):
            goal = self._find_nearest_valid_point(goal)
            if goal is None:
                return []

        if self.is_path_clear(start, goal):
            return [start, goal]

        nodes = [Node(start)]
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds

        for _ in range(max_iterations):
            if np.random.random() < goal_bias:
                sample = goal
            else:
                sample = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max)
                ])

            distances = [np.linalg.norm(n.position - sample) for n in nodes]
            nearest_idx = np.argmin(distances)
            nearest = nodes[nearest_idx]

            direction = sample - nearest.position
            distance = np.linalg.norm(direction)

            if distance < 1e-6:
                continue

            direction = direction / distance
            new_distance = min(distance, step_size)
            new_pos = nearest.position + direction * new_distance

            if not self.is_path_clear(nearest.position, new_pos):
                continue

            new_node = Node(new_pos, nearest)
            nodes.append(new_node)

            if np.linalg.norm(new_pos - goal) < step_size:
                if self.is_path_clear(new_pos, goal):
                    goal_node = Node(goal, new_node)
                    return self._reconstruct_rrt_path(goal_node)

        distances_to_goal = [np.linalg.norm(n.position - goal) for n in nodes]
        best_idx = np.argmin(distances_to_goal)
        return self._reconstruct_rrt_path(nodes[best_idx])

    def _reconstruct_rrt_path(self, node: Node) -> List[np.ndarray]:
        """Reconstruct path from RRT tree."""
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return self._simplify_path(path)

    def plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        algorithm: str = "astar"
    ) -> List[np.ndarray]:
        """
        Plan path from start to goal avoiding obstacles.

        Args:
            start: Start position
            goal: Goal position
            algorithm: "astar" or "rrt"

        Returns:
            List of waypoints from start to goal
        """
        if algorithm == "rrt":
            return self.plan_path_rrt(start, goal)
        else:
            return self.plan_path_astar(start, goal)

    def get_avoidance_waypoint(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacle: Obstacle
    ) -> np.ndarray:
        """
        Generate a waypoint to avoid a specific obstacle.

        Args:
            current_pos: Current drone position
            target_pos: Target position
            obstacle: Obstacle to avoid

        Returns:
            Waypoint that routes around the obstacle
        """
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)

        to_target = target_pos - current_pos
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)

        to_obstacle = obstacle.position - current_pos

        up = np.array([0, 0, 1])
        perp = np.cross(to_target_norm, up)
        perp_norm = np.linalg.norm(perp)

        if perp_norm < 1e-6:
            perp = np.array([1, 0, 0])
        else:
            perp = perp / perp_norm

        side = np.sign(np.dot(perp, to_obstacle))
        if side == 0:
            side = 1

        obstacle_radius = obstacle.get_radius() + self.safety_margin * 2

        avoidance_point = obstacle.position - side * perp * obstacle_radius

        above_point = obstacle.position.copy()
        above_point[2] = obstacle.position[2] + obstacle.get_height() + self.safety_margin * 2

        dist_side = np.linalg.norm(avoidance_point - current_pos) + np.linalg.norm(avoidance_point - target_pos)
        dist_above = np.linalg.norm(above_point - current_pos) + np.linalg.norm(above_point - target_pos)

        if dist_above < dist_side and self.is_point_valid(above_point):
            return above_point
        elif self.is_point_valid(avoidance_point):
            return avoidance_point
        else:
            avoidance_point = obstacle.position + side * perp * obstacle_radius
            if self.is_point_valid(avoidance_point):
                return avoidance_point
            return above_point

    def get_nearest_obstacle_distance(self, position: np.ndarray) -> Tuple[float, Optional[Obstacle]]:
        """
        Find distance to nearest obstacle.

        Args:
            position: Current position

        Returns:
            Tuple of (distance, obstacle) or (inf, None) if no obstacles
        """
        min_dist = float('inf')
        nearest = None

        for obstacle in self.obstacles:
            dist = obstacle.distance_to_point(position)
            if dist < min_dist:
                min_dist = dist
                nearest = obstacle

        return min_dist, nearest


def smooth_path(path: List[np.ndarray], smoothing_factor: float = 0.5) -> List[np.ndarray]:
    """
    Apply smoothing to a path.

    Args:
        path: List of waypoints
        smoothing_factor: How much to smooth (0 = no smoothing, 1 = max smoothing)

    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path

    smoothed = [np.array(p) for p in path]

    for _ in range(10):
        for i in range(1, len(smoothed) - 1):
            midpoint = (smoothed[i-1] + smoothed[i+1]) / 2
            smoothed[i] = smoothed[i] + smoothing_factor * (midpoint - smoothed[i])

    return smoothed


def interpolate_path(path: List[np.ndarray], max_segment_length: float = 1.0) -> List[np.ndarray]:
    """
    Interpolate path to have smaller segment lengths.

    Args:
        path: List of waypoints
        max_segment_length: Maximum distance between consecutive points

    Returns:
        Interpolated path with more waypoints
    """
    if len(path) < 2:
        return path

    interpolated = [path[0]]

    for i in range(1, len(path)):
        start = path[i-1]
        end = path[i]

        distance = np.linalg.norm(end - start)
        num_segments = max(1, int(np.ceil(distance / max_segment_length)))

        for j in range(1, num_segments + 1):
            t = j / num_segments
            point = start + t * (end - start)
            interpolated.append(point)

    return interpolated
