"""
Layer 2: PathFinder - Path Planning with Obstacle Avoidance

This layer handles:
- A* path planning for optimal routes
- RRT (Rapidly-exploring Random Trees) for complex environments
- Obstacle collision detection
- Path smoothing and optimization
- Real-time path adjustment

The PathFinder receives targets from Manager and plans HOW to get there.
"""

from drone_ai.layers.pathfinder.path_planning import (
    PathPlanner,
    Node,
    smooth_path,
    interpolate_path,
)

__all__ = [
    "PathPlanner",
    "Node",
    "smooth_path",
    "interpolate_path",
]
