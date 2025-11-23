"""
Layer 3: Perception - Real-time Obstacle Detection and Tracking

This layer handles:
- CNN-based object detection from camera images
- Depth estimation for 2D-to-3D conversion
- Camera-to-world coordinate transformation
- Multi-object tracking with persistent IDs
- Ground distance estimation

The Perception layer provides environmental awareness to PathFinder.
"""

from drone_ai.layers.perception.perception_ai import PerceptionAI

__all__ = [
    "PerceptionAI",
]
