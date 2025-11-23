"""
Main Perception AI Interface

This is the primary interface class that combines all perception components
to provide unified obstacle detection and tracking for drone navigation.

Integrates with PathFinder layer by providing obstacle data in world coordinates.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

from drone_ai.core.state import Obstacle, ObstacleType


@dataclass
class DetectedObstacle:
    """Represents a detected obstacle in world coordinates."""
    id: int                          # Tracking ID (same object = same ID)
    type: str                        # tree/building/person/vehicle/unknown
    position: List[float]            # World coordinates [x, y, z] in meters
    size: List[float]                # [width, depth, height] in meters
    velocity: List[float]            # [vx, vy, vz] in m/s (for moving objects)
    confidence: float                # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    def to_obstacle(self) -> Obstacle:
        """Convert to core Obstacle type for PathFinder."""
        type_map = {
            "tree": ObstacleType.TREE,
            "building": ObstacleType.BUILDING,
            "pole": ObstacleType.POLE,
            "sphere": ObstacleType.SPHERE,
            "cylinder": ObstacleType.CYLINDER,
            "box": ObstacleType.BOX,
        }
        obs_type = type_map.get(self.type, ObstacleType.UNKNOWN)

        return Obstacle(
            position=np.array(self.position),
            size=np.array(self.size),
            obstacle_type=obs_type,
            velocity=np.array(self.velocity),
            confidence=self.confidence,
            track_id=self.id
        )


class ObjectDetector:
    """CNN-based object detector for aerial imagery."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.

        Returns list of detections with bounding boxes and classes.
        """
        # Placeholder - in real implementation, runs CNN model
        # Returns simulated detections for integration testing
        return []

    def get_state_dict(self) -> Dict:
        """Get model state for saving."""
        return {}

    def load_state_dict(self, state: Dict):
        """Load model state."""
        pass


class DepthEstimator:
    """Depth estimation from monocular images."""

    def __init__(self, camera_params: Dict, device: str = "cpu"):
        self.camera_params = camera_params
        self.device = device

    def estimate(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Estimate depth for each detection.

        Adds 'depth' field to each detection.
        """
        for det in detections:
            # Placeholder depth estimation
            det['depth'] = 10.0  # meters
        return detections

    def get_state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict):
        pass


class CoordinateTransformer:
    """Transform coordinates between camera and world frames."""

    def __init__(self, camera_params: Dict):
        self.camera_params = camera_params

    def camera_to_world(
        self,
        detections: List[Dict],
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> List[Dict]:
        """Transform detections from camera to world coordinates."""
        world_detections = []

        for det in detections:
            # Transform based on drone pose
            world_pos = drone_position + np.array([det.get('depth', 10), 0, 0])
            det['world_position'] = world_pos.tolist()
            world_detections.append(det)

        return world_detections


class ObjectTracker:
    """Multi-object tracker using Kalman filters."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 0
        self._tracks = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections."""
        # Simplified tracking - assigns IDs to detections
        tracked = []
        for det in detections:
            track = {
                'id': self._next_id,
                'type': det.get('class', 'unknown'),
                'position': np.array(det.get('world_position', [0, 0, 0])),
                'size': np.array(det.get('size', [1, 1, 1])),
                'velocity': np.zeros(3),
                'confidence': det.get('confidence', 0.5)
            }
            self._next_id += 1
            tracked.append(track)
        return tracked

    def reset(self):
        """Reset tracker state."""
        self._next_id = 0
        self._tracks = {}


class PerceptionAI:
    """
    Main Perception AI interface for drone obstacle detection.

    This class integrates:
    - Object Detector: CNN-based detection from camera images
    - Depth Estimator: 2D to 3D conversion
    - Coordinate Transformer: Camera to world coordinates
    - Object Tracker: Multi-object tracking with persistent IDs

    Usage:
        perception = PerceptionAI()
        result = perception.detect(frame, drone_position, drone_orientation)

    Output format:
        {
            "obstacles": [...],
            "nearest_obstacle_distance": 10.5,
            "ground_distance": 25.0
        }
    """

    # Supported object types
    OBJECT_TYPES = ["tree", "building", "person", "vehicle", "pole", "wire", "bird", "unknown"]

    # Detection range limits (meters)
    MIN_DETECTION_RANGE = 5.0
    MAX_DETECTION_RANGE = 100.0

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_params: Optional[Dict] = None,
        device: str = "auto"
    ):
        """
        Initialize Perception AI with trained models.

        Args:
            model_path: Path to trained perception model (.pt file)
            camera_params: Camera intrinsic parameters
            device: Computation device ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)

        # Default camera parameters (640x480, ~60 degree FOV)
        self.camera_params = camera_params or {
            "width": 640,
            "height": 480,
            "fx": 554.0,
            "fy": 554.0,
            "cx": 320.0,
            "cy": 240.0,
        }

        # Initialize components
        self.detector = ObjectDetector(model_path=model_path, device=self.device)
        self.depth_estimator = DepthEstimator(camera_params=self.camera_params, device=self.device)
        self.transformer = CoordinateTransformer(camera_params=self.camera_params)
        self.tracker = ObjectTracker(max_age=30, min_hits=3)

        # Performance tracking
        self._frame_count = 0
        self._total_time = 0.0
        self._last_fps = 0.0

    def _resolve_device(self, device: str) -> str:
        """Resolve computation device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def detect(
        self,
        frame: np.ndarray,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a camera frame and detect obstacles in world coordinates.

        Args:
            frame: RGB image as numpy array with shape (H, W, 3)
            drone_position: Drone position [x, y, z] in meters (world frame)
            drone_orientation: Drone orientation as quaternion [w, x, y, z]
                              or euler angles [roll, pitch, yaw] in radians

        Returns:
            Dictionary containing:
                - obstacles: List of detected obstacles with world positions
                - nearest_obstacle_distance: Distance to closest obstacle (meters)
                - ground_distance: Height above ground (meters)
        """
        start_time = time.time()

        # Validate inputs
        frame = self._validate_frame(frame)
        drone_position = np.asarray(drone_position, dtype=np.float32)
        drone_orientation = self._normalize_orientation(drone_orientation)

        # Step 1: Detect objects in image
        detections = self.detector.detect(frame)

        # Step 2: Estimate depth for each detection
        detections_with_depth = self.depth_estimator.estimate(frame, detections)

        # Step 3: Transform to world coordinates
        world_detections = self.transformer.camera_to_world(
            detections_with_depth,
            drone_position,
            drone_orientation
        )

        # Step 4: Track objects across frames
        tracked_obstacles = self.tracker.update(world_detections)

        # Build output
        obstacles = []
        for track in tracked_obstacles:
            obstacle = DetectedObstacle(
                id=track["id"],
                type=track["type"],
                position=track["position"].tolist(),
                size=track["size"].tolist(),
                velocity=track["velocity"].tolist(),
                confidence=track["confidence"]
            )
            obstacles.append(obstacle.to_dict())

        # Calculate summary statistics
        nearest_distance = self._calculate_nearest_distance(obstacles, drone_position)
        ground_distance = float(drone_position[2])  # Simplified

        # Update performance metrics
        elapsed = time.time() - start_time
        self._update_performance(elapsed)

        return {
            "obstacles": obstacles,
            "nearest_obstacle_distance": nearest_distance,
            "ground_distance": ground_distance,
            "frame_time_ms": elapsed * 1000,
            "fps": self._last_fps
        }

    def detect_from_simulation(
        self,
        obstacles: List[Obstacle],
        drone_position: np.ndarray,
        max_range: float = 100.0
    ) -> Dict[str, Any]:
        """
        Create perception output from simulation obstacles.

        Used for testing integration without camera input.

        Args:
            obstacles: List of known obstacles from simulation
            drone_position: Current drone position
            max_range: Maximum detection range

        Returns:
            Perception result dict
        """
        detected = []
        for i, obs in enumerate(obstacles):
            distance = np.linalg.norm(obs.position - drone_position)
            if distance <= max_range:
                detected.append({
                    "id": obs.track_id if obs.track_id is not None else i,
                    "type": obs.obstacle_type.value,
                    "position": obs.position.tolist(),
                    "size": obs.size.tolist(),
                    "velocity": obs.velocity.tolist(),
                    "confidence": obs.confidence
                })

        nearest_distance = self._calculate_nearest_distance(detected, drone_position)

        return {
            "obstacles": detected,
            "nearest_obstacle_distance": nearest_distance,
            "ground_distance": float(drone_position[2]),
            "frame_time_ms": 0.0,
            "fps": 0.0
        }

    def get_obstacles_for_pathfinder(
        self,
        perception_result: Dict[str, Any]
    ) -> List[Obstacle]:
        """
        Convert perception result to Obstacle list for PathFinder.

        Args:
            perception_result: Output from detect()

        Returns:
            List of Obstacle objects
        """
        obstacles = []
        for obs_dict in perception_result.get("obstacles", []):
            detected = DetectedObstacle(**obs_dict)
            obstacles.append(detected.to_obstacle())
        return obstacles

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Validate and normalize input frame."""
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {frame.shape}")

        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        return frame

    def _normalize_orientation(self, orientation: np.ndarray) -> np.ndarray:
        """Normalize orientation to quaternion format [w, x, y, z]."""
        orientation = np.asarray(orientation, dtype=np.float32)

        if len(orientation) == 3:
            orientation = self._euler_to_quaternion(orientation)
        elif len(orientation) == 4:
            norm = np.linalg.norm(orientation)
            if norm > 0:
                orientation = orientation / norm
        else:
            raise ValueError(f"Orientation must have 3 or 4 elements, got {len(orientation)}")

        return orientation

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
        roll, pitch, yaw = euler

        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z], dtype=np.float32)

    def _calculate_nearest_distance(
        self,
        obstacles: List[Dict],
        drone_position: np.ndarray
    ) -> float:
        """Calculate distance to nearest obstacle."""
        if not obstacles:
            return float('inf')

        min_distance = float('inf')
        for obs in obstacles:
            pos = np.array(obs["position"])
            distance = np.linalg.norm(pos - drone_position)
            min_distance = min(min_distance, distance)

        return float(min_distance)

    def _update_performance(self, elapsed: float):
        """Update FPS and performance metrics."""
        self._frame_count += 1
        self._total_time += elapsed

        if self._total_time > 0:
            self._last_fps = self._frame_count / self._total_time

        if self._frame_count >= 1000:
            self._frame_count = 100
            self._total_time = 100 / self._last_fps if self._last_fps > 0 else 1.0

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            "average_fps": self._last_fps,
            "total_frames": self._frame_count,
            "total_time_s": self._total_time
        }

    def reset_tracker(self):
        """Reset the object tracker (e.g., for new scene)."""
        self.tracker.reset()
