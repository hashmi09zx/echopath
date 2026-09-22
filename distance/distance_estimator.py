"""
EchoPath - Distance Estimator Module (Phase 2: V0 Distance Estimation)

Provides geometric distance estimation for detected objects using pinhole camera projection geometry:
    distance = (focal_length_pixels * real_object_height_meters) / object_height_pixels

This is a prototype geometric estimator (V0) and serves as the baseline before integrating
hardware depth maps (ARCore/ARKit) or monocular depth neural networks.
"""

from typing import Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path

# Add project root to path if needed for standalone import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detection

# Default physical height assumptions in meters for standard COCO / EchoPath classes
DEFAULT_OBJECT_HEIGHTS: Dict[str, float] = {
    "person": 1.7,
    "car": 1.5,
    "bicycle": 1.1,
    "motorcycle": 1.2,
    "bus": 3.2,
    "truck": 3.0,
    "dog": 0.5,
    "chair": 0.9,
    "bench": 0.5,
}

# Prototype focal length default in pixels (~60 degree FOV at 1080p / 720p scale).
# Prototype focal length.
# This must be calibrated for the target camera for better accuracy.
DEFAULT_FOCAL_LENGTH_PIXELS: float = 700.0


class DistanceEstimator:
    """Estimates physical distance (in meters) of detected objects based on pinhole geometry."""

    def __init__(
        self,
        focal_length_pixels: float = DEFAULT_FOCAL_LENGTH_PIXELS,
        object_heights: Optional[Dict[str, float]] = None,
    ):
        """
        Initializes the Distance Estimator.

        Args:
            focal_length_pixels: Camera focal length in pixels.
            object_heights: Mapping of class names to physical heights in meters.
                            Defaults to DEFAULT_OBJECT_HEIGHTS if None.
        """
        if focal_length_pixels <= 0:
            raise ValueError(
                f"Focal length must be positive, got focal_length_pixels={focal_length_pixels}"
            )

        self.focal_length_pixels = float(focal_length_pixels)
        self.object_heights = (
            dict(object_heights) if object_heights is not None else dict(DEFAULT_OBJECT_HEIGHTS)
        )

    def estimate(
        self,
        detection: Union[Detection, dict],
    ) -> Optional[float]:
        """
        Estimates the distance in meters for a single detection instance or dictionary.

        Args:
            detection: Detection object or dict containing 'class_name' and 'bbox'.

        Returns:
            Estimated distance in meters as float, or None if distance cannot be estimated.
        """
        # Extract class name and bbox
        if isinstance(detection, Detection):
            class_name = detection.class_name
            bbox = detection.bbox
        elif isinstance(detection, dict):
            class_name = detection.get("class_name")
            bbox = detection.get("bbox")
        else:
            return None

        # Check if class physical height is known
        if not class_name or class_name not in self.object_heights:
            return None

        real_height = self.object_heights[class_name]
        if real_height <= 0:
            return None

        # Validate bounding box coordinates [x1, y1, x2, y2]
        if not bbox or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox

        # Invalid box check (x2 <= x1 or y2 <= y1)
        if x2 <= x1 or y2 <= y1:
            return None

        object_height_pixels = y2 - y1

        # Zero or negative height check (division-by-zero protection)
        if object_height_pixels <= 0:
            return None

        # Calculate geometric distance
        distance_m = (self.focal_length_pixels * real_height) / object_height_pixels
        return round(float(distance_m), 2)

    def estimate_detection(self, detection: Detection) -> Detection:
        """
        Estimates distance for a Detection object and updates its distance_m field.

        Args:
            detection: Detection instance to update.

        Returns:
            Updated Detection instance.
        """
        detection.distance_m = self.estimate(detection)
        return detection

    def estimate_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Estimates distance for a list of Detection objects in-place.

        Args:
            detections: List of Detection instances.

        Returns:
            List of Detection instances with populated distance_m fields.
        """
        for det in detections:
            self.estimate_detection(det)
        return detections
