"""
EchoPath - Spatial Position Estimator Module (Phase 4: V0 Spatial Position / Scene Understanding)

Determines the coarse horizontal image region (LEFT, CENTER, RIGHT) of detected objects based on
their horizontal bounding box center coordinates normalized by frame width:

    center_x = (x1 + x2) / 2
    relative_x = center_x / image_width

    relative_x < 1/3   → LEFT
    relative_x < 2/3   → CENTER
    otherwise          → RIGHT

This module operates independently from YOLO, distance estimation, and tracking.
"""

from typing import List, Optional, Tuple, Union
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detection

POSITION_LEFT = "LEFT"
POSITION_CENTER = "CENTER"
POSITION_RIGHT = "RIGHT"


class SpatialPositionEstimator:
    """Classifies bounding boxes into horizontal spatial regions (LEFT, CENTER, RIGHT)."""

    def __init__(self):
        """Initializes the SpatialPositionEstimator."""
        pass

    def classify(
        self,
        bbox: Tuple[float, float, float, float],
        image_width: float,
    ) -> Optional[str]:
        """
        Classifies the horizontal spatial region of a bounding box given the frame width.

        Args:
            bbox: Bounding box tuple (x1, y1, x2, y2).
            image_width: Total frame width in pixels (> 0).

        Returns:
            "LEFT", "CENTER", "RIGHT", or None if bbox is invalid.
        """
        if image_width <= 0:
            raise ValueError(f"image_width must be positive, got image_width={image_width}")

        if not bbox or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox

        # Invalid box check (x2 <= x1 or y2 <= y1)
        if x2 <= x1 or y2 <= y1:
            return None

        center_x = (x1 + x2) / 2.0
        relative_x = center_x / float(image_width)

        if relative_x < (1.0 / 3.0):
            return POSITION_LEFT
        elif relative_x < (2.0 / 3.0):
            return POSITION_CENTER
        else:
            return POSITION_RIGHT

    def estimate_detection(
        self,
        detection: Detection,
        image_width: float,
    ) -> Detection:
        """
        Estimates spatial position for a Detection instance and updates its position field.

        Args:
            detection: Detection instance to update.
            image_width: Frame width in pixels (> 0).

        Returns:
            Updated Detection instance.
        """
        detection.position = self.classify(detection.bbox, image_width)
        return detection

    def estimate_detections(
        self,
        detections: List[Detection],
        image_width: float,
    ) -> List[Detection]:
        """
        Estimates spatial position for a list of Detection objects in-place.

        Args:
            detections: List of Detection objects.
            image_width: Frame width in pixels (> 0).

        Returns:
            List of Detection objects with populated position fields.
        """
        for det in detections:
            self.estimate_detection(det, image_width)
        return detections
