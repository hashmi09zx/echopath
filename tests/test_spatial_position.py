"""
Unit tests for EchoPath Phase 4 — V0 Spatial Position Module
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.detector import Detection
from spatial.spatial_position import SpatialPositionEstimator


class TestSpatialPositionEstimator(unittest.TestCase):
    """Test suite covering SpatialPositionEstimator horizontal region classification."""

    def setUp(self):
        """Sets up default SpatialPositionEstimator."""
        self.estimator = SpatialPositionEstimator()
        self.image_width = 900.0

    def test_left_position(self):
        """
        Test 1 — Left:
        For image_width=900, center_x=100 (bbox=[50, 100, 150, 300]) -> relative_x = 100/900 = 0.11 < 1/3 -> LEFT.
        """
        bbox = (50.0, 100.0, 150.0, 300.0)  # center_x = 100
        pos = self.estimator.classify(bbox, self.image_width)
        self.assertEqual(pos, "LEFT")

    def test_center_position(self):
        """
        Test 2 — Center:
        For image_width=900, center_x=450 (bbox=[400, 100, 500, 300]) -> relative_x = 450/900 = 0.50 -> CENTER.
        """
        bbox = (400.0, 100.0, 500.0, 300.0)  # center_x = 450
        pos = self.estimator.classify(bbox, self.image_width)
        self.assertEqual(pos, "CENTER")

    def test_right_position(self):
        """
        Test 3 — Right:
        For image_width=900, center_x=800 (bbox=[750, 100, 850, 300]) -> relative_x = 800/900 = 0.88 -> RIGHT.
        """
        bbox = (750.0, 100.0, 850.0, 300.0)  # center_x = 800
        pos = self.estimator.classify(bbox, self.image_width)
        self.assertEqual(pos, "RIGHT")

    def test_left_boundary(self):
        """
        Test 4 — Left boundary:
        Boundary check around image_width / 3 = 300.
        center_x = 299 (relative_x = 299/900 < 1/3) -> LEFT
        center_x = 300 (relative_x = 300/900 == 1/3) -> CENTER
        """
        bbox_left = (298.0, 100.0, 300.0, 300.0)  # center_x = 299
        bbox_center = (299.0, 100.0, 301.0, 300.0)  # center_x = 300

        self.assertEqual(self.estimator.classify(bbox_left, self.image_width), "LEFT")
        self.assertEqual(self.estimator.classify(bbox_center, self.image_width), "CENTER")

    def test_center_right_boundary(self):
        """
        Test 5 — Center/Right boundary:
        Boundary check around 2 * image_width / 3 = 600.
        center_x = 599 (relative_x = 599/900 < 2/3) -> CENTER
        center_x = 600 (relative_x = 600/900 == 2/3) -> RIGHT
        """
        bbox_center = (598.0, 100.0, 600.0, 300.0)  # center_x = 599
        bbox_right = (599.0, 100.0, 601.0, 300.0)  # center_x = 600

        self.assertEqual(self.estimator.classify(bbox_center, self.image_width), "CENTER")
        self.assertEqual(self.estimator.classify(bbox_right, self.image_width), "RIGHT")

    def test_invalid_bbox(self):
        """
        Test 6 — Invalid bounding box:
        Inverted or zero/negative dimensions (x2 <= x1 or y2 <= y1) should return None.
        """
        bbox_inv_x = (300.0, 100.0, 200.0, 400.0)
        bbox_inv_y = (100.0, 400.0, 200.0, 100.0)

        self.assertIsNone(self.estimator.classify(bbox_inv_x, self.image_width))
        self.assertIsNone(self.estimator.classify(bbox_inv_y, self.image_width))

    def test_invalid_image_width(self):
        """
        Test 7 — Invalid image width:
        image_width <= 0 must raise ValueError.
        """
        bbox = (100.0, 100.0, 200.0, 300.0)
        with self.assertRaises(ValueError):
            self.estimator.classify(bbox, image_width=0)

        with self.assertRaises(ValueError):
            self.estimator.classify(bbox, image_width=-100)

    def test_information_preservation(self):
        """
        Test 8 — Information preservation:
        SpatialPositionEstimator must preserve existing distance_m, track_id, confidence,
        and class_name attributes on Detection objects.
        """
        det = Detection(
            class_name="car",
            confidence=0.88,
            bbox=(400.0, 100.0, 500.0, 300.0),
            class_id=2,
            distance_m=6.2,
            track_id=42,
        )

        updated = self.estimator.estimate_detection(det, self.image_width)

        self.assertEqual(updated.class_name, "car")
        self.assertEqual(updated.confidence, 0.88)
        self.assertEqual(updated.bbox, (400.0, 100.0, 500.0, 300.0))
        self.assertEqual(updated.class_id, 2)
        self.assertEqual(updated.distance_m, 6.2)
        self.assertEqual(updated.track_id, 42)
        self.assertEqual(updated.position, "CENTER")


if __name__ == "__main__":
    unittest.main()
