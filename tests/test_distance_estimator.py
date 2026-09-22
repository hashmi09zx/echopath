"""
Unit tests for EchoPath Phase 2 — V0 Distance Estimation Module
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.detector import Detection
from distance.distance_estimator import DistanceEstimator


class TestDistanceEstimator(unittest.TestCase):
    """Test suite covering V0 DistanceEstimator functionality and edge cases."""

    def setUp(self):
        """Sets up default DistanceEstimator with prototype focal length (700 px)."""
        self.focal_length = 700.0
        self.estimator = DistanceEstimator(focal_length_pixels=self.focal_length)

    def test_basic_person_distance(self):
        """
        Test 1 — Basic person distance calculation:
        Known:
            focal_length = 700 px
            person_height = 1.7 m
            bbox_height = 340 px (y1=100, y2=440)
        Expected:
            distance = (700 * 1.7) / 340 = 3.5 m
        """
        detection = Detection(
            class_name="person",
            confidence=0.91,
            bbox=(250.0, 100.0, 350.0, 440.0),  # height = 340 px
        )
        dist = self.estimator.estimate(detection)
        self.assertIsNotNone(dist)
        self.assertAlmostEqual(dist, 3.5, places=2)

        # Test estimate_detection updating object
        updated = self.estimator.estimate_detection(detection)
        self.assertEqual(updated.distance_m, 3.5)

    def test_larger_bbox_means_closer_object(self):
        """
        Test 2 — Larger bounding box means closer object:
        Increasing bbox height should decrease the estimated distance.
        """
        det_baseline = Detection(
            class_name="person",
            confidence=0.90,
            bbox=(250.0, 100.0, 350.0, 440.0),  # height = 340 px -> 3.5m
        )
        det_larger = Detection(
            class_name="person",
            confidence=0.90,
            bbox=(250.0, 100.0, 350.0, 780.0),  # height = 680 px -> 1.75m
        )

        dist_baseline = self.estimator.estimate(det_baseline)
        dist_larger = self.estimator.estimate(det_larger)

        self.assertIsNotNone(dist_baseline)
        self.assertIsNotNone(dist_larger)
        self.assertLess(dist_larger, dist_baseline)
        self.assertEqual(dist_larger, 1.75)

    def test_smaller_bbox_means_farther_object(self):
        """
        Test 3 — Smaller bounding box means farther object:
        Decreasing bbox height should increase the estimated distance.
        """
        det_baseline = Detection(
            class_name="person",
            confidence=0.90,
            bbox=(250.0, 100.0, 350.0, 440.0),  # height = 340 px -> 3.5m
        )
        det_smaller = Detection(
            class_name="person",
            confidence=0.90,
            bbox=(250.0, 100.0, 350.0, 270.0),  # height = 170 px -> 7.0m
        )

        dist_baseline = self.estimator.estimate(det_baseline)
        dist_smaller = self.estimator.estimate(det_smaller)

        self.assertIsNotNone(dist_baseline)
        self.assertIsNotNone(dist_smaller)
        self.assertGreater(dist_smaller, dist_baseline)
        self.assertEqual(dist_smaller, 7.0)

    def test_unknown_class(self):
        """
        Test 4 — Unknown object class:
        Classes not present in OBJECT_HEIGHTS map should gracefully return None.
        """
        det_unknown = Detection(
            class_name="traffic_sign",
            confidence=0.88,
            bbox=(100.0, 100.0, 200.0, 300.0),
        )
        dist = self.estimator.estimate(det_unknown)
        self.assertIsNone(dist)

    def test_invalid_bounding_box(self):
        """
        Test 5 — Invalid bounding box:
        Inverted or non-positive box dimensions (x2 <= x1 or y2 <= y1) should return None.
        """
        det_inverted_x = Detection(
            class_name="person",
            confidence=0.85,
            bbox=(350.0, 100.0, 250.0, 400.0),  # x2 < x1
        )
        det_inverted_y = Detection(
            class_name="person",
            confidence=0.85,
            bbox=(250.0, 400.0, 350.0, 100.0),  # y2 < y1
        )

        self.assertIsNone(self.estimator.estimate(det_inverted_x))
        self.assertIsNone(self.estimator.estimate(det_inverted_y))

    def test_zero_height_bounding_box(self):
        """
        Test 6 — Zero-height bounding box:
        Bounding box where y2 == y1 should return None cleanly without division by zero.
        """
        det_zero_h = Detection(
            class_name="person",
            confidence=0.85,
            bbox=(100.0, 200.0, 200.0, 200.0),  # y2 == y1 (height = 0)
        )
        dist = self.estimator.estimate(det_zero_h)
        self.assertIsNone(dist)

    def test_invalid_focal_length(self):
        """
        Test 7 — Invalid focal length:
        Initializing DistanceEstimator with focal_length <= 0 must raise ValueError.
        """
        with self.assertRaises(ValueError):
            DistanceEstimator(focal_length_pixels=0)

        with self.assertRaises(ValueError):
            DistanceEstimator(focal_length_pixels=-500.0)


if __name__ == "__main__":
    unittest.main()
