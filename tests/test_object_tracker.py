"""
Unit tests for EchoPath Phase 3 — V0 Object Tracking Module
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.detector import Detection
from distance.distance_estimator import DistanceEstimator
from tracking.tracker import ObjectTracker


class TestObjectTracker(unittest.TestCase):
    """Test suite covering ObjectTracker functionality, ID stability, and edge cases."""

    def setUp(self):
        """Initializes fresh ObjectTracker and DistanceEstimator before each test."""
        self.tracker = ObjectTracker()
        self.distance_estimator = DistanceEstimator(focal_length_pixels=700.0)

    def test_track_id_exists(self):
        """
        Test 1 — Track ID exists:
        A successfully tracked detection must receive a valid non-negative integer track_id.
        """
        det = Detection(
            class_name="person",
            confidence=0.91,
            bbox=(250.0, 100.0, 350.0, 440.0),
        )
        tracked = self.tracker.update([det])
        self.assertEqual(len(tracked), 1)
        self.assertIsNotNone(tracked[0].track_id)
        self.assertIsInstance(tracked[0].track_id, int)
        self.assertGreaterEqual(tracked[0].track_id, 1)

    def test_same_object_across_frames(self):
        """
        Test 2 — Same object across frames:
        When the same object moves slightly across consecutive frames, its track_id must remain stable.
        """
        # Frame 1
        det_f1 = Detection(class_name="person", confidence=0.90, bbox=(100.0, 100.0, 200.0, 400.0))
        res_f1 = self.tracker.update([det_f1])
        id_f1 = res_f1[0].track_id

        # Frame 2 (slightly shifted position)
        det_f2 = Detection(class_name="person", confidence=0.89, bbox=(105.0, 102.0, 205.0, 402.0))
        res_f2 = self.tracker.update([det_f2])
        id_f2 = res_f2[0].track_id

        # Frame 3 (further slight shift)
        det_f3 = Detection(class_name="person", confidence=0.91, bbox=(110.0, 105.0, 210.0, 405.0))
        res_f3 = self.tracker.update([det_f3])
        id_f3 = res_f3[0].track_id

        self.assertEqual(id_f1, id_f2)
        self.assertEqual(id_f2, id_f3)

    def test_multiple_objects(self):
        """
        Test 3 — Multiple objects:
        Two simultaneously visible objects must receive distinct track IDs.
        """
        det_person = Detection(class_name="person", confidence=0.90, bbox=(50.0, 100.0, 150.0, 400.0))
        det_car = Detection(class_name="car", confidence=0.85, bbox=(300.0, 200.0, 500.0, 350.0))

        tracked = self.tracker.update([det_person, det_car])
        self.assertEqual(len(tracked), 2)
        self.assertIsNotNone(tracked[0].track_id)
        self.assertIsNotNone(tracked[1].track_id)
        self.assertNotEqual(tracked[0].track_id, tracked[1].track_id)

    def test_distance_preservation(self):
        """
        Test 4 — Distance preservation:
        ObjectTracker must preserve existing distance_m values attached by DistanceEstimator.
        """
        det = Detection(class_name="person", confidence=0.91, bbox=(250.0, 100.0, 350.0, 440.0))
        # Estimate distance first (3.5m)
        self.distance_estimator.estimate_detection(det)
        self.assertEqual(det.distance_m, 3.5)

        # Pass through tracker
        tracked = self.tracker.update([det])
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].distance_m, 3.5)
        self.assertIsNotNone(tracked[0].track_id)

    def test_empty_detections(self):
        """
        Test 5 — Empty detections:
        ObjectTracker should handle empty detection lists [] safely without error.
        """
        res = self.tracker.update([])
        self.assertEqual(res, [])

    def test_reset(self):
        """
        Test 6 — Reset:
        Resetting ObjectTracker should clear internal tracking state and history.
        """
        det1 = Detection(class_name="person", confidence=0.90, bbox=(100.0, 100.0, 200.0, 400.0))
        res1 = self.tracker.update([det1])
        id1 = res1[0].track_id

        # Reset tracker state
        self.tracker.reset()
        self.assertEqual(self.tracker.next_track_id, 1)
        self.assertEqual(self.tracker.previous_tracks, [])

        # New object after reset should receive initial track ID
        det2 = Detection(class_name="car", confidence=0.85, bbox=(300.0, 200.0, 500.0, 350.0))
        res2 = self.tracker.update([det2])
        id2 = res2[0].track_id

        self.assertEqual(id2, 1)


if __name__ == "__main__":
    unittest.main()
