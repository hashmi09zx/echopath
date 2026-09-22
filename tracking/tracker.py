"""
EchoPath - Object Tracker Module (Phase 3: V0 Object Tracking)

Associates detections across consecutive video frames to assign persistent tracking IDs (track_id).
Designed to remain completely decoupled from distance estimation, priority scoring, or alerts.

Pipeline Flow:
    Image / Video Frame
           ↓
        YOLO11n
           ↓
       Detection
           ↓
    DistanceEstimator
           ↓
    Detection + distance_m
           ↓
      ObjectTracker
           ↓
    Detection + distance_m + track_id
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detection


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class ObjectTracker:
    """
    Associates object detections across frames and assigns persistent tracking IDs (track_id).
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack.yaml",
        iou_threshold: float = 0.3,
        model_path: str = "yolo11n.pt",
        device: str = "auto",
    ):
        """
        Initializes the ObjectTracker.

        Args:
            tracker_type: Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
            iou_threshold: Minimum IoU overlap required to maintain track identity.
            model_path: Path to PyTorch model weights file (.pt).
            device: Computing device ('cpu', 'mps', 'cuda', or 'auto').
        """
        self.tracker_type = tracker_type
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.device = device

        # Internal state for tracking
        self.next_track_id: int = 1
        self.previous_tracks: List[Detection] = []
        self._yolo_model = None

    def _get_yolo_model(self):
        """Lazy loader for Ultralytics YOLO model used in video tracking mode."""
        if self._yolo_model is None:
            from ultralytics import YOLO
            from pipeline.detector import resolve_device
            resolved = resolve_device(self.device)
            self._yolo_model = YOLO(self.model_path)
        return self._yolo_model

    def reset(self) -> None:
        """Resets all tracker state and tracking ID counters."""
        self.next_track_id = 1
        self.previous_tracks = []
        if self._yolo_model is not None:
            # Re-initialize tracker predictor state if possible
            if hasattr(self._yolo_model, "predictor") and self._yolo_model.predictor is not None:
                if hasattr(self._yolo_model.predictor, "trackers"):
                    self._yolo_model.predictor.trackers = None

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Associates input detections across frames and populates track_id on each Detection.

        Preserves existing fields (class_name, confidence, bbox, distance_m) untouched.

        Args:
            detections: List of Detection objects (from Detector + DistanceEstimator).
            frame: Optional input video frame (NumPy BGR array).

        Returns:
            List of Detection objects with updated track_id values.
        """
        if not detections:
            return []

        # If a video frame is provided, try using Ultralytics ByteTrack tracking API
        if frame is not None and isinstance(frame, np.ndarray):
            try:
                model = self._get_yolo_model()
                from pipeline.detector import resolve_device
                dev = resolve_device(self.device)

                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_type,
                    conf=0.1,  # Low conf threshold so tracker matches existing detections
                    iou=0.45,
                    device=dev,
                    verbose=False,
                )

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.id is not None:
                        tracked_boxes = boxes.xyxy.cpu().numpy()
                        tracked_ids = boxes.id.cpu().numpy().astype(int)

                        # Match each input detection to the closest tracked box by IoU
                        used_track_indices = set()
                        for det in detections:
                            best_iou = -1.0
                            best_idx = -1
                            for idx, (tbox, tid) in enumerate(zip(tracked_boxes, tracked_ids)):
                                if idx in used_track_indices:
                                    continue
                                iou = compute_iou(det.bbox, (float(tbox[0]), float(tbox[1]), float(tbox[2]), float(tbox[3])))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = idx

                            if best_idx != -1 and best_iou >= 0.2:
                                det.track_id = int(tracked_ids[best_idx])
                                used_track_indices.add(best_idx)
                            else:
                                det.track_id = self._assign_fallback_track_id(det)
                        
                        self.previous_tracks = list(detections)
                        return detections
            except Exception as e:
                # Fall back to IoU tracking if Ultralytics track call encounters issues
                pass

        # Fallback / Frameless IoU Tracker logic
        return self._update_iou_fallback(detections)

    def _assign_fallback_track_id(self, detection: Detection) -> int:
        """Assigns a new unique track ID and increments counter."""
        tid = self.next_track_id
        self.next_track_id += 1
        return tid

    def _update_iou_fallback(self, detections: List[Detection]) -> List[Detection]:
        """IoU-based tracker fallback for frameless detection sequences."""
        matched_prev_indices = set()

        for det in detections:
            best_iou = -1.0
            best_prev_idx = -1

            for idx, prev_det in enumerate(self.previous_tracks):
                if idx in matched_prev_indices:
                    continue
                # Class match requirement (or box proximity)
                if det.class_name == prev_det.class_name:
                    iou = compute_iou(det.bbox, prev_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev_idx = idx

            if best_prev_idx != -1 and best_iou >= self.iou_threshold:
                det.track_id = self.previous_tracks[best_prev_idx].track_id
                matched_prev_indices.add(best_prev_idx)
            else:
                det.track_id = self._assign_fallback_track_id(det)

        self.previous_tracks = list(detections)
        return detections
