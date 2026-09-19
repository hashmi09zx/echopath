"""
EchoPath - Object Detector Pipeline Module (Phase 1: Object Detection)

Provides a clean abstraction around Ultralytics YOLO models for real-time frame object detection.
Designed to remain completely decoupled from tracking, depth estimation, or mobile communication logic.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    """Resolves 'auto' device selection to PyTorch compatible device string ('0', 'mps', or 'cpu')."""
    if not device or device == "auto":
        if torch.cuda.is_available():
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    class_id: int = -1

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return f"Detection(class='{self.class_name}', conf={self.confidence:.2f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}))"


class Detector:
    """Wrapper class around Ultralytics YOLO object detector."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        Initializes the YOLO object detector.

        Args:
            model_path: Path to PyTorch model weights file (.pt) or export format.
            conf_threshold: Minimum confidence score to retain a detection.
            iou_threshold: Intersection-over-Union threshold for Non-Maximum Suppression (NMS).
            device: Computing device ('cpu', 'mps', 'cuda', '0', or 'auto').
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = resolve_device(device)

        print(f"[Detector] Loading model checkpoint: {model_path} on device: '{self.device}'...")
        self.model = YOLO(model_path)
        print(f"[Detector] Model loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a single image frame.

        Args:
            frame: Input image as a NumPy array (OpenCV BGR image, shape HxWxC).

        Returns:
            List of Detection instances representing detected objects.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray image.")

        # Run inference using configured thresholds
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract names dictionary
        names = first_result.names

        for box in boxes:
            # Coordinates (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            cls_name = names.get(cls_id, str(cls_id))

            detection = Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
            )
            detections.append(detection)

        return detections


if __name__ == "__main__":
    print("Testing Detector initialization with lightweight synthetic frame...")
    detector = Detector(model_path="yolo11n.pt", conf_threshold=0.25)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_frame)
    print(f"Detector dry test passed. Output detections on black frame: {len(results)}")
