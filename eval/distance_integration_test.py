"""
EchoPath - Distance Estimation Integration Test Utility (Phase 2: V0 Distance Estimation)

Runs the complete object detection + distance estimation pipeline on an input image or video file.
Combines Detector (YOLO11n) with DistanceEstimator to annotate detections with approximate physical distances.

Usage:
    python eval/distance_integration_test.py --input road_with_car_peole.jpg --output outputs/distance_test_output.jpg
"""

import argparse
import sys
from pathlib import Path
import cv2

# Import pipeline modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detector, Detection
from distance.distance_estimator import DistanceEstimator, DEFAULT_FOCAL_LENGTH_PIXELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detection + distance estimation pipeline on image or video."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="road_with_car_peole.jpg",
        help="Path to input image or video file (default: road_with_car_peole.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO model checkpoint (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=DEFAULT_FOCAL_LENGTH_PIXELS,
        help=f"Camera focal length in pixels (default: {DEFAULT_FOCAL_LENGTH_PIXELS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/distance_test_output.jpg",
        help="Path to save annotated output (default: outputs/distance_test_output.jpg)",
    )
    return parser.parse_args()


def prepare_output_path(output_arg: str) -> Path:
    """Resolves output path, creating parent directories automatically."""
    out_path = Path(output_arg).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def draw_detections_with_distance(frame, detections):
    """Draws bounding boxes, class labels, confidence scores, and estimated distances on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        dist_str = f" | {det.distance_m:.2f}m" if det.distance_m is not None else " | dist=unknown"
        label = f"{det.class_name} {det.confidence:.2f}{dist_str}"

        # Color: Cyan/Green if distance known, Magenta if unknown
        color = (255, 255, 0) if det.distance_m is not None else (255, 0, 255)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background box
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - 6)),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Draw text label in black
        cv2.putText(
            annotated,
            label,
            (x1, max(text_h + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def process_image(detector, distance_estimator, input_path, output_path):
    """Processes static image through Detector + DistanceEstimator pipeline."""
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"ERROR: Unable to load image file at '{input_path}'")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"EchoPath Pipeline: Processing input image '{input_path}'")
    print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]} px")
    print(f"Focal length setting: {distance_estimator.focal_length_pixels} px")
    print(f"=======================================================\n")

    # Step 1: Detect objects using YOLO11n
    raw_detections = detector.detect(frame)
    print(f"[Detector] Found {len(raw_detections)} objects.")

    # Step 2: Estimate distances for detected objects
    detections = distance_estimator.estimate_detections(raw_detections)

    # Step 3: Print summary of results
    print("\n--- Detection & Distance Estimation Results ---")
    for i, det in enumerate(detections, 1):
        if det.distance_m is not None:
            print(f"  [{i}] {det.class_name:<12} | confidence={det.confidence:.2f} | distance≈{det.distance_m:.2f}m")
        else:
            print(f"  [{i}] {det.class_name:<12} | confidence={det.confidence:.2f} | distance=unknown")

    # Step 4: Draw annotations & save output
    annotated = draw_detections_with_distance(frame, detections)
    out_file = prepare_output_path(output_path)
    success = cv2.imwrite(str(out_file), annotated)

    if success and out_file.exists():
        print(f"\n[Success] Saved annotated image to: {out_file}")
    else:
        raise RuntimeError(f"ERROR: Failed to save output image to '{out_file}'")

    return detections


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input file not found at '{input_path}'")
        print("Please provide a valid input image or video path.")
        sys.exit(1)

    # Initialize Detector and DistanceEstimator
    detector = Detector(model_path=args.model, conf_threshold=args.conf)
    distance_estimator = DistanceEstimator(focal_length_pixels=args.focal_length)

    process_image(detector, distance_estimator, input_path, args.output)


if __name__ == "__main__":
    main()
