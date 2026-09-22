"""
EchoPath - Object Tracking Integration Test Utility (Phase 3: V0 Object Tracking)

Runs the complete object detection + distance estimation + tracking pipeline on an input video stream.
Combines Detector (YOLO11n), DistanceEstimator, and ObjectTracker to track objects across video frames
and maintain persistent tracking IDs.

Usage:
    python eval/tracking_integration_test.py --input test_data/tracking_test.mp4 --output outputs/tracking_test_output.mp4
"""

import argparse
import sys
from pathlib import Path
import cv2

# Import pipeline modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detector, Detection
from distance.distance_estimator import DistanceEstimator, DEFAULT_FOCAL_LENGTH_PIXELS
from tracking.tracker import ObjectTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete EchoPath detection + distance + tracking pipeline on a video file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="test_data/tracking_test.mp4",
        help="Path to input video file (default: test_data/tracking_test.mp4)",
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
        default="outputs/tracking_test_output.mp4",
        help="Path to save annotated output video (default: outputs/tracking_test_output.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = process all frames)",
    )
    return parser.parse_args()


def prepare_output_path(output_arg: str) -> Path:
    """Resolves output path, creating parent directories automatically."""
    out_path = Path(output_arg).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def draw_tracked_detections(frame, detections):
    """Draws bounding boxes, class labels, track IDs, confidence scores, and distance estimates on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        
        # Format label strings
        id_str = f" | ID:{det.track_id}" if det.track_id is not None else ""
        dist_str = f" | {det.distance_m:.2f}m" if det.distance_m is not None else " | dist=unknown"
        label = f"{det.class_name}{id_str} {det.confidence:.2f}{dist_str}"

        # Distinct color palette per track ID if present
        if det.track_id is not None:
            # Generate deterministic bright color from track ID
            color_hue = (det.track_id * 45) % 180
            bgr = cv2.cvtColor(
                np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            color = (255, 255, 0)

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

        # Draw text label
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


import numpy as np


def process_video_stream(detector, distance_estimator, object_tracker, input_path, output_path, max_frames=0):
    """Processes video stream frame-by-frame through Detector -> DistanceEstimator -> ObjectTracker pipeline."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Unable to open video file at '{input_path}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_file = prepare_output_path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"ERROR: Failed to initialize VideoWriter for '{out_file}'. Check codec support.")

    print(f"\n=======================================================")
    print(f"EchoPath Pipeline: Tracking video stream '{input_path}'")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")
    print(f"Focal length: {distance_estimator.focal_length_pixels} px")
    print(f"=======================================================\n")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                print(f"Reached specified max frames limit ({max_frames}). Stopping early.")
                break

            # Step 1: Detect objects using YOLO11n
            raw_detections = detector.detect(frame)

            # Step 2: Estimate distances
            dist_detections = distance_estimator.estimate_detections(raw_detections)

            # Step 3: Track objects across frames
            tracked_detections = object_tracker.update(dist_detections, frame=frame)

            # Step 4: Log frame detections
            print(f"Frame {frame_count:03d}")
            if not tracked_detections:
                print("  (no detections)")
            else:
                for det in tracked_detections:
                    id_str = f"ID={det.track_id}" if det.track_id is not None else "ID=none"
                    dist_str = f"distance={det.distance_m:.2f}m" if det.distance_m is not None else "distance=unknown"
                    print(f"  {det.class_name:<10} | {id_str:<6} | confidence={det.confidence:.2f} | {dist_str}")

            # Step 5: Render annotated frame and write to output file
            annotated = draw_tracked_detections(frame, tracked_detections)
            writer.write(annotated)

    finally:
        cap.release()
        writer.release()

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"\n[Success] Processed {frame_count} video frames.")
        print(f"[Success] Saved annotated tracking video to: {out_file}")
    else:
        raise RuntimeError(f"ERROR: Failed to write output video file to '{out_file}'")


def main():
    args = parse_args()
    input_path = Path(args.input)

    # Check for required video file
    if not input_path.exists():
        print(f"ERROR: Input video file not found at '{input_path}'.")
        print("Please ensure the video file exists in the specified location.")
        sys.exit(1)

    detector = Detector(model_path=args.model, conf_threshold=args.conf)
    distance_estimator = DistanceEstimator(focal_length_pixels=args.focal_length)
    object_tracker = ObjectTracker(model_path=args.model)

    process_video_stream(
        detector,
        distance_estimator,
        object_tracker,
        input_path,
        args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
