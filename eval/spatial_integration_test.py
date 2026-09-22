"""
EchoPath - Spatial Position Integration Test Utility (Phase 4: V0 Spatial Position / Scene Understanding)

Runs the complete 4-stage EchoPath pipeline on an input video file:
    Video Frame -> Detector (YOLO11n) -> DistanceEstimator -> ObjectTracker -> SpatialPositionEstimator

Usage:
    python eval/spatial_integration_test.py --input test_data/tracking_test.mp4 --output outputs/spatial_test_output.mp4
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Import pipeline modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detector, Detection
from distance.distance_estimator import DistanceEstimator, DEFAULT_FOCAL_LENGTH_PIXELS
from tracking.tracker import ObjectTracker
from spatial.spatial_position import SpatialPositionEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete EchoPath detection + distance + tracking + spatial position pipeline."
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
        default="outputs/spatial_test_output.mp4",
        help="Path to save annotated output video (default: outputs/spatial_test_output.mp4)",
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


def draw_spatial_detections(frame, detections, image_width):
    """Draws vertical zone divider lines and annotated bounding boxes with position tags."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw vertical guide lines for LEFT / CENTER / RIGHT zones
    w1_3 = int(w / 3.0)
    w2_3 = int(2.0 * w / 3.0)

    # Subtle guide lines
    cv2.line(annotated, (w1_3, 0), (w1_3, h), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(annotated, (w2_3, 0), (w2_3, h), (100, 100, 100), 1, cv2.LINE_AA)

    # Zone header text at top of frame
    cv2.putText(annotated, "LEFT", (w1_3 // 2 - 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(annotated, "CENTER", (w1_3 + (w1_3 // 2) - 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(annotated, "RIGHT", (w2_3 + (w1_3 // 2) - 25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        
        # Build label string
        id_str = f" | ID:{det.track_id}" if det.track_id is not None else ""
        dist_str = f" | {det.distance_m:.2f}m" if det.distance_m is not None else " | dist=unknown"
        pos_str = f" | {det.position}" if det.position is not None else ""
        label = f"{det.class_name}{id_str} {det.confidence:.2f}{dist_str}{pos_str}"

        # Distinct color based on track ID
        if det.track_id is not None:
            color_hue = (det.track_id * 45) % 180
            bgr = cv2.cvtColor(
                np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            color = (255, 255, 0)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - 6)),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Label text
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


def process_video_spatial_pipeline(detector, distance_estimator, object_tracker, spatial_estimator, input_path, output_path, max_frames=0):
    """Processes video stream through 4-stage pipeline."""
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
    print(f"EchoPath Pipeline: 4-Stage Processing for '{input_path}'")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")
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

            # Stage 1: Object Detection (YOLO11n)
            raw_dets = detector.detect(frame)

            # Stage 2: Distance Estimation
            dist_dets = distance_estimator.estimate_detections(raw_dets)

            # Stage 3: Object Tracking
            tracked_dets = object_tracker.update(dist_dets, frame=frame)

            # Stage 4: Spatial Position Estimation
            final_dets = spatial_estimator.estimate_detections(tracked_dets, image_width=width)

            # Log frame summary
            print(f"Frame {frame_count:03d}")
            if not final_dets:
                print("  (no detections)")
            else:
                for det in final_dets:
                    id_str = f"ID={det.track_id}" if det.track_id is not None else "ID=none"
                    dist_str = f"distance={det.distance_m:.2f}m" if det.distance_m is not None else "distance=unknown"
                    pos_str = f"position={det.position}" if det.position is not None else "position=unknown"
                    print(f"  {det.class_name:<10} | {id_str:<6} | confidence={det.confidence:.2f} | {dist_str} | {pos_str}")

            # Render & write frame
            annotated = draw_spatial_detections(frame, final_dets, image_width=width)
            writer.write(annotated)

    finally:
        cap.release()
        writer.release()

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"\n[Success] Processed {frame_count} video frames.")
        print(f"[Success] Saved annotated spatial tracking video to: {out_file}")
    else:
        raise RuntimeError(f"ERROR: Failed to write output video file to '{out_file}'")


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input video file not found at '{input_path}'.")
        sys.exit(1)

    detector = Detector(model_path=args.model, conf_threshold=args.conf)
    distance_estimator = DistanceEstimator(focal_length_pixels=args.focal_length)
    object_tracker = ObjectTracker(model_path=args.model)
    spatial_estimator = SpatialPositionEstimator()

    process_video_spatial_pipeline(
        detector,
        distance_estimator,
        object_tracker,
        spatial_estimator,
        input_path,
        args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
