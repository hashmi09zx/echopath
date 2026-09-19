"""
EchoPath - Detection Verification Test Utility (Phase 1: Object Detection)

Accepts an input image or video file path, runs object detection using the Detector pipeline,
draws bounding boxes with class labels and confidence scores, and optionally displays or saves the output.

Usage Example:
    # Test on an image
    python eval/detection_test.py --input path/to/image.jpg --model yolo11n.pt --output outputs/baseline_test.jpg

    # Test on a video
    python eval/detection_test.py --input path/to/video.mp4 --model yolo11n.pt --output outputs/baseline_video.mp4
"""

import argparse
import sys
from pathlib import Path
import cv2

# Import Detector pipeline module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.detector import Detector, Detection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run object detection verification on an image or video file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to input image or video file",
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
        "--iou",
        type=float,
        default=0.45,
        help="IoU NMS threshold (default: 0.45)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output image or video (optional)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display rendered output in a desktop GUI window (if supported)",
    )
    return parser.parse_args()


def prepare_output_path(output_arg: str) -> Path:
    """
    Resolves output path relative to current working directory,
    creates parent directories automatically, and returns absolute Path object.
    """
    if not output_arg:
        return None
    out_path = Path(output_arg).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def draw_detections(frame, detections):
    """Draws bounding boxes, class labels, and confidence scores on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        label = f"{det.class_name} {det.confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background box
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - 6)),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1,
        )

        # Draw text
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


def process_image(detector, input_path, output_arg=None, show=False):
    """Processes a single static image file."""
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"ERROR: Unable to load image file at '{input_path}'")
        sys.exit(1)

    print(f"Running detection on image: {input_path}")
    detections = detector.detect(frame)
    print(f"Detected {len(detections)} objects:")
    for d in detections:
        print(f"  - {d}")

    annotated = draw_detections(frame, detections)

    if output_arg:
        out_path = prepare_output_path(output_arg)
        success = cv2.imwrite(str(out_path), annotated)
        if not success or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"ERROR: Failed to save annotated image to '{out_path}'. "
                f"Check file permissions, parent directory creation, and disk space."
            )
        print(f"Saved annotated image to: {out_path}")

    if show:
        try:
            cv2.imshow("EchoPath Detection Test", annotated)
            print("Press any key in window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Display window disabled (headless environment): {e}")


def process_video(detector, input_path, output_arg=None, show=False):
    """Processes a video file frame by frame."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Unable to open video file at '{input_path}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    out_path = None
    if output_arg:
        out_path = prepare_output_path(output_arg)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"ERROR: Failed to initialize VideoWriter for '{out_path}'. "
                f"Check video codec support and path permissions."
            )

    frame_count = 0
    print(f"Processing video stream: {input_path} ({width}x{height} @ {fps:.1f} FPS)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = detector.detect(frame)
        annotated = draw_detections(frame, detections)

        if writer:
            writer.write(annotated)

        if show:
            try:
                cv2.imshow("EchoPath Detection Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Video playback interrupted by user.")
                    break
            except cv2.error:
                show = False

    cap.release()
    if writer:
        writer.release()
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"ERROR: Failed to write video file to '{out_path}'. Output file was not created."
            )
        print(f"Saved annotated video to: {out_path}")

    if show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} video frames.")


def main():
    args = parse_args()
    if not args.input:
        print("=== EchoPath Detection Verification Test Utility ===")
        print("No --input file specified. Running synthetic diagnostic check...")
        detector = Detector(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
        import numpy as np
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic rectangle to simulate a target
        cv2.rectangle(synthetic_frame, (100, 100), (300, 300), (255, 255, 255), -1)
        dets = detector.detect(synthetic_frame)
        print(f"Diagnostic check complete. Pretrained model '{args.model}' loaded cleanly.")
        print("To test on a real file, run:")
        print("  python eval/detection_test.py --input path/to/test.jpg --output outputs/baseline_test.jpg")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found at '{input_path}'")
        sys.exit(1)

    detector = Detector(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)

    # Determine if image or video by extension
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if input_path.suffix.lower() in video_exts:
        process_video(detector, input_path, args.output, args.show)
    else:
        process_image(detector, input_path, args.output, args.show)


if __name__ == "__main__":
    main()
