# EchoPath — Object Tracking (Phase 3)

## Overview

The Object Tracker module associates detections across consecutive video frames to assign persistent **tracking IDs** (`track_id`).

This allows EchoPath to maintain object identity continuity over time:

```text
Frame 001: person | ID=1 | distance=4.62m
Frame 002: person | ID=1 | distance=4.48m
Frame 003: person | ID=1 | distance=4.31m
```

---

## Pipeline Integration Architecture

Object tracking operates strictly after distance estimation:

```text
  Video Frame
       ↓
 YOLO11n (Detector)
       ↓
  Raw Detections
       ↓
 DistanceEstimator
       ↓
 Detections + distance_m
       ↓
   ObjectTracker
       ↓
 Tracked Detections (track_id + distance_m)
```

> [!NOTE]
> Modularity rule: The tracker does not calculate distance or evaluate threat urgency. Its sole responsibility is associating identities across consecutive frames.

---

## Technical Details

* **Tracker Engine:** Integrates ByteTrack (`bytetrack.yaml`) via Ultralytics API.
* **Fallbacks:** Includes an IoU bounding box association fallback for frameless detection testing.
* **Representation:** Extends `Detection` dataclass with `track_id: Optional[int] = None`.
* **Preservation:** Preserves all existing detection properties (`class_name`, `confidence`, `bbox`, `distance_m`).

---

## Why EchoPath Needs Tracking

Object tracking provides the temporal foundation for upcoming pipeline stages:

1. **Spatial Continuity:** Distinguishes between multiple instances of the same object class (e.g. `person [ID:1]` vs `person [ID:2]`).
2. **Trajectory & Movement Analysis:** Tracks distance changes over time ($\Delta d = d_t - d_{t-1}$) to identify approaching hazards.
3. **Alert Debouncing:** Prevents repeated visual/audio warnings for the same stationary obstacle on every frame.

---

## Limitations of V0 Tracking

* **Identity Switching:** Severe occlusions, fast motion, or blurred frames can cause a tracker ID switch.
* **Short-Term Horizon:** V0 tracking is optimized for short-term frame association, not long-term re-identification across camera cuts.
* **No Threat Reasoning:** V0 tracking only assigns IDs; danger scoring and approaching alerts occur in Phase 5.
