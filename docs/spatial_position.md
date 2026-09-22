# EchoPath — Spatial Position / Scene Understanding (Phase 4)

## Overview

The Spatial Position Estimator module determines the coarse **horizontal spatial zone** (`LEFT`, `CENTER`, `RIGHT`) of each detected object within the camera image frame.

This provides visual spatial context for downstream alert generation:

```text
Frame 001: person | ID=1 | distance=4.62m | position=LEFT
Frame 001: car    | ID=2 | distance=6.46m | position=CENTER
Frame 001: bus    | ID=3 | distance=12.4m | position=RIGHT
```

---

## Horizontal Region Partitioning

Spatial position is calculated by normalizing the horizontal bounding box center relative to the total frame width ($W$):

$$x_{\text{center}} = \frac{x_1 + x_2}{2}, \quad \text{relative\_x} = \frac{x_{\text{center}}}{W}$$

```text
0.00 ───────── 0.33 ───────── 0.66 ───────── 1.00
     LEFT           CENTER            RIGHT
```

### Boundary Classification Rules:

* $\text{relative\_x} < \frac{1}{3} \implies \mathbf{LEFT}$
* $\frac{1}{3} \le \text{relative\_x} < \frac{2}{3} \implies \mathbf{CENTER}$
* $\text{relative\_x} \ge \frac{2}{3} \implies \mathbf{RIGHT}$

---

## Pipeline Integration Architecture

Spatial estimation runs cleanly after object detection, distance estimation, and tracking:

```text
    Video / Image Frame (width = W)
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
                 ↓
      SpatialPositionEstimator
                 ↓
Tracked Detections (track_id + distance_m + position)
```

---

## Important Limitations of V0

1. **Image-Relative vs. User-Relative Position:** V0 `LEFT` means the object appears in the left third of the 2D camera image frame. It does not yet incorporate device orientation (compass/gyroscope) or 3D camera extrinsics.
2. **Horizontal-Only Division:** V0 divides the image into 3 vertical columns. It does not yet perform 3D spatial mapping or vertical elevation zone mapping (e.g. overhead vs ground level).
3. **No Threat Assessment:** Spatial classification only labels position. Danger rating and collision prediction occur in Phase 5.
