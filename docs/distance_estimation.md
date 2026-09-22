# EchoPath — Distance Estimation (Phase 2)

## Overview

The Distance Estimator component computes approximate physical distance (in meters) for detected objects based on 2D bounding boxes returned by the object detector.

In Phase 2 (V0), distance estimation uses a **pinhole camera projection geometry model**:

$$\text{distance} = \frac{\text{focal\_length\_pixels} \times \text{real\_object\_height\_meters}}{\text{object\_height\_pixels}}$$

Where:
* `focal_length_pixels` ($f$): Camera focal length scaled to frame pixel resolution.
* `real_object_height_meters` ($H$): Assumed average physical height of the detected object class.
* `object_height_pixels` ($h = y_2 - y_1$): Height of object's bounding box in pixels.

---

## Configuration & Assumed Object Dimensions

`DistanceEstimator` uses a configurable class-to-height mapping (`OBJECT_HEIGHTS`):

```python
OBJECT_HEIGHTS = {
    "person": 1.7,       # meters
    "car": 1.5,
    "bicycle": 1.1,
    "motorcycle": 1.2,
    "bus": 3.2,
    "truck": 3.0,
    "dog": 0.5,
    "chair": 0.9,
    "bench": 0.5,
}
```

If an object class is not present in `OBJECT_HEIGHTS`, or if the bounding box is invalid/truncated ($h \le 0$), the estimator returns `None` (`distance_m = None`).

---

## Prototype Focal Length

For V0, focal length is configurable:

```python
estimator = DistanceEstimator(focal_length_pixels=700.0)
```

> [!NOTE]
> Prototype focal length default is set to `700.0` pixels. This must be calibrated for the specific mobile camera hardware (using camera intrinsics $f_y$) for improved metric accuracy.

---

## Important Limitations of V0

1. **Approximate physical height assumption:** Real-world objects vary significantly in size (e.g. child vs. adult, SUV vs. compact car).
2. **Bounding Box Quality:** Distances depend directly on bounding box height $y_2 - y_1$. Partial occlusions or box inaccuracies will distort distance estimations.
3. **Camera Pitch / Elevation:** High camera tilt angles alter apparent bounding box heights.
4. **Not a True Depth Map Replacement:** Geometric V0 is a lightweight baseline designed to unblock downstream pipeline modules without hardware or neural depth requirements.

---

## EchoPath Distance Estimation Upgrade Roadmap

The distance estimator component is modular and designed to be upgraded seamlessly in future phases:

* **V0:** Geometric pinhole projection ($d = \frac{f \cdot H}{h}$) — *(CURRENT)*
* **V1:** Mobile Native AR Depth (ARCore / ARKit depth APIs)
* **V2:** Monocular Depth Estimation Neural Network (e.g., Depth Anything / TFLite monocular depth)
* **V3:** Hybrid Distance Fusion (combining geometric projection, monocular depth maps, and AR sensor data)
