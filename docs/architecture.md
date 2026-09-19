# EchoPath High-Level Planned Architecture

EchoPath is designed as an on-device, real-time spatial awareness system to assist blind and visually impaired individuals. The system processes a mobile camera stream through a modular pipeline to deliver timely, prioritized audio alerts.

---

## High-Level Pipeline Diagram

```
+-------------------+
|   Camera Stream   |
+---------+---------+
          |
          v
+-------------------+
|  Object Detection |  (Identifies relevant obstacles & objects)
+---------+---------+
          |
          v
+-------------------+
| Depth Estimation  |  (Estimates relative or metric depth)
+---------+---------+
          |
          v
+-------------------+
|   Multi-Object    |  (Tracks persistent object IDs across frames)
|     Tracking      |
+---------+---------+
          |
          v
+-------------------+
| Spatial Mapping   |  (Determines 3D position: left, center, right, distance)
+---------+---------+
          |
          v
+-------------------+
| Priority Scoring  |  (Rates danger/urgency based on trajectory & proximity)
+---------+---------+
          |
          v
+-------------------+
| NLG (Natural Lang |  (Formulates concise visual warning text)
|    Generation)    |
+---------+---------+
          |
          v
+-------------------+
| Text-to-Speech    |  (Communicates audio warnings via device speaker/headphones)
|     (TTS)         |
+-------------------+
```

---

## Pipeline Stage Responsibilities

1. **Camera Stream Input:** Captures video frames in real time from the mobile device.
2. **Object Detection:** Identifies critical obstacles (e.g., stairs, vehicles, pedestrians, doors, low-hanging hazards) and outputs bounding boxes with class labels and confidence scores.
3. **Depth Estimation:** Computes depth maps or distance estimates corresponding to detected objects.
4. **Multi-Object Tracking:** Assigns consistent tracking IDs across consecutive frames to maintain object motion continuity.
5. **Spatial Mapping:** Maps bounding box positions and depth estimates into spatial coordinates relative to the user (e.g., "Left, 2 meters away").
6. **Priority / Danger Scoring:** Evaluates threat levels based on distance, trajectory, speed, and hazard type to select the most urgent alert.
7. **Natural Language Generation (NLG):** Formulates concise, natural phrase representations of high-priority hazards.
8. **Text-to-Speech (TTS):** Converts generated alert text into real-time spoken audio for the user.
