# EchoPath

**EchoPath** is a real-time, on-device mobile AI assistant designed for blind and visually impaired users. By utilizing a smartphone camera, EchoPath detects relevant objects in the user's environment, estimates approximate distance and spatial placement, prioritizes potential hazards, and delivers actionable voice alerts using text-to-speech.

---

## Planned Architecture Pipeline

```
[ Camera Stream ]
       │
       ▼
[ Object Detection ]
       │
       ▼
[ Depth Estimation ]
       │
       ▼
[ Multi-Object Tracking ]
       │
       ▼
[ Spatial Mapping ]
       │
       ▼
[ Priority / Danger Scoring ]
       │
       ▼
[ Natural Language Generation (NLG) ]
       │
       ▼
[ Text-to-Speech (TTS) Output ]
```

---

## Mobile Application Stack

The production mobile application will be built using **Flutter**, targeting both **Android** and **iOS** platforms for cross-platform efficiency and high-performance native integrations.

---

## Project Status & Development Phases

* **Phase 0 — Project Foundation:** In Progress
* **Phase 1 — Object Detection:** Not started
* **Phase 2 — Distance + Spatial Mapping:** Not started
* **Phase 3 — Multi-Object Tracking:** Not started
* **Phase 4 — Depth Estimation + Fusion:** Not started
* **Phase 5 — Priority / Danger Scoring:** Not started
* **Phase 6 — NLG + TTS:** Not started
* **Phase 7 — Flutter Mobile App:** Not started
* **Phase 8 — Native Depth Integration:** Not started
* **Phase 9 — Evaluation + Field Testing:** Not started

---

## Getting Started

Refer to [docs/development_rules.md](docs/development_rules.md) for project guidelines and [docs/architecture.md](docs/architecture.md) for the high-level architecture overview.
