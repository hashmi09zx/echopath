# EchoPath Project Development Rules

To ensure a structured, high-quality, and reproducible development process, all work on the EchoPath project must strictly adhere to the following rules:

---

## 1. Phase-by-Phase Development
* The project must be developed sequentially, phase-by-phase according to `docs/phase_status.md`.
* Each phase must be manually tested and verified before proceeding to the next phase.
* Do not implement future phases early or bypass intermediate phases.

## 2. Model Training Responsibilities
* Antigravity generates the training code; the user executes and analyzes the actual training.
* Model training will be performed by the user, **not** by the AI assistant.
* The AI assistant may generate training scripts, notebooks, or instructions, but the actual execution of training remains with the user.
* Heavy model training should preferably be executed in Google Colab with GPU support.
* Small experiments and normal lightweight development can run locally on the MacBook M4.

## 3. Data Collection & Testing
* Dataset collection, dataset preparation/annotation, and real-world testing will be performed by the user.

## 4. Architecture & Requirements Integrity
* Do not silently modify the architecture or requirements established for present or later phases.
* Do not introduce unexpected dependency or pipeline changes without prior approval.

## 5. Code Quality & Code Retention
* Do not delete working code without first explaining why and obtaining user confirmation.
* Keep implementations clear, modular, clean, understandable, and well-documented.
