# 🛡️ Shift-Aware Spatial-Temporal Modeling for Intelligent Restricted Zone Monitoring

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Static%20Baseline-orange.svg)
![Status](https://img.shields.io/badge/Status-Physics%20HUD%20Upgrade%20Complete-brightgreen.svg)

> **B.Tech Major Project Thesis Repository — Somaiya Vidyavihar University** <br>
> *Does explicit temporal memory combined with boundary physics improve pre-intrusion behavioral intent prediction compared with snapshot-based machine learning?*

## 🎬 Live Demonstrations & Documentation
Recruiters, reviewers, and operators can view the system outputs and manuals below:
* **[🎥 Watch the Video Demo & Telemetry Outputs (Google Drive)](https://drive.google.com/drive/folders/1XPZcmRp3bV3tBFcflgMaWMtF3CzxYUcN?usp=sharing)**
* **[📁 View Full Test Cases & Telemetry Outputs (Google Drive)](https://drive.google.com/drive/folders/1C1llGpaGta3qblRDAbTVEmHlP7fbB5yi?usp=sharing)**
* **[📄 Read the Final Research Paper (PDF)](https://drive.google.com/drive/folders/1eAP5oFu-9kOzQJB-uBzCR86tXPtjD-ti?usp=sharing)**
* **[📐 Read the Technical Manual (System Architecture & Math)](TECHNICAL_MANUAL.md)**
* **[📘 Read the User Manual (Operational & Deployment Guide)](USER_MANUAL.md)**

---

## 📖 Project Overview

Traditional intelligent video surveillance systems are **reactive**: they draw a bounding polygon and trigger an alert only *after* a physical breach occurs. This causes high false-positive rates and creates severe binary alert fatigue for security personnel.

This project introduces a **proactive, physics-augmented intelligence architecture** capable of predicting a subject's behavioral intent *before* they breach a restricted perimeter. By fusing deep learning tracking (YOLOv8 + ByteTrack), sequence modeling (PyTorch LSTM/GRU), and three advanced physics-based dynamics, the system classifies human intent into 6 distinct behavioral states:
- 🟢 **Casual Transit** (Normal pedestrian flow far from zone)
- 🟡 **Approaching Boundary** (Inward motion vector closing in on the perimeter)
- 🟠 **Active Scouting / Suspicious Loitering** (Boundary probing, perimeter casing, or erratic pacing)
- 🟣 **Fleeing** (Rapid high-speed retreat away from the perimeter)
- 🔴 **CRITICAL INTRUSION** (Inviolable point-in-polygon breach override)

---

## ⚡ Advanced Physics-Based Techniques

Located in `physics_advanced.py` and integrated into the decision logic as supplementary evidence:

### 1. Boundary Probing Dynamics (BPD)
- **Concept:** Intruders often probe a perimeter by approaching, pausing/retreating, and moving along the boundary repeatedly before entering.
- **Formulation:** Tracks approach-retreat oscillation cycles ($N_{\text{cycles}}$) and proximity dwell frames ($F_{\text{near}}$) within the critical buffer zone ($d < 130\text{ px}$):
  $$S_{\text{BPD}} = \min\left(1.0, 0.45 \cdot \min\left(1.0, \frac{N_{\text{cycles}}}{2}\right) + 0.35 \cdot \text{ProximityFactor} + 0.20 \cdot \text{FlankingFactor}\right)$$
- **Trigger:** Flags `repeated_probing = True` upon detecting $\ge 2$ cycles or persistent close contact, activating dynamic visual probing tethers and warning banners.

### 2. Trajectory Directional Entropy (TE)
- **Concept:** Measures unpredictability and randomness in a subject's movement direction. Straight-line purposeful walkers have near-zero directional entropy, whereas wandering, casing, and hesitant loitering yield high directional dispersion.
- **Formulation:** Computes heading angles $\theta_t = \text{atan2}(\Delta y_t, \Delta x_t)$ over the 15-frame history window, binned into 8 discrete sectors ($K=8$):
  $$H = -\sum_{i=1}^8 p_i \log_2(p_i), \quad TE = \frac{H}{\log_2(8)} \in [0.0, 1.0]$$
- **Interpretation:** $TE \in [0.0, 0.35]$ indicates directional transit; $TE > 0.60$ flags erratic loitering and scouting behavior.

### 3. Frenet Boundary Dynamics (FBD)
- **Concept:** Projects the subject's instantaneous 2D velocity vector $\vec{v} = (v_x, v_y)$ onto a local Frenet coordinate frame aligned with the closest edge of the restricted zone polygon.
- **Components:**
  - **Tangential Velocity ($V_t$):** $V_t = \vec{v} \cdot \mathbf{t}$ (speed parallel to the perimeter fence).
  - **Normal Velocity ($V_n$):** $V_n = \vec{v} \cdot \mathbf{n}$ ($V_n > 0$ = closing in towards zone; $V_n < 0$ = retreating away).
  - **Action Vector:** Dynamically categorized into `'Closing In'`, `'Moving Away'`, `'Parallel'`, or `'Neutral'`.

---

## 🎬 Live Video 2×2 Grid HUD Visualization

The processor renders an informative **2×2 Grid ($2560\times 1440$)** demonstration video:

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│     Baseline (Random Forest)          │   Strong Baseline (Physics + XGBoost) │
│                                       │                                       │
│   [CCTV with RF Bounding Boxes]       │   [CCTV with Live Physics HUD Overlay,│
│                                       │    BPD, TE, Vt, Vn & Probing Tethers] │
├───────────────────────────────────────┼───────────────────────────────────────┤
│     Sequence Analysis (LSTM)          │  Physics Telemetry & Intent Legend    │
│                                       │                                       │
│   [CCTV with Temporal Sequences       │   ■ Green:   Casual Transit           │
│    and Comet Trajectory Trails]       │   ■ Yellow:  Approaching Boundary     │
│                                       │   ■ Orange:  Active Scouting / Loiter │
│                                       │   ■ Purple:  Fleeing                  │
│                                       │   ■ Red:     CRITICAL INTRUSION       │
│                                       │   Live: FPS | Frame | Threat Breakdown│
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### Visual HUD Elements Embedded in Frames:
- **Tactical Corner-Bracket Bounding Boxes:** Color-coded strictly by predicted intent class.
- **Glassmorphic Telemetry Cards:** Header pill showing `ID`, `Intent`, `Risk %`; sub-rows showing `BPD`, `TE`, `Vt`, `Vn`, and `[Action Vector]`.
- **Directional Velocity Arrows:** Scaled vector arrows pointing along instantaneous movement headings.
- **Trajectory Comet Trails:** 15-frame history fading path trails.
- **Repeated Probing Highlights:** Pulsing amber borders, `! PROBING DETECTED !` banners, and animated tether lines linking the subject's foot point to the exact closest polygon contact point.
- **Restricted Zone Overlay:** Semi-transparent red fill with highlighted boundary outline and zone badge.

---

## 📋 Official Somaiya Test Cases Specification

All test cases match the official university project test specification. The 9 video test cases (`TC_02` through `TC_10`) are automatically captured into high-resolution 2×2 grid clips:

| Test Case | Test Input / Scenario | Expected Result | Status |
| :--- | :--- | :--- | :---: |
| **TC_01** | User clicks 4 points on frame | Polygon saves to JSON; coordinates match UI. | **Pass** |
| **TC_02** | Target $< 50\text{ px}$ from zone, moving Parallel | Intent = `"Active Scouting"`; Box turns **Orange**. | **Pass** |
| **TC_03** | Target foot-point enters zone | Risk Score increases to maximum; Intent = `"CRITICAL INTRUSION"`; Box turns **Red**. | **Pass** |
| **TC_04** | Occlusion: Target hidden for 15 frames | Tracker maintains the same ID through temporary occlusion without losing movement history/cache. | **Pass** |
| **TC_05** | Target walks normally far from restricted zone | Intent = `"Casual Transit"`; Risk Score remains low; Box turns **Green**. | **Pass** |
| **TC_06** | Target moves towards zone; distance decreases and vector = Closing In | Intent = `"Approaching Boundary"`; Box turns **Yellow**. | **Pass** |
| **TC_07** | Target moves very slowly near zone or stops repeatedly | Intent = `"Suspicious Loitering"`; Risk Score increases; Box turns **Orange**. | **Pass** |
| **TC_08** | Target enters zone after approaching/scouting | Point-in-polygon detects target inside; Distance = 0; Intent overridden to `"CRITICAL INTRUSION"`. | **Pass** |
| **TC_09** | Target moves away from zone at speed $> 100\text{ px/s}$ | Intent = `"Fleeing"`; Action Vector = Moving Away; Box turns **Purple**. | **Pass** |
| **TC_10** | Multiple targets appear simultaneously | Each person receives a unique Track ID and is analysed independently. | **Pass** |

---

## 🚀 Execution Commands

### 1. Environment Setup
```bash
git clone https://github.com/AtharvaRathi/Major-Pro.git
cd Major-Pro
source .venv/bin/activate
```

### 2. Run the Physics Unit Tests
```bash
python -m unittest tests.test_physics_advanced
```

### 3. Generate the Complete 21-Minute Demo Video
Generates the 2×2 grid video ($2560\times 1440$) for exactly 21 minutes (~18,900 frames at 14.97 FPS):
```bash
python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --duration-minutes 21 --width 1280
```

*(To run in the background safely:)*
```bash
nohup python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --duration-minutes 21 --width 1280 > processing_21min.log 2>&1 &
tail -f processing_21min.log
```

### 4. Interactive Live Display Mode (with GUI Window)
```bash
python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --display
```

### 5. Extract / Refresh All 9 Test Case Clips
```bash
python extract_test_cases.py
```

---

## 📁 Repository Structure

```text
Major-Pro/
├── TECHNICAL_MANUAL.md               # Production technical manual, math derivations & architecture
├── USER_MANUAL.md                    # Operational user manual, HUD guide & troubleshooting
├── physics_advanced.py               # Core BPD, TE, FBD physics algorithms & HUD drawing engine
├── run_processor.py                  # Master 2x2 grid video processor & telemetry streamer
├── extract_test_cases.py             # Official TC_02 to TC_10 demonstration video extractor
├── core/
│   ├── detector.py                   # YOLOv8n hardware-accelerated detector (MPS/CUDA)
│   ├── zone_manager.py               # Polygon geometric distance & point-in-polygon tester
│   ├── test_case_manager.py          # High-accuracy automated test case capture engine
│   └── physics_advanced.py           # Module bridge for physics imports
├── ml/
│   ├── xgboost_detector.py           # Preserved 7-feature XGBoost baseline model
│   ├── risk_scorer.py                # Multi-factor behavioral & physics risk aggregator
│   ├── anomaly_detector.py           # Random Forest baseline classifier
│   └── lstm_detector.py              # Recurrent temporal sequence classifier
├── tests/
│   └── test_physics_advanced.py      # Automated unit tests for BPD, TE, FBD & test cases
├── data/
│   └── zone_config.json              # Restricted perimeter polygon coordinates
├── results/                          # Output videos and continuous telemetry CSV logs
└── final_research_paper_material/
    └── test_cases/                   # Generated 2x2 grid clips (TC_02 to TC_10)
```
