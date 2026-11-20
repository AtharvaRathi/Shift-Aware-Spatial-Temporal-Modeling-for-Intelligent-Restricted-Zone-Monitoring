# 📘 User Manual & Operational Guide
# Shift-Aware Spatial-Temporal Modeling for Intelligent Restricted Zone Monitoring
### Real-Time Video Surveillance HUD with Physics-Augmented Threat Intelligence

---

> **Institution:** Somaiya Vidyavihar University — K. J. Somaiya College of Engineering  
> **Project Thesis:** *Does explicit temporal memory combined with boundary physics improve pre-intrusion behavioral intent prediction compared with snapshot-based machine learning?*  
> **System Release:** Version `2.4.0-PRODUCTION`  
> **Intended Audience:** Security Officers, Surveillance Operators, Facility Managers, and System Evaluators  

---

## 1. Introduction & Overview

Welcome to the **Shift-Aware Spatial-Temporal Restricted Zone Monitoring System**. 

Traditional CCTV surveillance systems are **reactive**: they draw a virtual tripwire or polygon and sound an alarm only *after* an intruder has physically breached the perimeter. This creates two critical flaws in high-security facilities:
1. **Zero Pre-Warning:** Security personnel receive no forewarning while an intruder cases the fence line or tests perimeter defenses.
2. **Binary Alert Fatigue:** Wind, birds, or innocent pedestrians walking near the fence trigger constant false alarms.

This software upgrades CCTV monitoring to a **proactive, intent-aware intelligence platform**. By tracking human ground foot-points and applying advanced boundary physics (**Boundary Probing Dynamics**, **Trajectory Entropy**, and **Frenet Velocity Vectors**), the system recognizes whether an individual is:
- Merely walking past (**Casual Transit**)
- Moving straight towards the boundary (**Approaching Boundary**)
- Pacing, casing, or probing the fence (**Active Scouting / Suspicious Loitering**)
- Running away in panic (**Fleeing**)
- Physically breaching the perimeter (**CRITICAL INTRUSION**)

All metrics and classifications are rendered in real time through an informative **2×2 Grid Heads-Up Display (HUD)**.

---

## 2. System Requirements & Prerequisites

### 2.1 Hardware Requirements

| Component | Minimum Specification | Recommended Production Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 Cores) | Apple M1/M2/M3/M4 or Intel Core i7 12th Gen+ |
| **Graphics (GPU)** | Integrated CPU Graphics | Apple Silicon Metal (MPS) or NVIDIA RTX 3060/4050+ |
| **RAM** | 8 GB | 16 GB or 32 GB |
| **Storage** | 5 GB free disk space | Fast NVMe SSD (for 4K video I/O) |
| **Display** | $1920\times 1080$ Full HD | $2560\times 1440$ 2K or $3840\times 2160$ 4K Monitor |

### 2.2 Software Environment
- **Operating System:** macOS (Sonoma / Sequoia recommended with Metal support), Ubuntu Linux 22.04 LTS, or Windows 11.
- **Python Version:** Python 3.9, 3.10, or 3.11.
- **Video Input:** Any standard `.mp4`, `.avi`, or `.mov` video file or RTSP CCTV camera feed.

---

## 3. Installation & Quick-Start Guide

### Step 1: Clone the Repository
Open your Terminal or Command Prompt and clone the repository:
```bash
git clone https://github.com/AtharvaRathi/Major-Pro.git
cd Major-Pro
```

### Step 2: Set Up Virtual Environment & Dependencies
Create and activate an isolated Python virtual environment:
```bash
# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Windows:
python -m venv .venv
.venv\Scripts\activate
```

Install all required production packages:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> If you are on macOS and using XGBoost with OpenMP, verify your Homebrew `libomp` installation:
> ```bash
> brew install libomp
> ```

### Step 3: Run the Verification Unit Tests
Confirm that the physics algorithms, zone geometry, and test cases function properly on your machine:
```bash
python -m unittest discover tests
```
*Expected Output:* `Ran 6 tests in 0.002s - OK`

---

## 4. Calibrating the Restricted Zone Boundary

Before processing live surveillance video, you must define the exact boundary of your restricted perimeter on the camera frame.

### 4.1 Interactive Polygon Mapping Tool (`map_zone.py`)

Run the interactive calibration tool:
```bash
python map_zone.py
```

```
┌────────────────────────────────────────────────────────┐
│               Interactive Mapping Window               │
│                                                        │
│       Click 1 ●───────────────● Click 2               │
│               │               │                        │
│               │  RESTRICTED   │                        │
│               │     ZONE      │                        │
│               │               │                        │
│       Click 4 ●───────────────● Click 3                │
│                                                        │
│  [c] Close Polygon   [s] Save Config   [q] Discard     │
└────────────────────────────────────────────────────────┘
```

#### Step-by-Step Instructions:
1. A window titled **"Map Restricted Zone"** will display the first frame of your camera video.
2. **Left-Click** sequentially on each corner of the restricted perimeter to drop vertices. A red marker and connecting lines will appear as you click.
3. Once all boundary points are placed (minimum 3 points), press the key **`c`** on your keyboard to automatically draw the closing line back to the first vertex.
4. Press the key **`s`** to save your calibrated coordinates to [`data/zone_config.json`](file:///Users/atharvarathi/Desktop/Projects/Major-Pro/data/zone_config.json).
5. To cancel without saving changes, press **`q`**.

---

## 5. Daily Operation & Execution Workflows

### 5.1 Generating the Master 21-Minute Processed Demo Video
To process ~21 minutes of footage from `input/sample.mp4` (~18,900 frames at 14.97 FPS) into the official 2×2 grid video:

```bash
python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --duration-minutes 21 --width 1280
```

#### Running in Background (Recommended for Long Video Runs):
You can safely launch the 21-minute processing job in the background and follow its progress:
```bash
nohup python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --duration-minutes 21 --width 1280 > processing_21min.log 2>&1 &
tail -f processing_21min.log
```

---

### 5.2 Real-Time Live Monitoring Mode (with GUI Window)
If you are operating at a security desk and wish to view the live video processing in real-time on your screen:

```bash
python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4 --display
```
- A window displaying the 2×2 Grid ($2560\times 1440$) will open.
- Press **`q`** at any time while the window is focused to safely stop processing and save all generated outputs up to that frame.

---

### 5.3 Single CCTV Live HUD View Mode
If you prefer a single full-screen view of the enhanced CCTV stream with tactical HUD boxes (without the 2×2 grid):

```bash
python run_processor.py --video input/sample.mp4 --output results/cctv_hud.mp4 --view hud --display
```

---

### 5.4 Extracting Official University Test Case Clips
To automatically extract and verify all 9 video test cases (`TC_02` to `TC_10`) into high-resolution clips:

```bash
python extract_test_cases.py
```
*Outputs are saved to:* [`final_research_paper_material/test_cases/`](file:///Users/atharvarathi/Desktop/Projects/Major-Pro/final_research_paper_material/test_cases/)

---

### 5.5 Command-Line Parameter Reference

| Parameter | Default | Choices / Format | Description |
| :--- | :---: | :---: | :--- |
| `--video` | `input/sample.mp4` | File path | Path to the raw surveillance video file or RTSP stream URL |
| `--output` | `results/demo_dual.mp4` | File path | Destination path for the processed `.mp4` video |
| `--zone` | `data/zone_config.json` | File path | Path to the calibrated perimeter polygon JSON |
| `--view` | `grid` | `grid`, `hud`, `full` | Visualization layout mode: 2×2 comparison grid (`grid`) or single full CCTV HUD (`hud`) |
| `--duration-minutes` | `None` | Float (e.g., `21`) | Automatically calculates and processes the exact number of frames corresponding to the given duration |
| `--max-frames` | `None` | Integer (e.g., `500`) | Explicit limit on the maximum number of frames to process |
| `--start-frame` | `0` | Integer | Starting frame index in the source video |
| `--width` | `1280` | `1280`, `1920` | Processing width per panel ($1280$ yields a $2560\times 1440$ 2×2 canvas) |
| `--display` | `False` | Flag | Displays an interactive live OpenCV playback window on the desktop |

---

## 6. Operator Guide: Interpreting the 2×2 HUD & Threat Telemetry

### 6.1 Understanding the 2×2 Grid Layout

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│              QUADRANT 1               │              QUADRANT 2               │
│        Baseline Tracking (RF)         │  Strong Baseline (Physics + XGBoost)  │
│                                       │                                       │
│ • Raw CCTV Surveillance Feed          │ • Full Live Tactical HUD Overlay      │
│ • Baseline Random Forest Bounding Box │ • Dynamic Color-Coded Intent Boxes    │
│ • Bottom-Center Foot-Point Markers    │ • BPD, TE, Vt, Vn Telemetry Cards     │
│ • Static Zone Polygon Boundary        │ • Visual Probing Tether Lines         │
├───────────────────────────────────────┼───────────────────────────────────────┤
│              QUADRANT 3               │              QUADRANT 4               │
│       Sequence Analysis (LSTM)        │   Dynamic Legend & Live Telemetry     │
│                                       │                                       │
│ • PyTorch Temporal Sequence Tracking  │ • 6-Class Intent Color Palette Legend │
│ • 15-Frame Comet Trajectory Trails    │ • Real-time Processing FPS & Latency  │
│ • Velocity Directional Vectors        │ • Active Track Count & Threat Levels  │
│ • Motion Smoothness Profiling         │ • Frame Counter & Timestamp Clock     │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

---

### 6.2 The 6 Behavioral Intent Classes

Every individual tracked in the camera's field of view is categorized into one of 6 distinct states:

| Color | Intent Classification | Operator Action / Threat Level | Description |
| :---: | :--- | :---: | :--- |
| 🟢 | **Casual Transit** | **Normal** (Risk $< 30\%$) | Subject is walking far from the perimeter along normal pedestrian paths. No action required. |
| 🟡 | **Approaching Boundary** | **Notice** (Risk $30\text{--}55\%$) | Subject has turned and is moving directly toward the restricted zone boundary ($V_n > 0$). Monitor closely. |
| 🟠 | **Active Scouting** | **Warning** (Risk $60\text{--}80\%$) | Subject is walking parallel to the fence ($< 50\text{ px}$), casing the perimeter, or displaying repeated approach-retreat probing. |
| 🟠 | **Suspicious Loitering** | **Warning** (Risk $55\text{--}75\%$) | Subject is lingering near the zone, moving at very low speed, stopping repeatedly, or showing high erratic path entropy ($TE > 0.60$). |
| 🟣 | **Fleeing** | **Urgent** (Risk $70\text{--}90\%$) | Subject is moving rapidly away from the perimeter at high speed ($> 100\text{ px/s}$) following an approach or breach. |
| 🔴 | **CRITICAL INTRUSION** | **EMERGENCY** (Risk $= 100\%$) | **Physical Breach:** Ground foot-point has crossed into the restricted zone. Immediate dispatch required. |

---

### 6.3 Reading the Telemetry Cards

Above each person's bounding box, the HUD renders an intuitive glassmorphic telemetry card:

```
┌────────────────────────────────────────────────────────┐
│  ID: 104  │  ACTIVE SCOUTING  │  RISK: 74%             │  <-- Header Pill
├────────────────────────────────────────────────────────┤
│  BPD: 0.82  │  TE: 0.68  │  Vt: 3.4  │  Vn: +0.2 px/f  │  <-- Physics Dynamics
├────────────────────────────────────────────────────────┤
│  VECTOR: [PARALLEL]  │  CYCLES: 2  │  DIST: 38 px      │  <-- Perimeter Context
└────────────────────────────────────────────────────────┘
```

- **ID:** The unique identifier assigned to the person by the tracker. Persists through temporary occlusions up to 15 frames.
- **Risk Score:** Combined threat probability from $0\%$ to $100\%$.
- **BPD (Boundary Probing Dynamics):** Ranges from `0.00` to `1.00`. Values $> 0.50$ indicate repeated boundary interactions or sustained casing.
- **TE (Trajectory Entropy):** Ranges from `0.00` to `1.00`. Values $< 0.35$ mean purposeful straight-line walking; values $> 0.60$ signify erratic wandering or pacing.
- **Vt (Tangential Velocity):** Speed moving parallel to the perimeter fence. High $V_t$ close to the boundary indicates perimeter casing.
- **Vn (Normal Velocity):** Speed moving towards or away from the boundary:
  - **Positive ($+V_n$):** Inward movement closing in on the zone.
  - **Negative ($-V_n$):** Outward movement retreating from the zone.
- **Action Vector:** Simplified direction indicator: `[Closing In]`, `[Moving Away]`, `[Parallel]`, or `[Neutral]`.

---

### 6.4 Visual Alarms & Visual Cues

```
                    Boundary Polygon Edge
            ──────────────────●──────────────────
                              ▲
                             /
        Dynamic Amber       /  Pulsing Contact Point
        Tether Line        /
                          /
                    ●────/──────────●
                    │   /           │
                    │  ● (Foot-pt)  │  <-- ID: 104
                    │               │      ! PROBING DETECTED !
                    ●───────────────●
```

1. **Repeated Probing Tether:** When a subject repeatedly approaches and retreats from the perimeter, an amber tether line connects the subject's feet directly to the closest contact point on the fence line, marked with a pulsing dot and an on-screen `! PROBING DETECTED !` banner.
2. **Velocity Arrows:** A colored arrow projects from the subject's center of mass, showing their instantaneous heading and velocity magnitude.
3. **Comet Trails:** A 15-frame fading history trail visualizes the subject's recent trajectory path.
4. **Critical Intrusion Alert:** The bounding box instantly snaps to bright red with thick tactical corner brackets, and a `CRITICAL INTRUSION` banner flashes immediately upon boundary penetration.

---

## 7. Incident Forensics & Telemetry Log Review

Whenever video processing runs, a structured CSV log is automatically generated alongside the video:
- Video file: [`results/demo_dual.mp4`](file:///Users/atharvarathi/Desktop/Projects/Major-Pro/results/demo_dual.mp4)
- Telemetry log: [`results/demo_dual_telemetry.csv`](file:///Users/atharvarathi/Desktop/Projects/Major-Pro/results/demo_dual_telemetry.csv)

### How to Use the CSV for Forensic Reports:
1. Open `results/demo_dual_telemetry.csv` in **Microsoft Excel**, **Google Sheets**, or **Python/Pandas**.
2. **Filter for Incursions:** Filter the column `is_inside == True` or `intent == 'CRITICAL INTRUSION'` to find the exact frame index and timestamp of boundary violations.
3. **Filter for Pre-Intrusion Probing:** Filter `repeated_probing == True` or `bpd_score > 0.60` to review individuals who scouted the facility before the breach.
4. **Identify Fleeing Suspects:** Filter `intent == 'Fleeing'` and `speed > 10.0` to locate subjects retreating rapidly from the scene.

---

## 8. University Test Cases Reference

The system has been verified against all 10 official **Somaiya Vidyavihar University** test cases:

| Test Case | Scenario | Expected Outcome | Verification Clip Location |
| :---: | :--- | :--- | :--- |
| **TC_01** | Interactive polygon definition | Coordinates match UI; saved to JSON | Tested in `tests/test_physics_advanced.py` |
| **TC_02** | Target $< 50\text{ px}$ moving parallel | Intent = **Active Scouting**; Box = **Orange** | `final_research_paper_material/test_cases/TC_02_Active_Scouting.mp4` |
| **TC_03** | Target foot-point enters zone | Intent = **CRITICAL INTRUSION**; Box = **Red** | `final_research_paper_material/test_cases/TC_03_Critical_Intrusion.mp4` |
| **TC_04** | Occlusion for 15 frames | Track ID persists without cache loss | `final_research_paper_material/test_cases/TC_04_Track_Persistence.mp4` |
| **TC_05** | Normal walking far from zone | Intent = **Casual Transit**; Box = **Green** | `final_research_paper_material/test_cases/TC_05_Casual_Transit.mp4` |
| **TC_06** | Moving towards zone | Intent = **Approaching Boundary**; Box = **Yellow** | `final_research_paper_material/test_cases/TC_06_Approaching_Boundary.mp4` |
| **TC_07** | Moving slowly / stopping repeatedly | Intent = **Suspicious Loitering**; Box = **Orange** | `final_research_paper_material/test_cases/TC_07_Suspicious_Loitering.mp4` |
| **TC_08** | Incursion after scouting | Point-in-polygon override to **CRITICAL INTRUSION** | `final_research_paper_material/test_cases/TC_08_Scouting_To_Intrusion.mp4` |
| **TC_09** | Rapid retreat ($> 100\text{ px/s}$) | Intent = **Fleeing**; Box = **Purple** | `final_research_paper_material/test_cases/TC_09_Fleeing.mp4` |
| **TC_10** | Multiple simultaneous targets | Independent track IDs, independent metrics | `final_research_paper_material/test_cases/TC_10_Multi_Target.mp4` |

---

## 9. Troubleshooting & FAQ

### Q1: The processor says "Headless environment detected; continuing purely in video output mode".
**Cause:** You ran with `--display` on a remote server (via SSH) without an active X11/Wayland display or Quartz window server.  
**Solution:** The software handles this automatically by continuing to process the video and saving the complete `.mp4` file to `results/demo_dual.mp4` without crashing.

### Q2: Why is the video processing slower than 15 FPS?
**Solution:** Ensure hardware acceleration is engaged:
- On Apple Silicon Mac, confirm `mps` is enabled (it is selected automatically).
- On Windows/Linux with NVIDIA, verify CUDA is installed (`torch.cuda.is_available()` returns `True`).
- You can process at `--width 1280` (default) rather than 4K for maximum throughput.

### Q3: An individual outside the zone was marked as "CRITICAL INTRUSION".
**Cause:** The subject's ground foot-point $(x, y_2)$ touched the red polygon boundary.  
**Solution:** Run `python map_zone.py` to recalibrate the perimeter polygon and align the vertices strictly with your physical fence or boundary line.

### Q4: Can I run this system on live RTSP CCTV camera feeds?
**Answer:** Yes. Simply pass the camera RTSP stream URL to the `--video` parameter:
```bash
python run_processor.py --video "rtsp://admin:password@192.168.1.100:554/stream1" --output results/live_output.mp4
```
