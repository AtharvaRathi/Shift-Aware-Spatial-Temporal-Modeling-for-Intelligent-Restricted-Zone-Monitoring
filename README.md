# Gate v2: Advanced Intrusion Detection & Intent Prediction System

> **Research upgrade** — adds LSTM sequential trajectory modeling alongside the XGBoost/RF baseline, generates a split-screen demonstration video, and produces all graphs needed for a research paper submission.

---

## Hardware Configurations

| Setting | Your Laptop (RTX 4050) | HPC Lab (RTX A5000) |
|---------|----------------------|---------------------|
| VRAM | 6 GB | **24 GB** |
| RAM | 16 GB | **128 GB** |
| CPU | Mobile Core i7 | **Xeon W-2265 (12C/24T)** |
| YOLO Model | yolov8n (auto) | **yolov8x (auto)** |
| Dataset Size | 10,000 samples | **50,000 samples** |
| LSTM Hidden Size | 128 | **256** |
| LSTM Layers | 2 | **3** |
| Batch Size | 256 | **1024** |
| Training Epochs | 40 | **60** |
| DataLoader Workers | 4 | **12** |
| Processing Width | 1280 px | **1920 px (Full HD)** |

> All of these parameters are **auto-detected at runtime** based on available VRAM. No manual changes needed.

---

## Quick Start (HPC / Any Machine)

### 1 — Clone & Setup Environment

```bash
git clone https://github.com/NightCrawler909/Gate.git
cd Gate

python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

### 2 — Install Dependencies

```bash
# Install core packages
pip install -r requirements.txt

# Install PyTorch with CUDA (match your CUDA version — check with: nvcc --version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Place Your Video

```
Gate/
└── input/
    └── sample.mp4      ← PUT YOUR VIDEO HERE
```

### 4 — Map the Restricted Zone (One-Time Setup)

Run this on a machine with a display (your laptop) **before** pushing to HPC:

```bash
python map_zone.py
```

Click on the video frame to draw the restricted polygon. Press `q` to save.
This creates `data/zone_config.json` — commit it to the repo so HPC picks it up automatically.

### 5 — Run the One-Click Pipeline

```bash
python run_pipeline.py
```

That's it. The script will automatically:
1. Generate training datasets (tabular + sequence)
2. Train Random Forest, XGBoost, and LSTM models
3. Process your video → split-screen demo video
4. Generate all research paper graphs

---

## Output Files

| File | Description |
|------|-------------|
| `results/demo_dual.mp4` | Split-screen demo video (XGBoost vs LSTM side-by-side) |
| `results/demo_dual_telemetry.xlsx` | Per-frame telemetry spreadsheet |
| `results/model_benchmark.csv` | Accuracy / Precision / Recall / F1 benchmark table |
| `results/graphs/01_confusion_matrices.png` | Confusion matrices for all 3 models |
| `results/graphs/02_model_comparison_bar.png` | Bar chart comparison |
| `results/graphs/03_roc_curves.png` | Multi-class ROC curves |
| `results/graphs/04_accuracy_vs_time.png` | Accuracy vs training time scatter |
| `results/graphs/05_radar_chart.png` | Radar chart model comparison |

---

## Advanced Options

```bash
# Quick local test — only process first 500 frames
python run_pipeline.py --max-frames 500

# Custom video path
python run_pipeline.py --video /path/to/your/video.mp4

# Skip retraining (use existing saved models)
python run_pipeline.py --skip-train

# Change processing resolution (default 1280)
python run_pipeline.py --width 960
```

---

## Architecture

```
run_pipeline.py             ← One-click master script
│
├── ml/train_all.py         ← Trains RF + XGBoost + LSTM
│   ├── ml/generate_dataset.py   ← Generates tabular & sequence data
│   ├── ml/anomaly_detector.py   ← Random Forest model class
│   └── ml/lstm_detector.py      ← PyTorch LSTM model class
│
├── run_processor.py        ← Dual-inference video processor
│   ├── core/detector.py         ← YOLOv8 + ByteTrack wrapper
│   └── core/zone_manager.py     ← Polygon zone math
│
└── ml/generate_graphs.py   ← Generates all 300 DPI paper figures
```

---

## Models

| Model | Type | Input | Research Role |
|-------|------|-------|---------------|
| **Random Forest** | Tree Ensemble | Single-frame snapshot (7 features) | Baseline |
| **XGBoost** | Gradient Boosting | Single-frame snapshot (7 features) | Strong Baseline |
| **LSTM** | Recurrent NN | 20-frame trajectory sequence (7×20) | Proposed Method |

**Research Hypothesis:** Sequential LSTM modeling of trajectory history outperforms static snapshot classifiers for pre-breach behavioral intent prediction.

---

## Project Structure

```
Gate/
├── input/                  ← PUT YOUR VIDEO HERE (sample.mp4)
├── core/                   ← Computer Vision & Tracking
│   ├── detector.py
│   └── zone_manager.py
├── ml/                     ← Machine Learning pipeline
│   ├── generate_dataset.py
│   ├── train_all.py
│   ├── anomaly_detector.py
│   ├── lstm_detector.py
│   ├── generate_graphs.py
│   └── data/               ← Saved models & datasets (auto-generated)
├── data/
│   └── zone_config.json    ← Zone polygon (commit this!)
├── results/                ← All outputs (auto-generated)
│   └── graphs/
├── run_pipeline.py         ← ONE-CLICK MASTER SCRIPT
├── run_processor.py        ← Video processing engine
├── map_zone.py             ← Zone drawing tool
└── requirements.txt
```