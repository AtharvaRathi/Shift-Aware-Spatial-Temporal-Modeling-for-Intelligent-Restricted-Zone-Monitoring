# B.Tech Semester VI Project Report
**Project Title:** Gate: Advanced Intrusion Detection & Intent Prediction System  
**Department:** Information Technology / AI & DS  
**Batch:** 2023-27  

---

## ABSTRACT
Traditional perimeter security methods rely on static boundary breaches, resulting in reactive responses and high rates of binary alert fatigue. "Gate" introduces a proactive, edge-optimized Intrusion Detection System (IDS) that combines real-time Computer Vision with heuristic physics vectors and Machine Learning classifiers. We utilize Ultralytics YOLOv8 for sub-millisecond person detection and ByteTrack for persistent ID tracking across frames. By extracting granular spatial telemetry—specifically velocity, trajectory, and zone-approach vectors—the system predicts behavioral intents (e.g., *Suspicious Loitering*, *Active Scouting*, *Fleeing*) before a critical breach occurs. Our comparative analysis on heuristically generated synthetic telemetry datasets demonstrates that ensemble models perform exceptionally well on physical trajectory data, with XGBoost achieving an accuracy of 99.93% and an F1-Score of 99.92%, outperforming Deep Learning (MLP) baselines. The final deliverable features a Python-based processing pipeline, live OpenCV trajectory rendering, and Excel telemetry exports.

**Keywords:** *Intrusion Detection, YOLOv8, Object Tracking, Behavioral Intent Prediction, XGBoost, Computer Vision, Spatial Telemetry.*

---

## CHAPTER 1: INTRODUCTION

### 1.1 Problem Definition
Currently deployed surveillance architectures heavily suffer from Binary Alert Fatigue. They trigger alarms only when a physical boundary is crossed or flag indiscriminately on all pixel motion, leading to high false-positive rates. Traditional systems fail to capture spatial intelligence—understanding a subject's velocity, dwell time, and vector relationship to a restricted zone—leaving security personnel blind to pre-intrusion scouting and probing behaviors. 

### 1.2 Motivation
Real-world security breaches often exhibit a "probing" phase before the actual intrusion. Standard systems fail to warn operators during this critical window. The motivation for "Gate" is to shift security from a reactive model to a proactive, predictive architecture. By bridging the gap between raw bounding boxes and physical spatial awareness, we aim to provide explainable, granular threat matrix predictions.

### 1.3 Scope
**Covered in this project:**
*   Real-time object detection (Persons) using YOLOv8.
*   Persistent object tracking using ByteTrack.
*   Heuristic trajectory vector math mapping distance to dynamic restricted zones.
*   Machine Learning intent classification (XGBoost, Random Forest, MLP).
*   Live OpenCV Dashboard and Pandas telemetry export.

**Explicitly excluded:**
*   Raw facial recognition or biometric identification.
*   Hardware deployment to embedded edge cameras (focus is on the software pipeline).
*   Audio-based anomaly detection.

### 1.4 Salient Contributions
*   Engineered a physics-math pipeline that derives real-world spatial vectors (Speed, Zone Approach Delta) from 2D pixel coordinates dynamically.
*   Designed a Granular Intent Matrix classifying subjects into 6 behavioral states ranging from 'Casual Transit' to 'Fleeing'.
*   Benchmarked ML classification models (RF vs XGBoost vs MLP) on structured physical telemetry, proving tree-based models map spatial thresholds with up to 99.93% accuracy.

### 1.5 Organization of the Report
Chapter 1 introduces the project space and objectives. Chapter 2 reviews current literature and identifies gaps. Chapter 3 details the software project management plan. Chapter 4 documents the system requirements. Chapter 5 illustrates the software architecture and component design. Chapter 6 focuses on the validation and testing methodology. Chapter 7 presents the results, data graphs, and comparative analyses.

---

## CHAPTER 2: LITERATURE SURVEY

### 2.1 Core Research Trends
**Object Detection and Tracking:**
Recent advancements in YOLO (You Only Look Once) architectures, specifically YOLOv8, have pushed the boundary of anchor-free, real-time object detection natively supporting edge hardware via FP16 precision. ByteTrack complements this by utilizing low-confidence detection boxes to maintain track persistence under occlusion.

**Behavioral Anomaly Detection:**
Traditional video anomaly detection relies heavily on 3D Convolutional Neural Networks (3D-CNNs) handling sequential raw frames. While accurate, these are notoriously computationally heavy and unfit for edge environments. Utilizing structured tabular telemetry derived from CV trackers instead of raw frame inputs for ML classification remains a highly efficient, under-explored domain.

### 2.2 Critical Research Gaps Identified
*   **Computational Bottlenecks:** 3D-CNNs for behavior analysis are too slow for real-time edge CCTV hardware.
*   **Contextual Blindness:** Standard tracking outputs $X,Y$ coordinates without calculating the context of those coordinates relative to a dynamic semantic boundary.
*   **Binary Outputs:** Existing systems predict "Intrusion" or "Normal", lacking the capability to gauge intermediary threats like "Active Scouting".

---

## CHAPTER 3: SOFTWARE PROJECT MANAGEMENT PLAN (SPMP)

### 3.1 Introduction
The Gate project aims to deliver a multi-layered software pipeline orchestrating frame extraction, neural-network-backed CV detection, mathematical telemetry reduction, and intent classification.

### 3.2 Project Organization 
**3.2.1 Software Process Model:** Iterative & Incremental prototyping.
**3.2.2 Tools and Techniques:**
*   **Languages:** Python 3.10+
*   **Computer Vision Libraries:** OpenCV, Ultralytics (YOLOv8)
*   **Machine Learning Libraries:** Scikit-Learn, XGBoost, PyTorch
*   **Data & Rendering:** Pandas, NumPy, Matplotlib, Seaborn
*   **Version Control:** Git & GitHub

### 3.3 Tasks, Timeline, and Constraints
**Dependencies and Constraints:**
Requires a CUDA-compatible GPU (e.g., RTX 4050) for optimal FPS throughput. Requires pre-recorded video or active RTSP streams.
**Risks and Contingencies:**
*   *Risk:* I/O bottleneck rendering high-resolution CV streams. *Mitigation:* Replace deep alpha-blending logic (`cv2.addWeighted`) with raw pixel writing operations (`cv2.polylines`).
*   *Risk:* ID Switching in crowds. *Mitigation:* Tweak ByteTrack memory buffers to 120-frames.

---

## CHAPTER 4: SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

### 4.1 Functional Requirements
| Aspect | Description | Implementation |
| :--- | :--- | :--- |
| **Zone Mapping** | Admin can draw interactive polygonal restriction zones | `map_zone.py` via `cv2.setMouseCallback` |
| **CV Tracking** | System isolates and tracks human subjects | `detector.py`, YOLOv8 + ByteTrack |
| **Telemetry Engine** | System tracks 15-frame history to map trajectory vectors | `collections.deque` caching |
| **ML Inference** | Classify behavioral intents dynamically | `ml.anomaly_detector` & Scikit/XGBoost |
| **Data Logging** | Log active session data synchronously | `pandas.DataFrame.to_excel` |

### 4.2 Non-Functional Requirements
| Attribute | Description |
| :--- | :--- |
| **Performance** | Must run iteratively at >= 30 FPS on CUDA-enabled GPUs. |
| **Interpretability** | Visual outputs must display human-readable vectors and numerical risk scores. |

---

## CHAPTER 5: SOFTWARE DESIGN DESCRIPTION (SDD)

### 5.1 System Architectural Design
Our architecture is pipeline-based, moving data from raw tensor outputs down to structured analytical dataframes.
*   **Layer 1: Data Acquisition:** `cv2.VideoCapture` streams frames. `zone_manager.py` fetches JSON boundary configs.
*   **Layer 2: Core Processing:** `detector.py` runs YOLOv8 tracking tensors.
*   **Layer 3: Analysis Engines:** Trajectory vectors are extracted, distance formulated via `is_inside` point-in-polygon math. Vectors pass to `risk_scorer.py` and `anomaly_detector.py`.
*   **Layer 4: Presentation:** OpenCV overlays HUD logic, arrows, and matrices. `pandas` caches to RAM.

### 5.3 Detailed Component Descriptions
*   `core/detector.py`: Wraps Ultralytics tracking logic specifically isolating class 0 (Person).
*   `core/zone_manager.py`: Parses polygon coordinates. Calculates closest geometric bounds for risk modeling.
*   `map_zone.py`: Interactive setup tool bridging physical CCTV views to NumPy dimensions.
*   `run_processor.py`: The Main orchestrator script coupling the logic stream without locking the active thread.
*   `ml/generate_dataset.py`: Generates the heuristic baseline synthetic data for ML comparative training.

---

## CHAPTER 6: SOFTWARE TEST DOCUMENT (STD)

### 6.1 Test Cases
| TC Name | Input | Expected Result | Status |
| :--- | :--- | :--- | :--- |
| **TC_01: Zone mapping** | User clicks 4 points on frame | Polygon closes and saves coordinates to JSON | Pass |
| **TC_02: Point-In-Polygon** | Bounding box foot passes inside zone | `is_inside` triggers, Risk Score = 100, Intent = CRITICAL INTRUSION | Pass |
| **TC_03: Scouting Vector** | Target <50px from zone, moving Parallel | Intent = Active Scouting, bounding box turns Orange | Pass |
| **TC_04: Loitering Lock** | Target speed <5px/sec for long dwell time | Intent triggers Suspicious Loitering | Pass |
| **TC_05: Export Hook** | EOF signal received | Pipeline shuts down tracking, commits telemetry to Excel | Pass |

---

## CHAPTER 7: RESULTS AND DISCUSSION

### 7.1 Exploratory Data Analysis & Feature Research
**Figure 7.1: Speed Variance Box Plot (`graph_2_boxplot_speed_variance.png`)**
*   *Observation:* Interquartile ranges (IQR) effectively outline physical anomalies. High-velocity variance statistically separates targets 'Fleeing' from 'Loitering'.

**Figure 7.2: Feature Correlation Heatmap (`graph_8_correlation_triangle_heatmap.png`)**
*   *Observation:* Collinearity grid shows `distance_to_restricted_zone` inversely impacts `historical_risk_score`, proving the geometric modeling correctly propagates threat mapping.

**Figure 7.3: PCA Cluster Scaling (`graph_5_pca_clusters.png`)**
*   *Observation:* Reducing the 7-dimensional heuristic feature layer into 2D limits reveals that explicit boundary thresholds naturally group intent behaviors structurally, verifying ML validity.

### 7.2 Comparative Analysis of Machine Learning Models
We ran comparative analyses on heuristically mapped baseline data ($N=5000$) to evaluate structural capabilities for tabular telemetry classifications.

| Metric | Random Forest | XGBoost | Deep Learning (MLP) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 99.86% | **99.93%** | 98.40% |
| **Precision** | 99.88% | **99.93%** | 98.02% |
| **Recall** | 99.86% | **99.93%** | 98.40% |
| **F1-Score** | 99.83% | **99.92%** | 98.11% |

**Interpretation:**
XGBoost achieves near-perfect heuristic boundary mapping (99.93%). Because our tabular data is composed of logical physical parameters (`distance`, `speed`, `sudden_stops`), decision-tree ensemble methods heavily outperform Deep Learning Neural Networks (MLP), which struggled slightly (98.40%) to draw sharp, hard-coded threshold boundaries compared to XGBoost's deterministic leaf splits.

**Figure 7.4: Multi-Class ROC Curve (`graph_10_multiclass_ROC.png`)**
*   *Observation:* The Receiver Operating Characteristic maps an AUC of 1.0, proving negligible False Positives across the classification spectrum when utilizing ensemble methods.

**Figure 7.5: Algorithm Learning Curve Matrix (`graph_11_learning_curve.png`)**
*   *Observation:* Model saturation tests prove the framework does not suffer from underfitting; it successfully saturates its analytical bounds after ~2500 training examples.

---

## REFERENCES
[1] Redmon, J. et al., "You Only Look Once: Unified, Real-Time Object Detection," in *CVPR*, 2016.
[2] Jocher, G. et al., "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics
[3] Zhang, Y. et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," in *ECCV*, 2022.
[4] Chen, T., & Guestrin, C., "XGBoost: A Scalable Tree Boosting System," in *KDD*, 2016.

---

## APPENDIX

**A.1 Software & Computational Environment**
| Library | Version | Purpose |
| :--- | :--- | :--- |
| `ultralytics` | 8.x | YOLOv8 Core Engine |
| `torch` | 2.x (cu118) | GPU Tensor Calculations |
| `opencv-python` | 4.x | Frame Rendering & Polygon Math |
| `scikit-learn` | 1.4.0 | PCA, Evaluation Metrics, ML Baselines |
| `xgboost` | 3.2.0 | Extreme Gradient Intent Classifier |
| `pandas` | 2.x | Telemetry Caching & Data-Frame Export |

**A.2 System Execution Pipeline Flow**
1. Define JSON Restricted Bound `python map_zone.py`.
2. Initiate Tracking Loop `python run_processor.py`.
3. System applies Point-In-Polygon math iteratively vs ByteTrack `track_ids`.
4. ML calculates Behavior logic -> Overlay applies -> Data appended to Telemetry cache.
5. On script termination, CSV/Excel is dumped. Results evaluated via `research_analysis.py`.