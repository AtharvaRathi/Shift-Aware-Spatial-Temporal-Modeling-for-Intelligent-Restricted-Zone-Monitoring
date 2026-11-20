"""
run_processor.py
────────────────
Restricted Zone Monitoring Video Processor with Advanced Physics Techniques:
1. Boundary Probing Dynamics (BPD)
2. Trajectory Entropy (TE)
3. Frenet Boundary Dynamics (FBD)

Features a real-time Live Video HUD Visualization displaying:
- Color-changing tactical bounding boxes for 6 intent classes:
    * Green:  Casual Transit
    * Yellow: Approaching Boundary
    * Orange: Suspicious Loitering / Active Scouting
    * Purple: Fleeing
    * Red:    Critical Intrusion
- Multi-field HUD cards: Track ID, Intent, Risk Score, BPD, TE, Vt, Vn
- Velocity direction arrows & trajectory comet trails
- Visual highlight for repeated boundary interaction (probing tether & banner)
- Semi-transparent restricted zone polygon overlay
- Both full-frame CCTV HUD and 2x2 comparison grid view modes
- Live interactive display and video export
"""

import os
import sys
import math
import time
import json
import argparse
import warnings
from collections import deque
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress sklearn/xgboost warnings during inference
warnings.filterwarnings("ignore")

from core.detector import PersonDetector
from core.zone_manager import ZoneManager
from core.test_case_manager import TestCaseManager
from ml.anomaly_detector import AnomalyDetector
from ml.xgboost_detector import XGBoostDetector
from ml.risk_scorer import RiskScorer
from ml.lstm_detector import LSTMDetector

# Import advanced physics techniques & HUD visualizers
from physics_advanced import (
    PhysicsTracker,
    INTENT_COLORS,
    resolve_intent_with_physics,
    draw_hud_person,
    draw_restricted_zone,
    draw_top_telemetry_bar,
    draw_hud_legend
)


class ThreadedCamera:
    """Threaded frame reader to maximize processing throughput."""
    def __init__(self, cap, start_frame=0):
        self.cap = cap
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.q = Queue(maxsize=128)
        self.stopped = False
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                self.q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        return True, self.q.get()

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()


class ThreadedWriter:
    """Threaded video writer to prevent disk I/O bottlenecks."""
    def __init__(self, out_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(out_path, fourcc, fps, size)
        self.q = Queue(maxsize=128)
        self.stopped = False
        self.t = Thread(target=self.write_frames, args=())
        self.t.daemon = True
        self.t.start()

    def write_frames(self):
        while not self.stopped or not self.q.empty():
            if not self.q.empty():
                frame = self.q.get()
                self.out.write(frame)
            else:
                time.sleep(0.005)

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.stopped = True
        self.t.join()
        self.out.release()


def draw_panel_title(frame, text, color=(255, 255, 255)):
    """Header label for grid comparison panels."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (15, 15, 20), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def process_video(
    video_path,
    output_path,
    zone_config='data/zone_config.json',
    target_width=1280,
    max_frames=None,
    start_frame=0,
    view_mode='grid',
    enable_display=False,
    duration_minutes=None
):
    """
    Main processing pipeline:
      - YOLOv8n person detection + ByteTrack tracking
      - Foot-point extraction & zone distance test
      - 15-frame rolling physics dynamics (FBD, TE, BPD)
      - 7-feature XGBoost baseline model inference
      - Physics-augmented intent classification (preserving 6 classes + intrusion override)
      - Real-time Live HUD visualization
      - Threaded video writer & telemetry CSV export
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print("═══════════════════════════════════════════════════════════════════════════")
    print("   RESTRICTED ZONE MONITORING - ADVANCED PHYSICS UPGRADE")
    print("   Techniques: Boundary Probing Dynamics | Trajectory Entropy | Frenet Dynamics")
    print(f"   Video: {video_path}  |  Output: {output_path}")
    print(f"   View:  {view_mode.upper()}  |  Display: {'ON' if enable_display else 'OFF'}")
    print("═══════════════════════════════════════════════════════════════════════════")

    # Initialize models and managers
    detector      = PersonDetector()
    zone_manager  = ZoneManager(config_path=zone_config)
    rf_detector   = AnomalyDetector()
    xgb_detector  = XGBoostDetector()
    lstm_detector = LSTMDetector()
    risk_scorer   = RiskScorer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_f -= start_frame
    if duration_minutes:
        calc_frames = int(duration_minutes * 60 * fps)
        max_frames = min(max_frames, calc_frames) if max_frames else calc_frames

    if max_frames:
        total_f = min(total_f, max_frames)

    # Resolution scaling
    scale = target_width / src_w
    proc_w = target_width
    proc_h = int(src_h * scale)
    proc_w = proc_w if proc_w % 2 == 0 else proc_w - 1
    proc_h = proc_h if proc_h % 2 == 0 else proc_h - 1

    # Scaled zone polygon
    zone_pts_raw = (zone_manager.polygon.reshape(-1, 2).astype(np.float32) * scale)
    zone_pts = zone_pts_raw.astype(np.int32).reshape((-1, 1, 2))
    zone_manager.polygon = zone_pts

    # Initialize PhysicsTracker with scaled boundary polygon (15-frame history)
    physics_tracker = PhysicsTracker(zone_pts_raw, history_len=15, fps=fps)

    # Setup Video Writer based on view_mode
    if view_mode in ['hud', 'full']:
        out_dims = (proc_w, proc_h)
    else:  # 'grid' mode (2x2)
        out_dims = (proc_w * 2, proc_h * 2)

    out = ThreadedWriter(output_path, fps, out_dims)
    tc_manager = TestCaseManager(fps, 'final_research_paper_material/test_cases')

    # Pre-render grid legend panel if in grid mode
    if view_mode == 'grid':
        pre_rendered_legend = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
        cv2.putText(pre_rendered_legend, 'Physics & Intent Legend', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos = 120
        for intent, color in INTENT_COLORS.items():
            if intent == 'Critical Intrusion':
                continue
            cv2.rectangle(pre_rendered_legend, (40, y_pos - 20), (75, y_pos + 10), color, -1)
            cv2.putText(pre_rendered_legend, intent, (95, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1, cv2.LINE_AA)
            y_pos += 45
        cv2.putText(pre_rendered_legend, 'Advanced Physics Metrics:', (40, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        y_pos += 60
        metrics_info = [
            "- BPD: Boundary Probing Dynamics score & cycles",
            "- TE:  Trajectory Directional Entropy [0.0 - 1.0]",
            "- Vt:  Tangential Velocity (parallel to boundary)",
            "- Vn:  Normal Velocity (+closing in, -retreating)"
        ]
        for m in metrics_info:
            cv2.putText(pre_rendered_legend, m, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
            y_pos += 35

    track_history = {}  # tid -> deque of 15 feature snapshots
    telemetry     = []
    frames_done   = 0

    print(f"  Processing {total_f} frames at {out_dims[0]}x{out_dims[1]} ({fps:.1f} FPS) ...")
    threaded_cap = ThreadedCamera(cap, start_frame=start_frame)
    pbar = tqdm(total=total_f, unit='frame')

    # FPS tracking
    t_start = time.time()
    fps_calc_window = deque(maxlen=30)
    display_supported = True

    while True:
        frame_time_start = time.time()
        ret, frame = threaded_cap.read()
        if not ret or (max_frames and frames_done >= max_frames):
            break

        frame = cv2.resize(frame, (proc_w, proc_h))
        results = detector.track(frame, persist=True)

        frame_telemetry = []
        counts_by_intent = {
            'Casual Transit': 0,
            'Approaching Boundary': 0,
            'Active Scouting': 0,
            'Suspicious Loitering': 0,
            'Fleeing': 0,
            'CRITICAL INTRUSION': 0,
        }

        # Track management
        active_ids = []
        detections_to_render = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            active_ids = track_ids

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                # Foot-point extraction
                foot = ((x1 + x2) // 2, y2)

                is_inside = zone_manager.is_inside(foot)
                dist_to_zone = 0.0 if is_inside else zone_manager.distance_to_polygon(foot)

                if tid not in track_history:
                    track_history[tid] = deque(maxlen=15)

                # ── 1. Update PhysicsTracker (FBD, TE, BPD) ──
                physics = physics_tracker.update_track(
                    track_id=tid,
                    foot_pt=foot,
                    dist_to_zone=dist_to_zone,
                    is_inside=is_inside,
                    frame_idx=frames_done
                )

                speed = physics['speed']
                action_vector = physics['action_vector']
                bpd_score = physics['bpd_score']
                entropy = physics['trajectory_entropy']
                vt = physics['tangential_velocity']
                vn = physics['normal_velocity']
                repeated_probing = physics['repeated_probing']
                cycles = physics['approach_retreat_cycles']

                # ── 2. Standard 7-feature input for XGBoost model ──
                features = {
                    'speed_variance':              speed,
                    'distance_to_restricted_zone': dist_to_zone,
                    'time_near_boundary':          float(sum(1 for r in track_history[tid] if r['distance'] < 80)),
                    'time_inside_zone':            float(sum(1 for r in track_history[tid] if r['distance'] <= 0)),
                    'sudden_stops':                int(speed < 6),
                    'carrying_baggage':            0,
                    'time_of_day_multiplier':      1.0,
                }

                track_history[tid].append({
                    'frame': frames_done,
                    'position': foot,
                    'distance': dist_to_zone,
                    'features': features
                })

                # ── 3. Risk Score calculation (with supplementary physics) ──
                risk_features = {
                    **features,
                    'bpd_score': bpd_score,
                    'normal_velocity': vn,
                    'trajectory_entropy': entropy,
                    'repeated_probing': repeated_probing
                }
                risk = min(risk_scorer.calculate_risk(tid, risk_features), 100.0)

                # ── 4. Model Predictions ──
                # XGBoost inference (single frame, 7 features)
                xgb_base = xgb_detector.predict_intent(features)
                # Random Forest inference
                rf_base = rf_detector.predict_intent(features)
                # LSTM sequence inference
                seq = [r['features'] for r in track_history[tid]]
                lstm_base = lstm_detector.predict_intent(seq)

                # ── 5. Intent Resolution with Physics Evidence ──
                # Preserves all 6 classes and Critical Intrusion override
                final_intent = resolve_intent_with_physics(xgb_base, is_inside, physics, risk)
                intent_rf = resolve_intent_with_physics(rf_base, is_inside, physics, risk)
                intent_lstm = resolve_intent_with_physics(lstm_base, is_inside, physics, risk)

                if final_intent in counts_by_intent:
                    counts_by_intent[final_intent] += 1
                elif 'INTRUSION' in final_intent.upper():
                    counts_by_intent['CRITICAL INTRUSION'] += 1

                detections_to_render.append({
                    'tid': tid,
                    'bbox': (x1, y1, x2, y2),
                    'intent': final_intent,
                    'rf_intent': intent_rf,
                    'xgb_intent': final_intent,
                    'lstm_intent': intent_lstm,
                    'risk': risk,
                    'physics': physics
                })

                # Telemetry log
                tel_record = {
                    'frame': frames_done,
                    'timestamp_s': round(frames_done / fps, 2),
                    'track_id': tid,
                    'intent': final_intent,
                    'risk_score': round(risk, 2),
                    'bpd_score': round(bpd_score, 3),
                    'trajectory_entropy': round(entropy, 3),
                    'tangential_velocity': round(vt, 2),
                    'normal_velocity': round(vn, 2),
                    'speed_px_s': round(speed, 2),
                    'dist_to_zone': round(dist_to_zone, 2),
                    'action_vector': action_vector,
                    'repeated_probing': repeated_probing,
                    'approach_retreat_cycles': cycles,
                    'rf_intent': intent_rf,
                    'xgb_intent': final_intent,
                    'lstm_intent': intent_lstm,
                }
                telemetry.append(tel_record)
                frame_telemetry.append(tel_record)

        # Cleanup stale tracks from physics manager
        physics_tracker.remove_stale_tracks(active_ids)

        # Live FPS calculation
        frame_time_end = time.time()
        fps_calc_window.append(max(0.001, frame_time_end - frame_time_start))
        current_fps = len(fps_calc_window) / sum(fps_calc_window)

        # ── 6. Render Output Frame ──
        if view_mode in ['hud', 'full']:
            # Base frame with zone overlay
            output_frame = frame.copy()
            draw_restricted_zone(output_frame, zone_pts, opacity=0.25)

            # Render HUD for each tracked person
            pulse_phase = frames_done * 0.1
            for d in detections_to_render:
                draw_hud_person(
                    frame=output_frame,
                    track_id=d['tid'],
                    bbox=d['bbox'],
                    intent=d['intent'],
                    risk_score=d['risk'],
                    physics=d['physics'],
                    pulse_phase=pulse_phase
                )

            # Top telemetry header and bottom legend
            draw_top_telemetry_bar(output_frame, current_fps, frames_done, counts_by_intent)
            draw_hud_legend(output_frame)

        else:
            # 2x2 Grid mode
            panel_rf   = frame.copy()
            panel_xgb  = frame.copy()
            panel_lstm = frame.copy()

            draw_restricted_zone(panel_rf, zone_pts, opacity=0.22)
            draw_restricted_zone(panel_xgb, zone_pts, opacity=0.22)
            draw_restricted_zone(panel_lstm, zone_pts, opacity=0.22)

            pulse_phase = frames_done * 0.1
            for d in detections_to_render:
                draw_hud_person(panel_rf,   d['tid'], d['bbox'], d['rf_intent'],   d['risk'], d['physics'], pulse_phase=pulse_phase)
                draw_hud_person(panel_xgb,  d['tid'], d['bbox'], d['xgb_intent'],  d['risk'], d['physics'], pulse_phase=pulse_phase)
                draw_hud_person(panel_lstm, d['tid'], d['bbox'], d['lstm_intent'], d['risk'], d['physics'], pulse_phase=pulse_phase)

            draw_panel_title(panel_rf,   'Baseline (Random Forest)')
            draw_panel_title(panel_xgb,  'Strong Baseline (Physics + XGBoost)')
            draw_panel_title(panel_lstm, 'Sequence Analysis (LSTM)')

            # Dynamic live telemetry on 4th panel
            legend_panel = pre_rendered_legend.copy()
            draw_panel_title(legend_panel, 'Physics Telemetry & Intent Legend')
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_live = proc_h - 130
            if y_live > 180:
                cv2.line(legend_panel, (30, y_live - 15), (proc_w - 30, y_live - 15), (60, 60, 70), 1)
                cv2.putText(legend_panel, 'Live System Status & Telemetry:', (35, y_live + 5), font, 0.75, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(legend_panel, f"Frame: {frames_done} | Time: {frames_done / fps:.1f}s | FPS: {current_fps:.1f}", (35, y_live + 35), font, 0.62, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(legend_panel, f"Active Tracks: {len(active_ids)} | Intrusion: {counts_by_intent.get('CRITICAL INTRUSION', 0)} | Scouting: {counts_by_intent.get('Active Scouting', 0)}", (35, y_live + 65), font, 0.62, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(legend_panel, f"Approaching: {counts_by_intent.get('Approaching Boundary', 0)} | Loitering: {counts_by_intent.get('Suspicious Loitering', 0)} | Fleeing: {counts_by_intent.get('Fleeing', 0)}", (35, y_live + 95), font, 0.62, (220, 220, 220), 1, cv2.LINE_AA)

            top_row = cv2.hconcat([panel_rf, panel_xgb])
            bottom_row = cv2.hconcat([panel_lstm, legend_panel])
            output_frame = cv2.vconcat([top_row, bottom_row])

        # Write frame to video
        out.write(output_frame)
        tc_manager.process_frame(output_frame, frame_telemetry)

        # Live Display
        if enable_display and display_supported:
            try:
                display_w = min(1280, output_frame.shape[1])
                scale_disp = display_w / output_frame.shape[1]
                display_h = int(output_frame.shape[0] * scale_disp)
                disp_frame = cv2.resize(output_frame, (display_w, display_h))
                cv2.imshow("Restricted Zone Monitoring - Live HUD", disp_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] User interrupted processing via live display.")
                    break
            except cv2.error:
                display_supported = False
                print("\n[WARN] Headless environment detected; continuing purely in video output mode.")

        frames_done += 1
        pbar.update(1)

    pbar.close()
    threaded_cap.release()
    out.release()
    tc_manager.release()
    if enable_display and display_supported:
        cv2.destroyAllWindows()

    # Save Telemetry CSV
    tel_path = output_path.replace('.mp4', '_telemetry.csv')
    if telemetry:
        df_tel = pd.DataFrame(telemetry)
        df_tel.to_csv(tel_path, index=False)
        print(f"  [OK] Physics Telemetry exported -> {tel_path} ({len(df_tel)} records)")

    elapsed_total = time.time() - t_start
    avg_fps = frames_done / max(0.01, elapsed_total)
    print(f"  [SUCCESS] Processed {frames_done} frames in {elapsed_total:.1f}s (Avg {avg_fps:.1f} FPS)")
    print(f"  [OUTPUT] Video saved -> {output_path}")
    return frames_done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Restricted Zone Perimeter Monitoring with Advanced Physics Techniques (BPD + TE + FBD)"
    )
    parser.add_argument('--video',       default='input/sample.mp4', help="Path to input video")
    parser.add_argument('--output',      default='results/demo_dual.mp4', help="Path to save processed video")
    parser.add_argument('--zone',        default='data/zone_config.json', help="Path to zone configuration JSON")
    parser.add_argument('--width',       type=int, default=1280, help="Processing frame width (1280 or 1920)")
    parser.add_argument('--view',        default='grid', choices=['grid', 'hud', 'full'],
                        help="View mode: 'grid' for 2x2 comparison with legend (default); 'hud' for single CCTV Live HUD")
    parser.add_argument('--display',     action='store_true', help="Enable live cv2.imshow playback")
    parser.add_argument('--start-frame', type=int, default=0, help="Starting frame number")
    parser.add_argument('--max-frames',  type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument('--duration-minutes', type=float, default=None,
                        help="Duration of output video in minutes (e.g. 21). Automatically calculates frames based on video FPS.")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        zone_config=args.zone,
        target_width=args.width,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        view_mode=args.view,
        enable_display=args.display,
        duration_minutes=args.duration_minutes
    )
