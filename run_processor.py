"""
run_processor.py
────────────────
Dual-inference video processor.
Reads an input video, runs YOLO+ByteTrack detection,
predicts intent via both Random Forest (baseline) and LSTM (sequence model),
and renders a clean split-screen demonstration video.

Usage:
    python run_processor.py --video input/sample.mp4 --output results/demo_dual.mp4
"""
import cv2
import numpy as np
import argparse
import warnings

# Suppress all sklearn and xgboost warnings during inference to prevent terminal I/O bottleneck
warnings.filterwarnings("ignore")
import json
import os
import torch
import time
import argparse
import sys
import pandas as pd
import math
from collections import deque
from threading import Thread
from queue import Queue
from tqdm import tqdm

from core.detector import PersonDetector
from core.zone_manager import ZoneManager
from core.test_case_manager import TestCaseManager
from ml.anomaly_detector import AnomalyDetector
from ml.xgboost_detector import XGBoostDetector
from ml.risk_scorer import RiskScorer
from ml.lstm_detector import LSTMDetector

# ─── Intent → Color map ─────────────────────────────────────────────────────
INTENT_COLOR = {
    'Casual Transit':       (80,  200, 80),
    'Approaching Boundary': (0,   220, 220),
    'Active Scouting':      (0,   165, 255),
    'Suspicious Loitering': (0,   140, 255),
    'CRITICAL INTRUSION':   (30,  30,  255),
    'Fleeing':              (200, 50,  200),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def resolve_intent(base_class, is_inside, action_vector, speed, dist, risk_score):
    if is_inside:
        return 'CRITICAL INTRUSION'
    if base_class == 'active_intrusion' or (action_vector == 'Closing In' and dist < 60):
        return 'Approaching Boundary'
    if base_class == 'scouting' or (dist < 80 and speed < 25 and action_vector == 'Parallel'):
        return 'Active Scouting'
    if base_class == 'loitering' or (speed < 8 and action_vector == 'Neutral' and dist < 120):
        return 'Suspicious Loitering'
    if action_vector == 'Moving Away' and speed > 90:
        return 'Fleeing'
    return 'Casual Transit'


def draw_hud(frame, track_id, intent, color, x1, y1, x2, y2, trail_pts=None):
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Only draw labels for non-casual events (reduces clutter)
    if intent != 'Casual Transit':
        label = f'ID:{track_id} | {intent}'
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, theight), _ = cv2.getTextSize(label, font, fs, th)
        ty = max(20, y1 - 8)
        cv2.rectangle(frame, (x1, ty - theight - 4), (x1 + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(frame, label, (x1 + 3, ty), font, fs, color, th, cv2.LINE_AA)

    # Trajectory trail (comet tail)
    if trail_pts and len(trail_pts) > 2:
        pts = np.array(trail_pts, np.int32)
        for i in range(1, len(pts)):
            alpha = int(255 * (i / len(pts)))
            c = tuple(int(v * alpha / 255) for v in color)
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), c, 3, cv2.LINE_AA)


def draw_zone(frame, polygon_pts, opacity=0.3):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_pts], (0, 0, 200))
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    cv2.polylines(frame, [polygon_pts], True, (0, 0, 255), 2)


def draw_panel_title(frame, text, color=(255,255,255)):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (10, 10, 10), -1)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

class ThreadedCamera:
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
                import time
                time.sleep(0.01)

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        return True, self.q.get()

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

class ThreadedWriter:
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
                import time
                time.sleep(0.01)

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.stopped = True
        self.t.join()
        self.out.release()

def process_video(video_path, output_path, zone_config='data/zone_config.json',
                  target_width=1280, max_frames=None, start_frame=0):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    detector      = PersonDetector()
    zone_manager  = ZoneManager(config_path=zone_config)
    rf_detector   = AnomalyDetector()
    xgb_detector  = XGBoostDetector()
    lstm_detector = LSTMDetector()
    risk_scorer   = RiskScorer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    total_f -= start_frame
    if max_frames:
        total_f = min(total_f, max_frames)

    # Scale to target_width for speed
    scale = target_width / src_w
    proc_w = target_width
    proc_h = int(src_h * scale)
    # Make even
    proc_w = proc_w if proc_w % 2 == 0 else proc_w - 1
    proc_h = proc_h if proc_h % 2 == 0 else proc_h - 1

    # Zone polygon needs rescaling too
    zone_pts_raw = zone_manager.polygon.reshape(-1, 2).astype(np.float32)
    zone_pts_raw *= scale
    zone_pts = zone_pts_raw.astype(np.int32).reshape((-1, 1, 2))

    # Output: 2x2 grid -> each panel is proc_w x proc_h
    out = ThreadedWriter(output_path, fps, (proc_w * 2, proc_h * 2))

    track_history = {}     # track_id -> deque(maxlen=20)
    telemetry     = []
    frames_done   = 0
    
    tc_manager = TestCaseManager(fps, 'final_research_paper_material/test_cases')

    print(f"  Processing {total_f} frames at {proc_w}x{proc_h} (2x2 grid) ...")
    
    # Swap cap for ThreadedCamera to start buffering
    threaded_cap = ThreadedCamera(cap, start_frame=start_frame)
    
    # Pre-render the completely static Legend panel to save CPU allocations
    pre_rendered_legend = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    cv2.putText(pre_rendered_legend, 'Legend & Telemetry', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    y_pos = 150
    for intent, color in INTENT_COLOR.items():
        cv2.rectangle(pre_rendered_legend, (50, y_pos-25), (90, y_pos+15), color, -1)
        cv2.putText(pre_rendered_legend, intent, (120, y_pos+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2, cv2.LINE_AA)
        y_pos += 60
    
    pbar = tqdm(total=total_f, unit='frame')

    while True:
        ret, frame = threaded_cap.read()
        if not ret or (max_frames and frames_done >= max_frames):
            break

        frame = cv2.resize(frame, (proc_w, proc_h))
        results = detector.track(frame, persist=True)
        
        # Pre-render zone on the base frame ONCE to save memory blending
        draw_zone(frame, zone_pts)

        # Base panels (copying the pre-rendered frame)
        panel_rf   = frame.copy()
        panel_xgb  = frame.copy()
        panel_lstm = frame.copy()
        
        frame_telemetry = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes    = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)

                is_inside   = zone_manager.is_inside(foot)
                dist_to_zone = 0.0 if is_inside else zone_manager.distance_to_polygon(foot)

                if tid not in track_history:
                    track_history[tid] = deque(maxlen=20)

                history = track_history[tid]

                speed        = 0.0
                action_vector = 'Neutral'
                if len(history) > 0:
                    old = history[0]
                    fp  = frames_done - old['frame']
                    if fp > 0:
                        dx = foot[0] - old['position'][0]
                        dy = foot[1] - old['position'][1]
                        speed = (math.hypot(dx, dy) / fp) * fps
                        delta = dist_to_zone - old['distance']
                        if delta < -5:
                            action_vector = 'Closing In'
                        elif delta > 5:
                            action_vector = 'Moving Away'
                        else:
                            action_vector = 'Parallel'
                        # Velocity arrow on panels
                        ep = (foot[0] + int(dx*2), foot[1] + int(dy*2))
                        cv2.arrowedLine(panel_rf,   foot, ep, (200,200,200), 2, tipLength=0.3)
                        cv2.arrowedLine(panel_xgb,  foot, ep, (200,200,200), 2, tipLength=0.3)
                        cv2.arrowedLine(panel_lstm, foot, ep, (200,200,200), 2, tipLength=0.3)

                features = {
                    'speed_variance':              speed,
                    'distance_to_restricted_zone': dist_to_zone,
                    'time_near_boundary':          float(sum(1 for r in history if r['distance'] < 80)),
                    'time_inside_zone':            float(sum(1 for r in history if r['distance'] <= 0)),
                    'sudden_stops':                int(speed < 6),
                    'carrying_baggage':            0,
                    'time_of_day_multiplier':      1.0,
                }

                history.append({'frame': frames_done, 'position': foot,
                                 'distance': dist_to_zone, 'features': features})

                risk = min(risk_scorer.calculate_risk(tid, features), 100.0)

                # ── RF inference (single frame)
                rf_base    = rf_detector.predict_intent(features)
                intent_rf  = resolve_intent(rf_base, is_inside, action_vector, speed, dist_to_zone, risk)

                # ── XGBoost inference (single frame)
                xgb_base   = xgb_detector.predict_intent(features)
                intent_xgb = resolve_intent(xgb_base, is_inside, action_vector, speed, dist_to_zone, risk)

                # ── LSTM inference (sequence)
                seq         = [r['features'] for r in history]
                lstm_base   = lstm_detector.predict_intent(seq)
                intent_lstm = resolve_intent(lstm_base, is_inside, action_vector, speed, dist_to_zone, risk)

                color_rf   = INTENT_COLOR.get(intent_rf,   (200,200,200))
                color_xgb  = INTENT_COLOR.get(intent_xgb,  (200,200,200))
                color_lstm = INTENT_COLOR.get(intent_lstm, (200,200,200))

                trail = [r['position'] for r in history]

                # Draw panels
                draw_hud(panel_rf,   tid, intent_rf,   color_rf,   x1, y1, x2, y2)
                draw_hud(panel_xgb,  tid, intent_xgb,  color_xgb,  x1, y1, x2, y2)
                
                show_trail = intent_lstm != 'Casual Transit'
                draw_hud(panel_lstm, tid, intent_lstm, color_lstm, x1, y1, x2, y2,
                         trail_pts=trail if show_trail else None)
                
                tel_record = {
                    'frame': frames_done,
                    'timestamp_s': round(frames_done / fps, 2),
                    'track_id': tid,
                    'rf_intent': intent_rf,
                    'xgb_intent': intent_xgb,
                    'lstm_intent': intent_lstm,
                    'speed_px_s': round(speed, 2),
                    'dist_to_zone': round(dist_to_zone, 2),
                    'risk_score': round(risk, 2),
                    'action_vector': action_vector,
                }
                telemetry.append(tel_record)
                frame_telemetry.append(tel_record)

        draw_panel_title(panel_rf,   'Baseline (Random Forest)')
        draw_panel_title(panel_xgb,  'Strong Baseline (XGBoost)')
        draw_panel_title(panel_lstm, 'Proposed Method (LSTM Sequence)')
        
        top_row = cv2.hconcat([panel_rf, panel_xgb])
        bottom_row = cv2.hconcat([panel_lstm, pre_rendered_legend])
        grid = cv2.vconcat([top_row, bottom_row])
        
        out.write(grid)
        tc_manager.process_frame(grid, frame_telemetry)
        
        frames_done += 1
        pbar.update(1)

    pbar.close()
    threaded_cap.release()
    out.release()
    tc_manager.release()

    # Save telemetry
    tel_path = output_path.replace('.mp4', '_telemetry.xlsx')
    if telemetry:
        pd.DataFrame(telemetry).to_excel(tel_path, index=False)
        print(f"  Telemetry -> {tel_path}")

    print(f"  [OK] Video saved -> {output_path}  ({frames_done} frames)")
    return frames_done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  default='input/sample.mp4')
    parser.add_argument('--output', default='results/demo_dual.mp4')
    parser.add_argument('--zone',   default='data/zone_config.json')
    parser.add_argument('--width',  type=int, default=1920,
                        help='Processing width. HPC default: 1920. Laptop: use 1280.')
    parser.add_argument('--start-frame', type=int, default=0, help='Frame number to start processing from')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    process_video(args.video, args.output, args.zone, args.width, args.max_frames, args.start_frame)
