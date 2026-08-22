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
import json
import os
import torch
import time
import pandas as pd
import math
from collections import deque
from tqdm import tqdm

from core.detector import PersonDetector
from core.zone_manager import ZoneManager
from ml.anomaly_detector import AnomalyDetector
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


def draw_legend(frame):
    """Draws a bottom legend strip."""
    items = [
        ('Casual Transit',       (80, 200, 80)),
        ('Approaching',          (0, 220, 220)),
        ('Scouting/Loitering',   (0, 165, 255)),
        ('INTRUSION',            (30, 30, 255)),
        ('Fleeing',              (200, 50, 200)),
    ]
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 36), (w, h), (15, 15, 15), -1)
    x = 10
    for label, color in items:
        cv2.rectangle(frame, (x, h - 28), (x + 14, h - 14), color, -1)
        cv2.putText(frame, label, (x + 18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1, cv2.LINE_AA)
        x += len(label) * 8 + 40


# ─── Main ────────────────────────────────────────────────────────────────────
def process_video(video_path, output_path, zone_config='data/zone_config.json',
                  target_width=1280, max_frames=None):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    detector      = PersonDetector()
    zone_manager  = ZoneManager(config_path=zone_config)
    rf_detector   = AnomalyDetector()
    lstm_detector = LSTMDetector()
    risk_scorer   = RiskScorer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    # Output: side-by-side -> each panel is proc_w x proc_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (proc_w * 2, proc_h))

    track_history = {}     # track_id -> deque(maxlen=20)
    telemetry     = []
    frames_done   = 0

    print(f"  Processing {total_f} frames at {proc_w}x{proc_h} ...")
    pbar = tqdm(total=total_f, unit='frame')

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frames_done >= max_frames):
            break

        frame = cv2.resize(frame, (proc_w, proc_h))
        results = detector.track(frame, persist=True)

        # Base panels
        panel_rf   = frame.copy()
        panel_lstm = frame.copy()
        draw_zone(panel_rf,   zone_pts)
        draw_zone(panel_lstm, zone_pts)

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
                        # Velocity arrow on both panels
                        ep = (foot[0] + int(dx*2), foot[1] + int(dy*2))
                        cv2.arrowedLine(panel_rf,   foot, ep, (200,200,200), 2, tipLength=0.3)
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

                # ── LSTM inference (sequence)
                seq         = [r['features'] for r in history]
                lstm_base   = lstm_detector.predict_intent(seq)
                intent_lstm = resolve_intent(lstm_base, is_inside, action_vector, speed, dist_to_zone, risk)

                color_rf   = INTENT_COLOR.get(intent_rf,   (200,200,200))
                color_lstm = INTENT_COLOR.get(intent_lstm, (200,200,200))

                trail = [r['position'] for r in history]

                # Draw RF panel
                draw_hud(panel_rf,   tid, intent_rf,   color_rf,   x1, y1, x2, y2)

                # Draw LSTM panel with comet tail only for non-casual
                show_trail = intent_lstm != 'Casual Transit'
                draw_hud(panel_lstm, tid, intent_lstm, color_lstm, x1, y1, x2, y2,
                         trail_pts=trail if show_trail else None)

                telemetry.append({
                    'frame': frames_done,
                    'timestamp_s': round(frames_done / fps, 2),
                    'track_id': tid,
                    'rf_intent': intent_rf,
                    'lstm_intent': intent_lstm,
                    'speed_px_s': round(speed, 2),
                    'dist_to_zone': round(dist_to_zone, 2),
                    'risk_score': round(risk, 2),
                    'action_vector': action_vector,
                })

        draw_panel_title(panel_rf,   'Baseline (Random Forest)')
        draw_panel_title(panel_lstm, 'Sequence Model (LSTM)')
        draw_legend(panel_rf)
        draw_legend(panel_lstm)

        dual = cv2.hconcat([panel_rf, panel_lstm])
        out.write(dual)
        frames_done += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

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
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    process_video(args.video, args.output, args.zone, args.width, args.max_frames)
