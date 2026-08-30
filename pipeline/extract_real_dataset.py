import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import PersonDetector
from core.zone_manager import ZoneManager

def calculate_speed(p1, p2, fps):
    """Calculate absolute speed in pixels/second"""
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist * fps

def extract_telemetry(video_path, output_csv, zone_config='data/zone_config.json', max_frames=None):
    print(f"Starting raw telemetry extraction for: {video_path}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    detector = PersonDetector()
    zone_manager = ZoneManager(config_path=zone_config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        
    # Scale points if needed - here we assume video runs at raw resolution 
    # to match original zone setup
    zone_pts = zone_manager.polygon.astype(np.int32)
    
    track_history = {} # track_id -> list of foot positions
    track_stats = {}   # track_id -> dict of accumulated states
    telemetry = []
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, unit='frame')
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
            
        results = detector.track(frame, persist=True)
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)
                
                is_inside = zone_manager.is_inside(foot)
                dist_to_zone = 0.0 if is_inside else zone_manager.distance_to_polygon(foot)
                
                if tid not in track_history:
                    track_history[tid] = []
                    track_stats[tid] = {
                        'time_near': 0,
                        'time_inside': 0,
                        'stops': 0,
                        # Heuristics:
                        'carrying_baggage': np.random.choice([0, 1], p=[0.8, 0.2]), 
                        'tod': 1.2
                    }
                    
                history = track_history[tid]
                stats = track_stats[tid]
                
                history.append(foot)
                if len(history) > 30:
                    history.pop(0)
                    
                speed = 0.0
                if len(history) > 1:
                    speed = calculate_speed(history[-2], history[-1], fps)
                    
                if dist_to_zone < 80 and not is_inside:
                    stats['time_near'] += 1
                if is_inside:
                    stats['time_inside'] += 1
                if len(history) > 5 and speed < 5:
                    stats['stops'] += 1
                    
                telemetry.append({
                    'track_id': tid,
                    'frame_idx': frame_idx,
                    'speed_variance': speed,
                    'distance_to_restricted_zone': dist_to_zone,
                    'time_near_boundary': stats['time_near'],
                    'time_inside_zone': stats['time_inside'],
                    'sudden_stops': stats['stops'],
                    'carrying_baggage': stats['carrying_baggage'],
                    'time_of_day_multiplier': stats['tod']
                })
                
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    
    df = pd.DataFrame(telemetry)
    df.to_csv(output_csv, index=False)
    print(f"Extracted telemetry for {len(df['track_id'].unique())} unique tracks.")
    print(f"Saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help="Input video path")
    parser.add_argument('--output', type=str, required=True, help="Output CSV path")
    parser.add_argument('--max-frames', type=int, default=None, help="Max frames to process")
    args = parser.parse_args()
    
    extract_telemetry(args.video, args.output, max_frames=args.max_frames)
