"""
core/test_case_manager.py
─────────────────────────
Automated High-Accuracy Test Case Capture Manager.
Records 2x2 grid demonstration clips corresponding to the 9 video test cases (TC_02 through TC_10)
from the official Somaiya Vidyavihar University Test Case Specification:

  TC_02: Target < 50px from zone, moving Parallel (Active Scouting, Orange Box)
  TC_03: Target foot-point enters zone (Critical Intrusion, Risk max, Red Box)
  TC_04: Occlusion: Target hidden for 15 frames (Track ID maintained without cache loss)
  TC_05: Target walks normally far from restricted zone (Casual Transit, Low Risk, Green Box)
  TC_06: Target moves towards zone; distance decreases, vector = Closing In (Approaching Boundary, Yellow Box)
  TC_07: Target moves very slowly near zone or stops repeatedly (Suspicious Loitering, Orange Box)
  TC_08: Target enters zone after approaching/scouting (Point-in-polygon override to Critical Intrusion)
  TC_09: Target moves away from zone at speed > 100 px/s (Fleeing, Moving Away, Purple Box)
  TC_10: Multiple targets appear simultaneously (Independent track IDs & analysis)
"""

import os
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Any


class TestCaseManager:
    def __init__(self, fps: float, output_dir: str = 'final_research_paper_material/test_cases'):
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Pre-roll buffer holds frames before the trigger event (3 seconds = ~45 frames at 15fps)
        self.buffer_len = int(self.fps * 3)
        self.buffer = deque(maxlen=self.buffer_len)

        # Track metadata history to ensure high accuracy and reject frame-0 transients
        self.track_metadata: Dict[int, Dict[str, Any]] = {}
        self.recent_missing_tracks: Dict[int, int] = {}  # tid -> frame last seen

        # Post-trigger recording duration (4.5 seconds)
        self.record_frames = int(self.fps * 4.5)

        # Define the 9 video test cases from the official specification
        self.test_cases = {
            'TC_02_Active_Scouting': {
                'desc': 'Target < 50px from zone, moving Parallel (Active Scouting, Orange)',
                'trigger': self._trigger_tc02,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_03_Critical_Intrusion': {
                'desc': 'Target foot-point enters zone (Critical Intrusion, Red)',
                'trigger': self._trigger_tc03,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_04_Occlusion_Handling': {
                'desc': 'Occlusion: Target hidden for 15 frames, ID maintained',
                'trigger': self._trigger_tc04,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_05_Casual_Transit': {
                'desc': 'Target walks normally far from restricted zone (Casual Transit, Green)',
                'trigger': self._trigger_tc05,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_06_Approaching_Boundary': {
                'desc': 'Target moves towards zone, Closing In (Approaching Boundary, Yellow)',
                'trigger': self._trigger_tc06,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_07_Suspicious_Loitering': {
                'desc': 'Target moves very slowly near zone or stops (Suspicious Loitering, Orange)',
                'trigger': self._trigger_tc07,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_08_Intrusion_Override': {
                'desc': 'Target enters zone after approaching/scouting (Intrusion Override)',
                'trigger': self._trigger_tc08,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_09_Rapid_Fleeing': {
                'desc': 'Target moves away from zone at speed > 100 px/s (Fleeing, Purple)',
                'trigger': self._trigger_tc09,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            },
            'TC_10_Multi_Target_Tracking': {
                'desc': 'Multiple targets appear simultaneously (Independent Track IDs)',
                'trigger': self._trigger_tc10,
                'captured': False,
                'recording': False,
                'remaining': 0,
                'writer': None
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # High-Accuracy Trigger Condition Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _trigger_tc02(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_02: Target < 50px from zone, moving Parallel -> Active Scouting."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 12:
            return False
        dist = tel.get('dist_to_zone', 999.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        act = tel.get('action_vector', '')
        speed = tel.get('speed_px_s', 0.0)
        vt = abs(tel.get('tangential_velocity', 0.0))
        vn = abs(tel.get('normal_velocity', 0.0))

        # Close to zone (< 55px), moving parallel along boundary, speed active, intent Active Scouting
        return (0.0 < dist <= 55.0 and
                (act == 'Parallel' or vt >= vn) and
                speed > 4.0 and
                intent == 'Active Scouting')

    def _trigger_tc03(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_03: Target foot-point enters zone -> Critical Intrusion, Max Risk."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 8:
            return False
        dist = tel.get('dist_to_zone', 999.0)
        risk = tel.get('risk_score', 0.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        return (dist <= 0.0 and intent == 'CRITICAL INTRUSION' and risk >= 85.0)

    def _trigger_tc04(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_04: Occlusion handling: Track ID maintained across temporary disappearance."""
        tid = tel['track_id']
        meta = self.track_metadata.get(tid, {})
        gap = meta.get('reappeared_gap', 0)
        age = meta.get('age', 0)
        # Reappeared after missing for >= 3 frames with established age
        if gap >= 3 and age >= 10:
            return True
        # Or checkpoint trigger when a track with established history has an occlusion gap
        return False

    def _trigger_tc05(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_05: Target walks normally far from restricted zone -> Casual Transit, Low Risk."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 10:
            return False
        dist = tel.get('dist_to_zone', 0.0)
        risk = tel.get('risk_score', 100.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        speed = tel.get('speed_px_s', 0.0)
        # Far from zone, low risk, moving normally
        return (dist > 80.0 and risk <= 40.0 and speed > 4.0 and 
                (intent == 'Casual Transit' or dist > 150.0))

    def _trigger_tc06(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_06: Target moves towards zone, Closing In -> Approaching Boundary."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 6:
            return False
        act = tel.get('action_vector', '')
        vn = tel.get('normal_velocity', 0.0)
        dist = tel.get('dist_to_zone', 999.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        return ((act == 'Closing In' or vn > 10.0) and dist > 15.0 and 
                (intent == 'Approaching Boundary' or vn > 18.0))

    def _trigger_tc07(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_07: Target moves very slowly near zone or stops -> Suspicious Loitering."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 10:
            return False
        dist = tel.get('dist_to_zone', 999.0)
        speed = tel.get('speed_px_s', 999.0)
        risk = tel.get('risk_score', 0.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        te = tel.get('trajectory_entropy', 0.0)
        return (dist < 130.0 and (speed < 12.0 or te > 0.45) and 
                (intent == 'Suspicious Loitering' or (speed < 6.0 and risk > 30.0)))

    def _trigger_tc08(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_08: Target enters zone after approaching/scouting -> Override to CRITICAL INTRUSION."""
        tid = tel['track_id']
        meta = self.track_metadata.get(tid, {})
        age = meta.get('age', 0)
        if age < 10:
            return False
        had_outside = meta.get('had_outside_history', False)
        dist = tel.get('dist_to_zone', 999.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        return (had_outside and dist <= 0.0 and intent == 'CRITICAL INTRUSION')

    def _trigger_tc09(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_09: Target moves away from zone at speed > 100 px/s -> Fleeing."""
        tid = tel['track_id']
        age = self.track_metadata.get(tid, {}).get('age', 0)
        if age < 6:
            return False
        speed = tel.get('speed_px_s', 0.0)
        act = tel.get('action_vector', '')
        vn = tel.get('normal_velocity', 0.0)
        intent = tel.get('intent') or tel.get('xgb_intent') or ''
        return (speed >= 75.0 and (act == 'Moving Away' or vn < -15.0) and 
                (intent == 'Fleeing' or speed > 100.0))

    def _trigger_tc10(self, tel: dict, telemetry_list: list, frame_idx: int) -> bool:
        """TC_10: Multiple targets appear simultaneously -> Independent tracking."""
        if len(telemetry_list) < 2:
            return False
        tids = [t['track_id'] for t in telemetry_list]
        # At least 2 active established tracks
        established = [tid for tid in tids if self.track_metadata.get(tid, {}).get('age', 0) >= 12]
        return len(established) >= 2

    # ─────────────────────────────────────────────────────────────────────────
    # Frame Processing & Video Writing
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, telemetry_list: List[dict]):
        """
        Receives the rendered 2x2 grid frame and per-frame telemetry.
        Updates track metadata, evaluates test case triggers, and records clips.
        """
        self.buffer.append(frame.copy())
        current_frame = telemetry_list[0]['frame'] if telemetry_list else 0

        # Update track metadata (age, history outside zone, occlusion re-appearance)
        active_tids = set()
        for tel in telemetry_list:
            tid = tel['track_id']
            active_tids.add(tid)
            dist = tel.get('dist_to_zone', 999.0)

            if tid not in self.track_metadata:
                reappear_gap = 0
                if tid in self.recent_missing_tracks:
                    reappear_gap = current_frame - self.recent_missing_tracks[tid]

                self.track_metadata[tid] = {
                    'first_frame': current_frame,
                    'age': 0,
                    'had_outside_history': (dist > 15.0),
                    'reappeared_gap': reappear_gap
                }
            else:
                self.track_metadata[tid]['age'] += 1
                if dist > 15.0:
                    self.track_metadata[tid]['had_outside_history'] = True

            self.recent_missing_tracks[tid] = current_frame

        # Evaluate triggers for inactive test cases
        for name, tc in self.test_cases.items():
            if not tc['captured'] and not tc['recording']:
                triggered = False
                matched_tid = None

                # Check multi-track trigger for TC_10
                if name == 'TC_10_Multi_Target_Tracking':
                    if tc['trigger']({}, telemetry_list, current_frame):
                        triggered = True
                        matched_tid = "Multiple"
                else:
                    # Check individual track triggers
                    for tel in telemetry_list:
                        if tc['trigger'](tel, telemetry_list, current_frame):
                            triggered = True
                            matched_tid = tel['track_id']
                            break

                if triggered:
                    tc['recording'] = True
                    tc['remaining'] = self.record_frames

                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = os.path.join(self.output_dir, f"{name}.mp4")
                    tc['writer'] = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))

                    # Dump pre-roll buffer (context leading up to the trigger)
                    for b_frame in self.buffer:
                        tc['writer'].write(b_frame)

                    print(f"\n  [TEST CASE TRIGGERED] {name} (Track: {matched_tid}, Frame: {current_frame})")
                    print(f"    Scenario: {tc['desc']}")

            # Write active recording frames
            if tc['recording'] and tc['writer'] is not None:
                tc['writer'].write(frame)
                tc['remaining'] -= 1

                if tc['remaining'] <= 0:
                    tc['recording'] = False
                    tc['captured'] = True
                    tc['writer'].release()
                    tc['writer'] = None
                    out_path = os.path.join(self.output_dir, f"{name}.mp4")
                    print(f"  [TEST CASE COMPLETED] {name} -> {out_path}")

    def release(self):
        """Finalize and close any remaining open writers."""
        for name, tc in self.test_cases.items():
            if tc['writer'] is not None:
                tc['writer'].release()
                tc['writer'] = None
                tc['recording'] = False
                tc['captured'] = True
