import os
import cv2
import numpy as np
from collections import deque

class TestCaseManager:
    def __init__(self, fps, output_dir):
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Buffer holds frames before the event happens (pre-roll)
        self.buffer = deque(maxlen=60) 
        
        # Define the 7 test cases
        self.test_cases = {
            'TC_07_Baseline_Transit': {'trigger': lambda tel: tel.get('lstm_intent') == 'Casual Transit'},
            'TC_05_Perimeter_Approach': {'trigger': lambda tel: tel.get('lstm_intent') == 'Approaching Boundary'},
            'TC_01_Active_Scouting': {'trigger': lambda tel: tel.get('lstm_intent') == 'Active Scouting'},
            'TC_03_Suspicious_Loitering': {'trigger': lambda tel: tel.get('lstm_intent') == 'Suspicious Loitering'},
            'TC_02_Point_In_Polygon': {'trigger': lambda tel: tel.get('lstm_intent') == 'CRITICAL INTRUSION'},
            'TC_06_Rapid_Fleeing': {'trigger': lambda tel: tel.get('lstm_intent') == 'Fleeing'},
            'TC_04_High_Risk_Escalation': {'trigger': lambda tel: tel.get('risk_score', 0) >= 85.0}
        }
        
        for name in self.test_cases:
            self.test_cases[name]['captured'] = False
            self.test_cases[name]['recording'] = False
            self.test_cases[name]['remaining'] = 0
            self.test_cases[name]['writer'] = None
            
    def process_frame(self, frame, telemetry_list):
        self.buffer.append(frame)
        
        # Check triggers
        for name, tc in self.test_cases.items():
            if not tc['captured'] and not tc['recording']:
                # See if any tracked object in this frame meets the criteria
                for tel in telemetry_list:
                    if tc['trigger'](tel):
                        # Trigger recording!
                        tc['recording'] = True
                        tc['remaining'] = int(self.fps * 4) # record for 4 seconds after trigger
                        
                        # Initialize writer
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_path = os.path.join(self.output_dir, f"{name}.mp4")
                        tc['writer'] = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
                        
                        # Dump buffer (pre-roll context)
                        for b_frame in self.buffer:
                            tc['writer'].write(b_frame)
                            
                        print(f"\n  [TEST CASE] Triggered {name} (Track ID: {tel['track_id']})")
                        break # Only trigger once for this test case
                        
            # If currently recording this test case, write the new frame
            if tc['recording']:
                tc['writer'].write(frame)
                tc['remaining'] -= 1
                
                if tc['remaining'] <= 0:
                    # Done recording
                    tc['recording'] = False
                    tc['captured'] = True
                    tc['writer'].release()
                    tc['writer'] = None
                    print(f"  [TEST CASE] Completed {name}")

    def release(self):
        for name, tc in self.test_cases.items():
            if tc['writer'] is not None:
                tc['writer'].release()
                tc['writer'] = None
