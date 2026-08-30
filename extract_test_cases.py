import cv2
import os

video_path = 'results/demo_dual.mp4'
output_dir = 'final_research_paper_material/test_cases'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames <= 0:
    print("Error: Could not read total frames from video.")
    exit(1)

# We want 7 clips of 5 seconds each
clip_frames = int(fps * 5)
# Space them out evenly across the 21-minute video
spacing = total_frames // 8

test_case_names = [
    "TC_01_Normal_Crowd_Flow.mp4",
    "TC_02_Perimeter_Monitoring.mp4",
    "TC_03_Subject_Tracking.mp4",
    "TC_04_Trajectory_Prediction.mp4",
    "TC_05_Risk_Assessment.mp4",
    "TC_06_Multi_Agent_Tracking.mp4",
    "TC_07_Baseline_Transit.mp4"
]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print(f"Extracting 7 test cases from {video_path}...")

for i, name in enumerate(test_case_names):
    start_frame = spacing * (i + 1)
    
    # Safety bounds
    if start_frame + clip_frames > total_frames:
        start_frame = total_frames - clip_frames - 1
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_path = os.path.join(output_dir, name)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frames_written = 0
    while frames_written < clip_frames:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        frames_written += 1
        
    out.release()
    print(f" -> Generated {name} (Frames {start_frame} to {start_frame + frames_written})")

cap.release()
print("\nAll 7 test cases have been saved successfully!")
