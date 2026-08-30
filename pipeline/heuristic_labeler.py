import pandas as pd
import argparse
from tqdm import tqdm

def label_track(group):
    """
    Applies heuristic rules to determine the true intent of a track based on its entire trajectory.
    Returns the integer class label.
    """
    max_time_inside = group['time_inside_zone'].max()
    max_time_near = group['time_near_boundary'].max()
    max_stops = group['sudden_stops'].max()
    max_speed = group['speed_variance'].max()
    min_dist = group['distance_to_restricted_zone'].min()
    
    # Rule 1: CRITICAL INTRUSION (5)
    # If they ever step inside the zone
    if max_time_inside > 0 or min_dist <= 0:
        return 5
        
    # Rule 2: Fleeing (4)
    # If they hit very high speeds
    if max_speed > 300: # Assuming pixels/sec, adjust based on video resolution
        return 4
        
    # Rule 3: Active Scouting (3)
    # They spend a long time near the boundary and make several stops
    if max_time_near > 150 and max_stops > 10: # Assuming 30fps -> 5 seconds near
        return 3
        
    # Rule 4: Suspicious Loitering (2)
    # They don't approach the boundary much, but stay completely still for a long time
    if max_stops > 300: # Assuming 30fps -> 10 seconds still
        return 2
        
    # Rule 5: Approaching Boundary (1)
    # They get close to the boundary but never enter
    if min_dist < 50:
        return 1
        
    # Rule 6: Casual Transit (0)
    # Everyone else just passing by in the background
    return 0

def auto_label_dataset(input_csv, output_csv):
    print(f"Loading raw extracted telemetry from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if df.empty:
        print("Dataset is empty. Exiting.")
        return
        
    print(f"Applying Heuristic Auto-Labeler to {len(df['track_id'].unique())} tracks...")
    
    # Determine the label for each track
    track_labels = {}
    for track_id, group in tqdm(df.groupby('track_id')):
        track_labels[track_id] = label_track(group)
        
    # Map labels back to all frames for that track
    df['intent_class'] = df['track_id'].map(track_labels)
    
    # Save the labeled dataset
    df.to_csv(output_csv, index=False)
    
    # Print label distribution
    dist = df.groupby('intent_class')['track_id'].nunique()
    print("\nLabel Distribution (Number of Unique Tracks):")
    print(dist)
    
    print(f"\nSaved labeled dataset to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input raw CSV path")
    parser.add_argument('--output', type=str, required=True, help="Output labeled CSV path")
    args = parser.parse_args()
    
    auto_label_dataset(args.input, args.output)
