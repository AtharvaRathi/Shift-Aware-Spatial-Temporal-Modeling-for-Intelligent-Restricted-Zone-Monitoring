import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "Casual Transit",
    "Approaching Boundary",
    "Suspicious Loitering",
    "Active Scouting",
    "Fleeing",
    "CRITICAL INTRUSION"
]

FEATURE_COLUMNS = [
    "speed_variance",
    "distance_to_restricted_zone",
    "time_near_boundary",
    "time_inside_zone",
    "sudden_stops",
    "carrying_baggage",
    "time_of_day_multiplier"
]

SEQUENCE_LENGTH = 15

def generate_synthetic_tracks(num_tracks=2000, max_frames=80):
    """
    Generates track-aware synthetic surveillance data.
    Each track corresponds to a single person's trajectory over time.
    """
    np.random.seed(42)
    rows = []
    
    # Class probabilities
    probs = [0.4, 0.15, 0.15, 0.1, 0.05, 0.15]
    intents = np.random.choice(range(6), size=num_tracks, p=probs)
    
    for track_id, intent in enumerate(intents):
        # Base features for this person
        carrying = np.random.choice([0, 1], p=[0.8, 0.2])
        tod = np.random.uniform(0.8, 2.0)
        
        # Initial states
        if intent == 0: # Casual
            dist = np.random.uniform(150, 400)
            speed_mu = 15
        elif intent == 1: # Approaching
            dist = np.random.uniform(100, 250)
            speed_mu = 12
        elif intent == 2: # Loitering
            dist = np.random.uniform(50, 150)
            speed_mu = 3
        elif intent == 3: # Scouting
            dist = np.random.uniform(30, 80)
            speed_mu = 6
        elif intent == 4: # Fleeing
            dist = np.random.uniform(10, 50)
            speed_mu = 35
        else: # Intrusion
            dist = np.random.uniform(40, 120)
            speed_mu = 20

        time_near = 0
        time_inside = 0
        stops = 0
        
        # Track duration
        track_len = np.random.randint(SEQUENCE_LENGTH + 5, max_frames)
        
        for t in range(track_len):
            # Dynamic updates
            if intent == 0: # Casual
                dist = max(50, dist - np.random.normal(5, 5))
                speed = max(0, np.random.normal(speed_mu, 3))
            elif intent == 1: # Approaching
                dist = max(10, dist - np.random.normal(10, 3))
                speed = max(0, np.random.normal(speed_mu, 4))
                if dist < 60: time_near += 1
            elif intent == 2: # Loitering
                dist = max(10, dist + np.random.normal(0, 5))
                speed = max(0, np.random.normal(speed_mu, 2))
                if dist < 120: time_near += 1
                if speed < 2: stops += 1
            elif intent == 3: # Scouting
                dist = max(5, dist + np.random.normal(0, 8))
                speed = max(0, np.random.normal(speed_mu, 3))
                if dist < 80: time_near += 1
                if speed < 4: stops += 1
            elif intent == 4: # Fleeing
                dist += np.random.normal(20, 5) # moving away
                speed = max(0, np.random.normal(speed_mu, 10))
            elif intent == 5: # Intrusion
                dist = dist - np.random.normal(15, 4)
                speed = max(0, np.random.normal(speed_mu, 5))
                if dist < 60: time_near += 1
                if dist <= 0:
                    dist = 0
                    time_inside += 1
            
            # Record frame
            rows.append({
                'track_id': track_id,
                'frame_idx': t,
                'speed_variance': speed,
                'distance_to_restricted_zone': dist,
                'time_near_boundary': time_near,
                'time_inside_zone': time_inside,
                'sudden_stops': stops,
                'carrying_baggage': carrying,
                'time_of_day_multiplier': tod,
                'intent_class': intent
            })
            
    df = pd.DataFrame(rows)
    return df

def track_aware_split(df, test_size=0.15, val_size=0.15):
    """
    Splits the dataset strictly by track_id to prevent data leakage.
    Returns Train, Val, Test DataFrames.
    """
    track_ids = df['track_id'].unique()
    
    # If no test set is requested (e.g., when splitting real train data for early stopping)
    if test_size == 0.0:
        train_ids, val_ids = train_test_split(track_ids, test_size=val_size, random_state=42)
        test_ids = []
    else:
        # First split: Train vs Temp (Val + Test)
        train_ids, temp_ids = train_test_split(track_ids, test_size=(test_size + val_size), random_state=42)
        
        # Second split: Val vs Test
        ratio = test_size / (test_size + val_size)
        val_ids, test_ids = train_test_split(temp_ids, test_size=ratio, random_state=42)
    
    train_df = df[df['track_id'].isin(train_ids)].copy()
    val_df = df[df['track_id'].isin(val_ids)].copy()
    test_df = df[df['track_id'].isin(test_ids)].copy()
    
    return train_df, val_df, test_df

class SequenceDataset(Dataset):
    """
    PyTorch Dataset that generates sliding window sequences from track DataFrames.
    Shape: (N, SEQUENCE_LENGTH, num_features)
    Target: Label of the final frame in the sequence window.
    """
    def __init__(self, df, seq_length=SEQUENCE_LENGTH):
        self.X = []
        self.y = []
        
        # Group by track to prevent sequences crossing different tracks
        for track_id, group in df.groupby('track_id'):
            group = group.sort_values('frame_idx')
            features = group[FEATURE_COLUMNS].values
            labels = group['intent_class'].values
            
            if len(group) >= seq_length:
                for i in range(len(group) - seq_length + 1):
                    window_x = features[i:i+seq_length]
                    window_y = labels[i+seq_length-1] # target is the label at end of window
                    self.X.append(window_x)
                    self.y.append(window_y)
                    
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def prepare_data_pipeline(data_dir='research_framework/data', regenerate=True, use_real_data=False):
    """
    Master function to generate tracks, split them, and save everything cleanly.
    If use_real_data is True, it will attempt to load real pre-labeled video data.
    """
    splits_dir = os.path.join(data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    if use_real_data:
        print("Loading REAL WORLD telemetry data...")
        train_path = os.path.join(data_dir, 'raw', 'real_train.csv')
        test_path = os.path.join(data_dir, 'raw', 'real_test.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Real data CSVs not found! Please run the pipeline/ extraction scripts first.")
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # We don't have a distinct validation video, so we split train_df for early stopping
        print("Splitting real train data into Train and Validation for early stopping...")
        train_df, val_df, _ = track_aware_split(train_df, test_size=0.0, val_size=0.2)
        
        train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
        
        print(f"Real Data splits generated. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} frames.")
        return train_df, val_df, test_df
        
    if regenerate:
        print("Generating track-aware synthetic dataset...")
        df = generate_synthetic_tracks()
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        df.to_csv(os.path.join(data_dir, 'raw', 'all_tracks.csv'), index=False)
        
        train_df, val_df, test_df = track_aware_split(df)
        
        train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
        
        print(f"Data splits generated. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} frames.")
        return train_df, val_df, test_df
    else:
        train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
        return train_df, val_df, test_df

if __name__ == '__main__':
    # Test execution
    train, val, test = prepare_data_pipeline('../research_framework/data')
