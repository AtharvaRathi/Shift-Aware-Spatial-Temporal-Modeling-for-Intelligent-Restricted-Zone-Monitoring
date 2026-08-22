import pandas as pd
import numpy as np
import os

INTENT_PARAMS = {
    'passing_by':       dict(speed_mu=15, speed_sd=4, dist_mu=350, dist_sd=100, time_near_mu=3,  time_near_sd=5,  stops_mu=0, stops_sd=0.5),
    'loitering':        dict(speed_mu=2,  speed_sd=1.5,dist_mu=80,  dist_sd=40,  time_near_mu=45, time_near_sd=20, stops_mu=2, stops_sd=1.5),
    'scouting':         dict(speed_mu=5,  speed_sd=2,  dist_mu=30,  dist_sd=20,  time_near_mu=70, time_near_sd=25, stops_mu=4, stops_sd=2),
    'active_intrusion': dict(speed_mu=28, speed_sd=8,  dist_mu=5,   dist_sd=10,  time_near_mu=15, time_near_sd=10, stops_mu=1, stops_sd=1),
}
PROBS = [0.55, 0.25, 0.12, 0.08]
CLASSES = list(INTENT_PARAMS.keys())


def generate_tabular_dataset(num_rows=10000, output_path='ml/data/tabular_dataset.csv'):
    """Generates a flat, tabular (snapshot) dataset for RF/XGBoost training."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    intents = np.random.choice(CLASSES, size=num_rows, p=PROBS)
    rows = []
    for intent in intents:
        p = INTENT_PARAMS[intent]
        carrying_baggage = np.random.choice([0, 1], p=[0.8, 0.2])
        time_of_day = np.random.uniform(0.8, 2.0)
        inside = (intent == 'active_intrusion') and (np.random.random() < 0.6)

        row = {
            'speed_variance':              max(0, np.random.normal(p['speed_mu'], p['speed_sd'])),
            'distance_to_restricted_zone': max(0, np.random.normal(p['dist_mu'],  p['dist_sd'])) if not inside else 0,
            'time_near_boundary':          max(0, np.random.normal(p['time_near_mu'], p['time_near_sd'])),
            'time_inside_zone':            max(0, np.random.normal(10, 5)) if inside else 0,
            'sudden_stops':                max(0, int(np.random.normal(p['stops_mu'], p['stops_sd']))),
            'carrying_baggage':            carrying_baggage,
            'time_of_day_multiplier':      time_of_day,
            'intent_class':                intent,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"  [Dataset] Tabular: {len(df)} rows -> {output_path}")
    return output_path


def generate_sequence_dataset(num_samples=10000, seq_length=20, output_dir='ml/data'):
    """Generates temporal sequence data for LSTM training. Shape: (N, seq_length, features)."""
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
    intents = np.random.choice(CLASSES, size=num_samples, p=PROBS)
    num_features = 7
    X = np.zeros((num_samples, seq_length, num_features))
    y = np.array([CLASS_MAP[i] for i in intents])

    for i, intent in enumerate(intents):
        p = INTENT_PARAMS[intent]
        carrying = np.random.choice([0, 1], p=[0.8, 0.2])
        tod = np.random.uniform(0.8, 2.0)

        dist = np.random.normal(p['dist_mu'], p['dist_sd'])
        time_near, time_inside, stops = 0, 0, 0

        for t in range(seq_length):
            if intent == 'passing_by':
                dist = max(0, dist - np.random.normal(12, 4))
                speed = max(0, np.random.normal(p['speed_mu'], p['speed_sd']))
            elif intent == 'loitering':
                dist = max(10, dist + np.random.normal(0, 8))
                speed = max(0, np.random.normal(p['speed_mu'], p['speed_sd']))
                if dist < 100: time_near += 1
                if speed < 4: stops += 1
            elif intent == 'scouting':
                dist = max(5, dist - np.random.normal(1.5, 1))
                speed = max(0, np.random.normal(p['speed_mu'], p['speed_sd']))
                if dist < 50: time_near += 1
                if speed < 3: stops += 1
            elif intent == 'active_intrusion':
                dist = max(0, dist - np.random.normal(22, 6))
                speed = max(0, np.random.normal(p['speed_mu'], p['speed_sd']))
                if dist < 50: time_near += 1
                if dist <= 0: time_inside += 1

            X[i, t] = [
                max(0, speed + np.random.normal(0, 1.5)),
                max(0, dist  + np.random.normal(0, 8)),
                time_near,
                time_inside,
                stops,
                carrying,
                tod,
            ]

    # Shuffle
    idx = np.random.permutation(num_samples)
    X, y = X[idx], y[idx]

    np.save(os.path.join(output_dir, 'X_seq.npy'), X)
    np.save(os.path.join(output_dir, 'y_seq.npy'), y)
    print(f"  [Dataset] Sequences: {X.shape} -> {output_dir}")
    return os.path.join(output_dir, 'X_seq.npy'), os.path.join(output_dir, 'y_seq.npy')


if __name__ == '__main__':
    generate_tabular_dataset()
    generate_sequence_dataset()