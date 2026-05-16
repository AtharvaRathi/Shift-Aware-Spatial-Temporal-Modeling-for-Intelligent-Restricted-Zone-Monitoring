import pandas as pd
import numpy as np

def generate_dataset(num_rows=5000, output_path='custom_dataset.csv'):
    np.random.seed(42)
    
    # Realistic probability distributions for human behavior
    intents = np.random.choice(
        ['passing_by', 'loitering', 'scouting', 'active_intrusion'], 
        size=num_rows, 
        p=[0.55, 0.25, 0.12, 0.08] # 55% normal, 25% loitering, 12% scouting, 8% invading
    )
    
    data = {
        'speed_variance': np.zeros(num_rows),
        'distance_to_restricted_zone': np.zeros(num_rows),
        'time_near_boundary': np.zeros(num_rows),
        'time_inside_zone': np.zeros(num_rows),
        'sudden_stops': np.zeros(num_rows),
        'carrying_baggage': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'time_of_day_multiplier': np.random.uniform(0.8, 2.0, num_rows)
    }
    
    scores = []
    
    # Inject Gaussian Noise and Overlaps for realistic confusion boundaries
    for i, intent in enumerate(intents):
        if intent == 'passing_by':
            data['speed_variance'][i] = max(0, np.random.normal(4.0, 1.5))
            data['distance_to_restricted_zone'][i] = max(10, np.random.normal(300, 120))
            data['time_near_boundary'][i] = max(0, np.random.normal(5, 8))
            data['time_inside_zone'][i] = 0
            data['sudden_stops'][i] = max(0, int(np.random.normal(0, 1)))
            base_score = np.random.normal(15, 10)
            
        elif intent == 'loitering':
            data['speed_variance'][i] = max(0, np.random.normal(1.2, 0.9))
            data['distance_to_restricted_zone'][i] = max(5, np.random.normal(70, 40))
            data['time_near_boundary'][i] = max(10, np.random.normal(50, 25))
            data['time_inside_zone'][i] = 0
            data['sudden_stops'][i] = max(0, int(np.random.normal(2, 2)))
            base_score = np.random.normal(50, 15)
            
        elif intent == 'scouting':
            # Scouting looks like loitering but closer with more stops
            data['speed_variance'][i] = max(0, np.random.normal(1.5, 1.2))
            data['distance_to_restricted_zone'][i] = max(1, np.random.normal(25, 20))
            data['time_near_boundary'][i] = max(20, np.random.normal(80, 30))
            data['time_inside_zone'][i] = 0
            data['sudden_stops'][i] = max(1, int(np.random.normal(5, 3)))
            base_score = np.random.normal(75, 12)
            
        elif intent == 'active_intrusion':
            data['speed_variance'][i] = max(0, np.random.normal(5.0, 2.5))
            data['distance_to_restricted_zone'][i] = max(0, np.random.normal(0, 5)) # Might slightly glitch to outside bounds due to tracking noise
            data['time_near_boundary'][i] = max(5, np.random.normal(20, 15))
            data['time_inside_zone'][i] = max(1, np.random.normal(15, 10))
            data['sudden_stops'][i] = max(0, int(np.random.normal(1, 2)))
            base_score = np.random.normal(90, 8)
            
        # Add random systemic noise to simulate real-world CCTV tracking inaccuracies
        data['speed_variance'][i] += np.random.normal(0, 0.5)
        data['distance_to_restricted_zone'][i] += np.random.normal(0, 10)
        
        # Calculate Risk Score (with natural human variance)
        final_risk = base_score * (1 + 0.15 * data['carrying_baggage'][i]) * data['time_of_day_multiplier'][i]
        final_risk = max(0.0, min(100.0, final_risk + np.random.normal(0, 8)))
        scores.append(round(final_risk, 2))
        
    df = pd.DataFrame(data)
    df['intent_class'] = intents
    df['historical_risk_score'] = scores
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated noisy realistic dataset at {output_path}")

if __name__ == '__main__':
    generate_dataset()