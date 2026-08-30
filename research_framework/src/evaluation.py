import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

CLASS_NAMES = [
    "Casual Transit",
    "Approaching Boundary",
    "Suspicious Loitering",
    "Active Scouting",
    "Fleeing",
    "CRITICAL INTRUSION"
]

def calculate_base_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], average='macro', zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], average='weighted', zero_division=0)
    
    per_class_p, per_class_r, per_class_f1, per_class_s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], zero_division=0)
    
    return {
        'accuracy': acc,
        'macro_precision': mac_p,
        'macro_recall': mac_r,
        'macro_f1': mac_f1,
        'weighted_f1': w_f1,
        'per_class': {
            'precision': per_class_p,
            'recall': per_class_r,
            'f1': per_class_f1,
            'support': per_class_s
        }
    }

def calculate_custom_research_metrics(y_true, y_pred):
    # Critical Intent Recall (Classes 1, 2, 3, 5)
    critical_classes = [1, 2, 3, 5]
    crit_true = []
    crit_pred = []
    for t, p in zip(y_true, y_pred):
        if t in critical_classes:
            crit_true.append(t)
            crit_pred.append(p)
            
    if len(crit_true) > 0:
        _, crit_recall, _, _ = precision_recall_fscore_support(crit_true, crit_pred, average='micro', zero_division=0)
    else:
        crit_recall = 0.0
        
    # False Alarm Rate (Class 0 predicted as 1, 2, 3, 5)
    casual_transit_count = 0
    false_alarm_count = 0
    for t, p in zip(y_true, y_pred):
        if t == 0:
            casual_transit_count += 1
            if p in critical_classes:
                false_alarm_count += 1
                
    false_alarm_rate = false_alarm_count / casual_transit_count if casual_transit_count > 0 else 0.0
    
    return {
        'critical_recall': crit_recall,
        'false_alarm_rate': false_alarm_rate,
        'false_alarm_count': false_alarm_count
    }

def simulate_early_warning(test_df, predictions):
    """
    Simulates early warning time by aligning predictions with tracks.
    Since predictions map 1:1 with the final frames of sequences, we approximate.
    """
    total_intrusions = 0
    early_warnings = 0
    warning_times = []
    
    # Just an approximation metric for the research paper based on the test set
    df = test_df.copy()
    if len(predictions) == len(df):
        df['pred'] = predictions
        for track_id, group in df.groupby('track_id'):
            if 5 in group['intent_class'].values:
                total_intrusions += 1
                # Find first time prediction was suspicious (1, 2, 3, 5)
                suspicious = group[group['pred'].isin([1, 2, 3, 5])]
                if len(suspicious) > 0:
                    first_suspicious = suspicious.iloc[0]['frame_idx']
                    intrusion_frame = group[group['intent_class'] == 5].iloc[0]['frame_idx']
                    if first_suspicious < intrusion_frame:
                        early_warnings += 1
                        warning_times.append(intrusion_frame - first_suspicious)
                        
    early_warning_rate = early_warnings / total_intrusions if total_intrusions > 0 else 0.0
    mean_ew_time = np.mean(warning_times) if warning_times else 0.0
    
    return {
        'early_warning_rate': early_warning_rate,
        'mean_early_warning_frames': mean_ew_time
    }

def measure_latency(model_predict_func, sample_input, iterations=100):
    start_time = time.time()
    for _ in range(iterations):
        model_predict_func(sample_input)
    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    return avg_latency_ms
