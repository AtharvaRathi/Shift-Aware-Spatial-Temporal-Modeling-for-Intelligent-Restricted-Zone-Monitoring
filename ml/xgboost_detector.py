import os
import joblib
import numpy as np
import warnings

FEATURE_COLS = [
    'speed_variance', 'distance_to_restricted_zone',
    'time_near_boundary', 'time_inside_zone',
    'sudden_stops', 'carrying_baggage', 'time_of_day_multiplier'
]

CLASS_NAMES = [
    "Casual Transit",
    "Approaching Boundary",
    "Suspicious Loitering",
    "Active Scouting",
    "Fleeing",
    "CRITICAL INTRUSION"
]

class XGBoostDetector:
    def __init__(self, model_path='research_framework/models/static/xgb_best.pkl'):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                warnings.warn(f"[XGBoostDetector] Could not load model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

    def predict_intent(self, features):
        """
        Predicts intent class from the 7 standard features.
        Accepts a dict containing:
          - speed_variance
          - distance_to_restricted_zone
          - time_near_boundary
          - time_inside_zone
          - sudden_stops
          - carrying_baggage
          - time_of_day_multiplier
        """
        if self.model is None:
            return 'unknown'

        try:
            feat_vals = [float(features.get(col, 0.0)) for col in FEATURE_COLS]
            X = np.array([feat_vals], dtype=np.float32)
            pred_idx = int(self.model.predict(X)[0])
            if 0 <= pred_idx < len(CLASS_NAMES):
                return CLASS_NAMES[pred_idx]
            return 'unknown'
        except Exception:
            return 'unknown'
