import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURE_COLS = [
    'speed_variance', 'distance_to_restricted_zone',
    'time_near_boundary', 'time_inside_zone',
    'sudden_stops', 'carrying_baggage', 'time_of_day_multiplier'
]

class AnomalyDetector:
    def __init__(self, model_path='ml/data/rf_model.pkl', scaler_path='ml/data/rf_scaler.pkl', encoder_path='ml/data/rf_encoder.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.model = None
        self.scaler = None
        self.encoder = None

    def train_model(self, csv_path='ml/data/tabular_dataset.csv'):
        df = pd.read_csv(csv_path)
        X = df[FEATURE_COLS]
        y = df['intent_class']

        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        self.model.fit(X_scaled, y_enc)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.encoder, self.encoder_path)
        print(f"  [RF] Model trained and saved.")

    def _load(self):
        if self.model is None:
            self.model = joblib.load(self.model_path)
            self.model.n_jobs = 1  # Force single thread for single-sample inference
            self.scaler = joblib.load(self.scaler_path)
            self.encoder = joblib.load(self.encoder_path)

    def predict_intent(self, features_dict):
        self._load()
        x_input = np.array([[features_dict.get(c, 0.0) for c in FEATURE_COLS]])
        x_scaled = self.scaler.transform(x_input)
        pred_enc = self.model.predict(x_scaled)[0]
        return self.encoder.inverse_transform([pred_enc])[0]

    def predict_batch(self, X):
        """For use in benchmarking. X is a numpy array."""
        self._load()
        X_scaled = self.scaler.transform(X)
        preds_enc = self.model.predict(X_scaled)
        return self.encoder.inverse_transform(preds_enc)
