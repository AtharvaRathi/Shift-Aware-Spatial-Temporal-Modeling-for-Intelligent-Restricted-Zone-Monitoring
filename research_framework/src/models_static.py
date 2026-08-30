import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .dataset import FEATURE_COLUMNS

class StaticModelPipeline:
    def __init__(self, model_type='rf', models_dir='research_framework/models/static'):
        self.model_type = model_type.lower()
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model = None
        self.model_path = os.path.join(self.models_dir, f"{self.model_type}_best.pkl")
        
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == 'xgb':
            self.model = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError("Unsupported model_type. Use 'rf' or 'xgb'.")
            
    def train(self, train_df, val_df=None):
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df['intent_class'].values
        
        print(f"Training Static {self.model_type.upper()} Model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        self.save()
        
    def predict(self, df):
        if self.model is None:
            self.load()
        X = df[FEATURE_COLUMNS].values
        return self.model.predict(X)
        
    def predict_proba(self, df):
        if self.model is None:
            self.load()
        X = df[FEATURE_COLUMNS].values
        return self.model.predict_proba(X)
        
    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"Saved {self.model_type} to {self.model_path}")
        
    def load(self):
        self.model = joblib.load(self.model_path)
        
    def get_feature_importance(self):
        if self.model is None:
            self.load()
        return self.model.feature_importances_
