import numpy as np
from sklearn.metrics import f1_score

class HybridFusionModel:
    def __init__(self):
        self.best_alpha = 0.5
        self.alpha_history = []
        self.f1_history = []
        
    def find_best_alpha(self, p_static, p_seq, y_true):
        """
        Grid search for the best alpha to balance Static and Sequential probabilities.
        P_final = alpha * p_static + (1 - alpha) * p_seq
        """
        best_f1 = 0.0
        
        # Test alphas from 0.0 to 1.0
        alphas = np.linspace(0.0, 1.0, 11)
        for alpha in alphas:
            p_final = alpha * p_static + (1 - alpha) * p_seq
            y_pred = np.argmax(p_final, axis=1)
            
            macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], average='weighted', zero_division=0)
            
            self.alpha_history.append(alpha)
            self.f1_history.append(macro_f1)
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                self.best_alpha = alpha
                
        print(f"Hybrid Grid Search complete. Best Alpha: {self.best_alpha:.2f} -> Val Weighted F1: {best_f1:.4f}")
        return self.best_alpha
        
    def predict_proba(self, p_static, p_seq, alpha=None):
        if alpha is None:
            alpha = self.best_alpha
        return alpha * p_static + (1 - alpha) * p_seq
        
    def predict(self, p_static, p_seq, alpha=None):
        p_final = self.predict_proba(p_static, p_seq, alpha)
        return np.argmax(p_final, axis=1)
