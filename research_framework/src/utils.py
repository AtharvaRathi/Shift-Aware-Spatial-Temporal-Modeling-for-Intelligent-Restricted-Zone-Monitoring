import os
import json
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed=42):
    """Ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_experiment(config, metrics, models_tested, output_dir='research_framework/experiments/logs'):
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "models_tested": models_tested,
        "results": metrics
    }
    
    log_path = os.path.join(output_dir, f"exp_{experiment_id}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print(f"Experiment log saved to {log_path}")
    return log_path
