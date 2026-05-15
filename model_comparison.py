import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support

def run_model_comparisons():
    input_csv = 'custom_dataset.csv'
    out_dir = 'results/model_comparison'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(input_csv)
    
    # Exclude non-input features
    X = df.select_dtypes(include=[np.number]).drop(columns=['historical_risk_score', 'pca_component_1', 'pca_component_2'], errors='ignore')
    y = df['intent_class']
    
    # Scale features for Deep Learning (MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Deep Learning (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    }
    
    results = []
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = len(classes)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    print("\n--- Training and Evaluating Models ---")
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Weighted)': prec,
            'Recall (Weighted)': rec,
            'F1-Score (Weighted)': f1
        })
        
        # Error Matrix (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', xticklabels=classes, yticklabels=classes)
        plt.title(f'Error Matrix: {name}')
        plt.ylabel('Actual Intent')
        plt.xlabel('Predicted Intent')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/error_matrix_{name.replace(' ', '_')}.png", dpi=300)
        plt.close()

        # ROC Curve (Micro-average)
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        plt.figure(1) # Switch back to ROC plot figure
        plt.plot(fpr_micro, tpr_micro, color=colors[i], lw=2, 
                 label=f'{name} (Micro-AUC = {roc_auc_micro:0.3f})')
                 
    # Finalize ROC Curve Graph
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison Across Models')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_model_comparison_roc.png", dpi=300)
    plt.close()
    
    # Save Results Table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/model_metrics_comparison.csv", index=False)
    print("\n--- Model Comparison Results ---")
    print(results_df.to_string(index=False))
    
    # Bar Chart for Metrics
    results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_melted, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Performance Metrics Comparison')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_performance_metrics_bar.png", dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] Generated models, error matrices, and ROC curves in '{out_dir}/'")

if __name__ == '__main__':
    run_model_comparisons()