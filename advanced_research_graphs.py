import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def generate_additional_research_graphs(input_csv='ml/data/tabular_dataset.csv', out_dir='results/research_analysis/'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading dataset from {input_csv} to generate advanced scientific plots...")
    df = pd.read_csv(input_csv)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Prepare data for ML metrics testing
    # Exclude non-input features
    X = df.select_dtypes(include=[np.number]).drop(columns=['historical_risk_score', 'pca_component_1', 'pca_component_2'], errors='ignore')
    y = df['intent_class']
    classes = y.unique()
    
    # 70/30 Train/Test split for evaluation graphics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # ---------------------------------------------------------
    # Graph 7: Confusion Matrix Heatmap
    # ---------------------------------------------------------
    print("Generating Confusion Matrix Heatmap...")
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='gray')
    plt.title('Classification Confusion Matrix Heatmap (Random Forest)')
    plt.ylabel('Actual Behavior Intent')
    plt.xlabel('Predicted Behavior Intent')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_7_confusion_matrix_heatmap.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Graph 8: Lower-Triangle Correlation Heatmap (Academic Standard)
    # ---------------------------------------------------------
    print("Generating Masked Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = X.corr()
    # Mask the upper triangle for a cleaner academic look
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title('Lower Triangle Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_8_correlation_triangle_heatmap.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Graph 9: 2D Hexbin Density Heatmap
    # ---------------------------------------------------------
    print("Generating 2D Hexbin Density Heatmap...")
    plt.figure(figsize=(8, 6))
    # Logarithmic scale for better visual clustering ranges
    hb = plt.hexbin(df['distance_to_restricted_zone'], df['speed_variance'], gridsize=30, cmap='YlOrRd', bins='log', edgecolors='white', linewidths=0.1)
    cb = plt.colorbar(hb, label='log10(N)')
    plt.title('Bivariate Density Heatmap: Distance vs Speed')
    plt.xlabel('Distance to Restricted Polygon (px)')
    plt.ylabel('Speed Variance (px/sec)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_9_hexbin_density_heatmap.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Graph 10: Multi-class ROC Curve
    # ---------------------------------------------------------
    print("Generating Multi-class ROC Evaluation Curves...")
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    y_score = rf.predict_proba(X_test)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, color in zip(range(n_classes), colors):
        if i < len(classes): 
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC ({classes[i]}) (AUC = {roc_auc:0.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Mapping')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_10_multiclass_ROC.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Graph 11: Machine Learning Convergence / Learning Curve
    # ---------------------------------------------------------
    print("Generating Algorithm Learning Curve Matrix...")
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="crimson")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="navy")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="crimson", label="Training Accuracy Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="navy", label="Cross-Validation Accuracy")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy Score")
    plt.title("Model Saturation & Learning Curve")
    plt.legend(loc="best")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/graph_11_learning_curve.png", dpi=300)
    plt.close()
    
    print("\n[SUCCESS] Successfully generated additional academic research graphs!")

if __name__ == '__main__':
    generate_additional_research_graphs()