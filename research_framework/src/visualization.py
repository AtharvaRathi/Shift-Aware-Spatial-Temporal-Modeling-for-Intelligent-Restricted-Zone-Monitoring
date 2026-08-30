import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .dataset import CLASS_NAMES, FEATURE_COLUMNS

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def save_fig(name, dir_path='research_framework/results/figures'):
    os.makedirs(dir_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, f"{name}.png"), dpi=300)
    plt.savefig(os.path.join(dir_path, f"{name}.pdf"), format='pdf')
    plt.close()

def plot_figure_1_overall_f1(metrics_dict):
    models = list(metrics_dict.keys())
    f1s = [metrics_dict[m]['weighted_f1'] for m in models]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=f1s, hue=models, legend=False, palette="viridis")
    plt.title("Figure 1: Overall Model Comparison (Weighted F1)")
    plt.ylabel("Weighted F1-Score")
    save_fig("fig_1_overall_f1")

def plot_figure_2_accuracy_metrics(metrics_dict):
    models = list(metrics_dict.keys())
    data = []
    for m in models:
        data.append({'Model': m, 'Metric': 'Accuracy', 'Score': metrics_dict[m]['accuracy']})
        data.append({'Model': m, 'Metric': 'Macro Precision', 'Score': metrics_dict[m]['macro_precision']})
        data.append({'Model': m, 'Metric': 'Macro Recall', 'Score': metrics_dict[m]['macro_recall']})
        data.append({'Model': m, 'Metric': 'Weighted F1', 'Score': metrics_dict[m]['weighted_f1']})
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette="Set2")
    plt.title("Figure 2: Accuracy Comparison Across Metrics")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig("fig_2_accuracy_metrics")

def plot_figure_3_static_vs_sequential(metrics_dict):
    static_f1 = np.mean([metrics_dict['Random Forest']['weighted_f1'], metrics_dict['XGBoost']['weighted_f1']])
    seq_f1 = np.mean([metrics_dict['LSTM']['weighted_f1'], metrics_dict['GRU']['weighted_f1']])
    
    static_cr = np.mean([metrics_dict['Random Forest']['critical_recall'], metrics_dict['XGBoost']['critical_recall']])
    seq_cr = np.mean([metrics_dict['LSTM']['critical_recall'], metrics_dict['GRU']['critical_recall']])
    
    df = pd.DataFrame({
        'Paradigm': ['Static (Avg)', 'Static (Avg)', 'Sequential (Avg)', 'Sequential (Avg)'],
        'Metric': ['Weighted F1', 'Critical Recall', 'Weighted F1', 'Critical Recall'],
        'Score': [static_f1, static_cr, seq_f1, seq_cr]
    })
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Paradigm', palette="magma")
    plt.title("Figure 3: Static vs Sequential Comparison")
    save_fig("fig_3_static_vs_sequential")

def plot_figure_4_confusion_matrices(y_true_dict, y_pred_dict):
    out_dir = 'research_framework/results/confusion_matrices'
    os.makedirs(out_dir, exist_ok=True)
    
    for model_name in y_true_dict.keys():
        cm = confusion_matrix(y_true_dict[model_name], y_pred_dict[model_name])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"Figure 4: Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name.lower().replace(' ', '_')}_cm.png"), dpi=300)
        plt.close()

def plot_figure_5_per_class_f1(metrics_dict):
    data = []
    for model_name, metrics in metrics_dict.items():
        f1s = metrics['per_class']['f1']
        for i, class_name in enumerate(CLASS_NAMES):
            data.append({'Model': model_name, 'Class': class_name, 'F1-Score': f1s[i]})
            
    df = pd.DataFrame(data)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Class', y='F1-Score', hue='Model', palette="Set1")
    plt.title("Figure 5: Per-Class F1 Comparison")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig("fig_5_per_class_f1")

def plot_figure_6_precision_vs_recall(metrics_dict):
    plt.figure(figsize=(8, 8))
    for m in metrics_dict:
        plt.scatter(metrics_dict[m]['macro_precision'], metrics_dict[m]['macro_recall'], label=m, s=150)
        plt.annotate(m, (metrics_dict[m]['macro_precision'], metrics_dict[m]['macro_recall']), xytext=(5, 5), textcoords='offset points')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim(0.5, 1.05)
    plt.ylim(0.5, 1.05)
    plt.xlabel("Macro Precision")
    plt.ylabel("Macro Recall")
    plt.title("Figure 6: Precision vs Recall")
    plt.grid(True)
    plt.legend()
    save_fig("fig_6_precision_vs_recall")

def plot_figure_8_fusion_alpha(alpha_history, f1_history):
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_history, f1_history, marker='o', linewidth=2, color='purple')
    plt.title("Figure 8: Fusion Weight (\u03B1) vs Performance")
    plt.xlabel("\u03B1 (0 = Pure Sequential, 1 = Pure Static)")
    plt.ylabel("Weighted F1")
    plt.grid(True)
    save_fig("fig_8_fusion_alpha")

def plot_figure_10_accuracy_vs_latency(metrics_dict):
    plt.figure(figsize=(10, 6))
    for m in metrics_dict:
        plt.scatter(metrics_dict[m]['latency_ms'], metrics_dict[m]['weighted_f1'], label=m, s=200, alpha=0.8)
        plt.annotate(m, (metrics_dict[m]['latency_ms'], metrics_dict[m]['weighted_f1']), xytext=(8, 8), textcoords='offset points')
        
    plt.xlabel("Inference Latency (ms)")
    plt.ylabel("Weighted F1")
    plt.title("Figure 10: Accuracy vs Latency Trade-off")
    plt.grid(True)
    plt.legend()
    save_fig("fig_10_accuracy_vs_latency")

def plot_figure_14_feature_importance(rf_imp, xgb_imp):
    df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS * 2,
        'Importance': list(rf_imp) + list(xgb_imp),
        'Model': ['Random Forest'] * len(FEATURE_COLUMNS) + ['XGBoost'] * len(FEATURE_COLUMNS)
    })
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, y='Feature', x='Importance', hue='Model', orient='h', palette='cubehelix')
    plt.title("Figure 14: Feature Importance (Static Models)")
    save_fig("fig_14_feature_importance")

def plot_training_history(lstm_hist, gru_hist):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_hist['train_loss'], label='LSTM Train')
    plt.plot(lstm_hist['val_loss'], label='LSTM Val')
    plt.plot(gru_hist['train_loss'], label='GRU Train')
    plt.plot(gru_hist['val_loss'], label='GRU Val')
    plt.title("Figure 12: Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_hist['val_f1'], label='LSTM Val F1')
    plt.plot(gru_hist['val_f1'], label='GRU Val F1')
    plt.title("Figure 13: Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    
    save_fig("fig_12_13_training_history")

def generate_all_visualizations(metrics, y_true_dict, y_pred_dict, alpha_hist, f1_hist, rf_imp, xgb_imp, lstm_hist, gru_hist):
    print("Generating all 14 research figures...")
    plot_figure_1_overall_f1(metrics)
    plot_figure_2_accuracy_metrics(metrics)
    plot_figure_3_static_vs_sequential(metrics)
    plot_figure_4_confusion_matrices(y_true_dict, y_pred_dict)
    plot_figure_5_per_class_f1(metrics)
    plot_figure_6_precision_vs_recall(metrics)
    plot_figure_8_fusion_alpha(alpha_hist, f1_hist)
    plot_figure_10_accuracy_vs_latency(metrics)
    plot_figure_14_feature_importance(rf_imp, xgb_imp)
    plot_training_history(lstm_hist, gru_hist)
    print("All visualizations saved successfully in research_framework/results/figures")
