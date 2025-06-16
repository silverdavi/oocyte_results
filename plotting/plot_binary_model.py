#!/usr/bin/env python3
"""
Create binary classification visualization plots using REAL DATA.
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_processed_data():
    """Load processed real data"""
    data = {}
    base_path = Path("data_analysis/processed")
    
    # Load real performance metrics
    metrics_file = base_path / "binary_performance_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Load real ROC data
    roc_file = base_path / "binary_roc_real.pkl"
    if roc_file.exists():
        with open(roc_file, 'rb') as f:
            data['roc_data'] = pickle.load(f)
    
    # Load real confusion matrix
    confusion_file = base_path / "binary_confusion_real.pkl"
    if confusion_file.exists():
        with open(confusion_file, 'rb') as f:
            data['confusion_data'] = pickle.load(f)
    
    # Load real CV results
    cv_file = base_path / "binary_cv_results_real.csv"
    if cv_file.exists():
        data['cv_results'] = pd.read_csv(cv_file)
    
    # Load real labels
    labels_file = base_path / "binary_labels_real.csv"
    if labels_file.exists():
        data['labels'] = pd.read_csv(labels_file)
    
    # Load real label statistics
    stats_file = base_path / "binary_label_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            data['label_stats'] = json.load(f)
    
    return data

def create_roc_curve_plot(data, output_dir):
    """Create ROC curve plot using real data"""
    if 'roc_data' not in data:
        print("No ROC data available")
        return
    
    roc_data = data['roc_data']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, label=f"ROC Curve (AUC = {data['metrics']['overall_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Real Binary Classification Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: binary_roc_curve.png")

def create_precision_recall_plot(data, output_dir):
    """Create Precision-Recall curve using real data"""
    if 'metrics' not in data or 'pr_curve' not in data['metrics']:
        print("No PR data available")
        return
    
    pr_data = data['metrics']['pr_curve']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot PR curve
    ax.plot(pr_data['recall'], pr_data['precision'], linewidth=2, label='PR Curve')
    
    # Add baseline (random classifier)
    if 'label_stats' in data:
        baseline = data['label_stats']['positive_rate']
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Random Classifier (P={baseline:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Real Binary Classification Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: binary_precision_recall.png")

def create_confusion_matrix_plot(data, output_dir):
    """Create confusion matrix plot using real data"""
    if 'confusion_data' not in data:
        print("No confusion matrix data available")
        return
    
    confusion_data = data['confusion_data']
    cm = np.array(confusion_data['confusion_matrix'])
    class_names = confusion_data['class_names']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix - Real Binary Classification Model',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: binary_confusion_matrix.png")

def create_cv_performance_plot(data, output_dir):
    """Create CV performance plot using real fold metrics"""
    if 'metrics' not in data or 'fold_metrics' not in data['metrics']:
        print("No CV results data available")
        return
    
    fold_metrics = data['metrics']['fold_metrics']
    df = pd.DataFrame(fold_metrics)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Accuracy across folds
    axes[0,0].bar(df['fold'], df['accuracy'])
    axes[0,0].set_title('Accuracy by Fold')
    axes[0,0].set_xlabel('Fold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    
    # Precision across folds
    axes[0,1].bar(df['fold'], df['precision'])
    axes[0,1].set_title('Precision by Fold')
    axes[0,1].set_xlabel('Fold')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].grid(True, alpha=0.3)
    
    # Recall across folds
    axes[0,2].bar(df['fold'], df['recall'])
    axes[0,2].set_title('Recall by Fold')
    axes[0,2].set_xlabel('Fold')
    axes[0,2].set_ylabel('Recall')
    axes[0,2].grid(True, alpha=0.3)
    
    # F1 across folds
    axes[1,0].bar(df['fold'], df['f1'])
    axes[1,0].set_title('F1 Score by Fold')
    axes[1,0].set_xlabel('Fold')
    axes[1,0].set_ylabel('F1')
    axes[1,0].grid(True, alpha=0.3)
    
    # AUC across folds
    axes[1,1].bar(df['fold'], df['auc'])
    axes[1,1].set_title('AUC by Fold')
    axes[1,1].set_xlabel('Fold')
    axes[1,1].set_ylabel('AUC')
    axes[1,1].grid(True, alpha=0.3)
    
    # Summary statistics
    metrics_summary = {
        'Accuracy': f"{data['metrics']['overall_accuracy']:.3f} ± {df['accuracy'].std():.3f}",
        'Precision': f"{data['metrics']['overall_precision']:.3f} ± {df['precision'].std():.3f}",
        'Recall': f"{data['metrics']['overall_recall']:.3f} ± {df['recall'].std():.3f}",
        'F1': f"{data['metrics']['overall_f1']:.3f} ± {df['f1'].std():.3f}",
        'AUC': f"{data['metrics']['overall_auc']:.3f} ± {df['auc'].std():.3f}",
        'Samples': data['metrics']['total_samples']
    }
    
    text_str = '\n'.join([f"{k}: {v}" for k, v in metrics_summary.items()])
    axes[1,2].text(0.1, 0.7, text_str, transform=axes[1,2].transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1,2].set_title('Performance Summary')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_cv_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: binary_cv_performance.png")

def create_performance_summary(data, output_dir):
    """Create performance summary plot"""
    if 'metrics' not in data:
        print("No metrics data available")
        return
    
    metrics = data['metrics']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of main metrics
    performance_metrics = ['overall_accuracy', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values = [metrics[metric] for metric in performance_metrics]
    
    bars = ax1.bar(metric_names, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
    ax1.set_ylabel('Score')
    ax1.set_title('Binary Classification Performance Summary\n(Real Data Results)')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Class distribution
    if 'label_stats' in data:
        stats = data['label_stats']
        labels = ['Negative', 'Positive']
        sizes = [stats['negative_count'], stats['positive_count']]
        colors = ['lightcoral', 'lightblue']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution in Real Data')
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: binary_performance_summary.png")

def main():
    """Main function to create all binary model plots"""
    print("📊 Creating Binary Model Plots")
    print("="*50)
    
    # Setup output directory
    output_dir = Path("plotting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Create plots
    print("Creating ROC curve...")
    create_roc_curve_plot(data, output_dir)
    
    print("Creating Precision-Recall curve...")
    create_precision_recall_plot(data, output_dir)
    
    print("Creating confusion matrix...")
    create_confusion_matrix_plot(data, output_dir)
    
    print("Creating CV performance plot...")
    create_cv_performance_plot(data, output_dir)
    
    print("Creating performance summary...")
    create_performance_summary(data, output_dir)
    
    print(f"\n✅ All binary model plots saved to {output_dir}")
    
    # List generated files
    generated_files = [f for f in output_dir.glob("binary_*.png")]
    if generated_files:
        print("Generated files:")
        for f in generated_files:
            print(f"  - {f.name}")

if __name__ == "__main__":
    main() 