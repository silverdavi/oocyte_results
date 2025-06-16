#!/usr/bin/env python3
"""
Create focused oocyte quality plots: correlation, ROC curves, and classification metrics
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from scipy import stats

# Set publication style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def load_real_data():
    """Load processed real data"""
    data = {}
    base_path = Path("data_analysis/processed")
    
    # Load real calibration data
    calibration_file = base_path / "continuous_calibration_real.pkl"
    if calibration_file.exists():
        with open(calibration_file, 'rb') as f:
            data['calibration'] = pickle.load(f)
    
    # Load real CV results
    cv_file = base_path / "binary_cv_results_real.csv"
    if cv_file.exists():
        data['cv_results'] = pd.read_csv(cv_file)
    
    # Load performance metrics
    metrics_file = base_path / "binary_performance_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    return data

def create_correlation_plot(data, output_dir):
    """Create histogram-based correlation plot with bins [0:0.1:1] colored by prediction"""
    
    if 'calibration' not in data:
        print("No calibration data available")
        return
    
    calibration = data['calibration']
    true_scores = np.array(calibration['true_scores'])
    pred_scores = np.array(calibration['predicted_scores'])
    
    # Calculate correlation
    correlation = np.corrcoef(true_scores, pred_scores)[0, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Histogram by true score bins, colored by average prediction
    bins = np.arange(0, 1.1, 0.1)  # [0:0.1:1]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Digitize true scores into bins
    bin_indices = np.digitize(true_scores, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    
    # Calculate average prediction for each bin
    avg_predictions = []
    counts = []
    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            avg_predictions.append(np.mean(pred_scores[mask]))
            counts.append(np.sum(mask))
        else:
            avg_predictions.append(0)
            counts.append(0)
    
    # Create histogram colored by prediction
    colors = plt.cm.viridis(np.array(avg_predictions))
    bars = ax1.bar(bin_centers, counts, width=0.08, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label('Average Predicted Score', fontweight='bold')
    
    ax1.set_xlabel('True Quality Score (Binned)', fontweight='bold')
    ax1.set_ylabel('Sample Count', fontweight='bold')
    ax1.set_title('Distribution by True Score\n(Colored by Average Prediction)', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count, avg_pred in zip(bars, counts, avg_predictions):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{count}\n({avg_pred:.2f})', ha='center', va='bottom', fontsize=9)
    
    # Right plot: Boxplot of predicted scores for each true label bin
    # Prepare data for boxplots
    boxplot_data = []
    bin_labels = []
    
    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_predictions = pred_scores[mask]
            boxplot_data.append(bin_predictions)
            bin_labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f}')
        else:
            boxplot_data.append([])
            bin_labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f}')
    
    # Create boxplot
    bp = ax2.boxplot(boxplot_data, tick_labels=bin_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Color boxplots by their median values
    medians = [np.median(data) if len(data) > 0 else 0 for data in boxplot_data]
    colors = plt.cm.viridis(np.array(medians))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the boxplot elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black', alpha=0.8)
    plt.setp(bp['means'], color='red', linewidth=2)
    
    ax2.set_xlabel('True Quality Score Bins', fontweight='bold')
    ax2.set_ylabel('Predicted Quality Score Distribution', fontweight='bold')
    ax2.set_title('Prediction Distribution by True Score Bins\n(Boxplot View)', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.45, 0.7)
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    # Add legend for boxplot elements - positioned outside plot area
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        patches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, label='IQR (25-75%)')
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', 
              fontsize=9, frameon=True, fancybox=True)
    
    # Add correlation text box for boxplot - positioned to avoid overlap
    # Calculate additional statistics
    mae = np.mean(np.abs(true_scores - pred_scores))
    rmse = np.sqrt(np.mean((true_scores - pred_scores)**2))
    
    textstr = f'Overall Performance:\nCorrelation: r = {correlation:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}\nSamples: {len(true_scores):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.68, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / "oocyte_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: oocyte_correlation.png")

def create_roc_comparison_plot(data, output_dir):
    """Create ROC curves with area shading for validation and label shuffle"""
    
    if 'cv_results' not in data:
        print("No CV results available")
        return
    
    cv_results = data['cv_results']
    true_labels = cv_results['label'].values
    pred_probs = cv_results['mean'].values
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Note: We'll show CV median as our main model performance (removed redundant "Our Model" line)
    
    # 2. Generate multiple validation curves (using cross-validation folds)
    fold_cols = [col for col in cv_results.columns if col.startswith('fold')]
    validation_fprs = []
    validation_tprs = []
    validation_aucs = []
    
    for fold_col in fold_cols[:5]:  # Use first 5 folds for validation curves
        fold_probs = cv_results[fold_col].values
        fpr_val, tpr_val, _ = roc_curve(true_labels, fold_probs)
        auc_val = roc_auc_score(true_labels, fold_probs)
        validation_fprs.append(fpr_val)
        validation_tprs.append(tpr_val)
        validation_aucs.append(auc_val)
    
    # Interpolate validation curves to common FPR points
    common_fpr = np.linspace(0, 1, 100)
    interpolated_tprs = []
    for fpr_val, tpr_val in zip(validation_fprs, validation_tprs):
        interpolated_tpr = np.interp(common_fpr, fpr_val, tpr_val)
        interpolated_tprs.append(interpolated_tpr)
    
    # Calculate median and confidence interval for validation
    interpolated_tprs = np.array(interpolated_tprs)
    median_tpr = np.median(interpolated_tprs, axis=0)
    tpr_25 = np.percentile(interpolated_tprs, 25, axis=0)
    tpr_75 = np.percentile(interpolated_tprs, 75, axis=0)
    
    ax.plot(common_fpr, median_tpr, linewidth=2, color='darkblue', alpha=0.8,
            label=f'CV Folds Median (AUC = {np.median(validation_aucs):.3f})', zorder=2)
    ax.fill_between(common_fpr, tpr_25, tpr_75, alpha=0.2, color='blue', 
                   label='CV 25-75 percentile', zorder=1)
    
    # 3. Label shuffle with area shading
    shuffle_fprs = []
    shuffle_tprs = []
    shuffle_aucs = []
    
    for i in range(10):  # Multiple shuffle runs
        shuffled_labels = shuffle(true_labels, random_state=i)
        fpr_shuffle, tpr_shuffle, _ = roc_curve(shuffled_labels, pred_probs)
        auc_shuffle = roc_auc_score(shuffled_labels, pred_probs)
        shuffle_fprs.append(fpr_shuffle)
        shuffle_tprs.append(tpr_shuffle)
        shuffle_aucs.append(auc_shuffle)
    
    # Calculate median and confidence interval for shuffle
    interpolated_shuffle_tprs = []
    for fpr_shuf, tpr_shuf in zip(shuffle_fprs, shuffle_tprs):
        interpolated_tpr = np.interp(common_fpr, fpr_shuf, tpr_shuf)
        interpolated_shuffle_tprs.append(interpolated_tpr)
    
    interpolated_shuffle_tprs = np.array(interpolated_shuffle_tprs)
    median_shuffle_tpr = np.median(interpolated_shuffle_tprs, axis=0)
    shuffle_tpr_25 = np.percentile(interpolated_shuffle_tprs, 25, axis=0)
    shuffle_tpr_75 = np.percentile(interpolated_shuffle_tprs, 75, axis=0)
    
    ax.plot(common_fpr, median_shuffle_tpr, linewidth=2, color='darkred', alpha=0.8,
            label=f'Label Shuffle Median (AUC = {np.median(shuffle_aucs):.3f})', zorder=1)
    ax.fill_between(common_fpr, shuffle_tpr_25, shuffle_tpr_75, alpha=0.2, color='red', 
                   label='Shuffle 25-75 percentile', zorder=0)
    
    # 4. Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7,
            label='Random Classifier (AUC = 0.500)', zorder=1)
    
    # Formatting
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('ROC Curve Comparison with Validation Areas\nOocyte Blastulation Prediction', 
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=True, fancybox=True, loc='lower right')
    
    # Add performance text with quantitative significance
    # Statistical significance test: CV AUCs vs Shuffle AUCs
    statistic, p_value = stats.mannwhitneyu(validation_aucs, shuffle_aucs, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(validation_aucs)-1)*np.var(validation_aucs, ddof=1) + 
                         (len(shuffle_aucs)-1)*np.var(shuffle_aucs, ddof=1)) / 
                        (len(validation_aucs) + len(shuffle_aucs) - 2))
    cohens_d = (np.mean(validation_aucs) - np.mean(shuffle_aucs)) / pooled_std
    
    # Format p-value
    if p_value < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_value:.3f}"
    
    textstr = f'Model Performance:\n• CV median AUC: {np.median(validation_aucs):.3f}\n• CV range: {np.min(validation_aucs):.3f}-{np.max(validation_aucs):.3f}\n• vs Shuffle: {p_str}\n• Effect size (d): {cohens_d:.2f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.6, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "oocyte_roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: oocyte_roc_comparison.png")

def create_classification_metrics_plot(data, output_dir):
    """Create binary classification metrics visualization with better label positioning"""
    
    if 'metrics' not in data or 'cv_results' not in data:
        print("No metrics or CV results available")
        return
    
    metrics = data['metrics']
    cv_results = data['cv_results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate metrics for each CV fold to get error bars
    fold_cols = [col for col in cv_results.columns if col.startswith('fold')]
    true_labels = cv_results['label'].values
    
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold_col in fold_cols:
        fold_probs = cv_results[fold_col].values
        fold_preds = (fold_probs > 0.5).astype(int)
        
        fold_metrics['accuracy'].append(accuracy_score(true_labels, fold_preds))
        fold_metrics['precision'].append(precision_score(true_labels, fold_preds, zero_division=0))
        fold_metrics['recall'].append(recall_score(true_labels, fold_preds, zero_division=0))
        fold_metrics['f1'].append(f1_score(true_labels, fold_preds, zero_division=0))
        fold_metrics['auc'].append(roc_auc_score(true_labels, fold_probs))
    
    # Left plot: Main metrics bar chart with error bars
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metric_values = [
        np.mean(fold_metrics['accuracy']),
        np.mean(fold_metrics['precision']), 
        np.mean(fold_metrics['recall']),
        np.mean(fold_metrics['f1']),
        np.mean(fold_metrics['auc'])
    ]
    
    metric_errors = [
        np.std(fold_metrics['accuracy']),
        np.std(fold_metrics['precision']), 
        np.std(fold_metrics['recall']),
        np.std(fold_metrics['f1']),
        np.std(fold_metrics['auc'])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(metric_names, metric_values, yerr=metric_errors, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1,
                   capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars with better positioning
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        # Position label inside bar if high value, outside if low
        if height > 0.7:
            ax1.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                    f'{value:.3f}', ha='center', va='top', fontweight='bold', 
                    fontsize=12, color='white')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', 
                    fontsize=12, color='black')
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Binary Classification Performance\n(702 Real Oocyte Samples)', 
                  fontweight='bold', pad=20)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add reference line
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Random baseline')
    ax1.legend(fontsize=11, loc='upper left')
    
    # Right plot: Confusion matrix as heatmap with better annotations
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        
        # Create labels with percentages
        total = cm.sum()
        cm_percent = cm / total * 100
        annot_text = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                               for j in range(cm.shape[1])] 
                               for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', ax=ax2,
                   cbar_kws={'label': 'Count'}, square=True, linewidths=2,
                   xticklabels=['Predicted: No', 'Predicted: Yes'],
                   yticklabels=['Actual: No', 'Actual: Yes'],
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax2.set_title('Confusion Matrix\n(Counts and Percentages)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "oocyte_classification_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: oocyte_classification_metrics.png")

def main():
    """Create focused oocyte quality plots"""
    print("📊 Creating Focused Oocyte Quality Plots")
    print("="*50)
    
    # Setup output directory
    output_dir = Path("plotting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading real data...")
    data = load_real_data()
    
    # Create focused plots
    print("Creating correlation plot...")
    create_correlation_plot(data, output_dir)
    
    print("Creating ROC comparison plot...")
    create_roc_comparison_plot(data, output_dir)
    
    print("Creating classification metrics plot...")
    create_classification_metrics_plot(data, output_dir)
    
    print(f"\n✅ Oocyte quality plots saved to {output_dir}")

if __name__ == "__main__":
    main() 