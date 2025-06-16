#!/usr/bin/env python3
"""
Create continuous model visualization plots using REAL DATA.
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_processed_data():
    """Load processed real data"""
    data = {}
    base_path = Path("data_analysis/processed")
    
    # Load real performance metrics
    metrics_file = base_path / "continuous_performance_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Load real calibration data
    calibration_file = base_path / "continuous_calibration_real.pkl"
    if calibration_file.exists():
        with open(calibration_file, 'rb') as f:
            data['calibration'] = pickle.load(f)
    
    # Load real CV results
    cv_file = base_path / "continuous_cv_results_real.csv"
    if cv_file.exists():
        data['cv_results'] = pd.read_csv(cv_file)
    
    # Load real labels
    labels_file = base_path / "continuous_labels_real.csv"
    if labels_file.exists():
        data['labels'] = pd.read_csv(labels_file)
    
    # Load real score statistics
    stats_file = base_path / "continuous_score_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            data['score_stats'] = json.load(f)
    
    return data

def create_calibration_plot(data, output_dir):
    """Create calibration plot using real data"""
    if 'calibration' not in data:
        print("No calibration data available")
        return
    
    calibration = data['calibration']
    true_scores = calibration['true_scores']
    pred_scores = calibration['predicted_scores']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(pred_scores, true_scores, alpha=0.6, s=20)
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax1.set_xlabel('Predicted Quality Score')
    ax1.set_ylabel('True Quality Score')
    ax1.set_title('Calibration: Predicted vs True Quality Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = calibration['residuals']
    ax2.scatter(pred_scores, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Quality Score')
    ax2.set_ylabel('Residuals (True - Predicted)')
    ax2.set_title('Prediction Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "continuous_calibration.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: continuous_calibration.png")

def create_cv_performance_plot(data, output_dir):
    """Create CV performance plot using real fold metrics"""
    if 'metrics' not in data or 'fold_metrics' not in data['metrics']:
        print("No CV results data available")
        return
    
    fold_metrics = data['metrics']['fold_metrics']
    df = pd.DataFrame(fold_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MAE across folds
    axes[0,0].bar(df['fold'], df['mae'])
    axes[0,0].set_title('Mean Absolute Error by Fold')
    axes[0,0].set_xlabel('Fold')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].grid(True, alpha=0.3)
    
    # RMSE across folds
    axes[0,1].bar(df['fold'], df['rmse'])
    axes[0,1].set_title('Root Mean Square Error by Fold')
    axes[0,1].set_xlabel('Fold')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].grid(True, alpha=0.3)
    
    # Correlation across folds
    axes[1,0].bar(df['fold'], df['correlation'])
    axes[1,0].set_title('Correlation by Fold')
    axes[1,0].set_xlabel('Fold')
    axes[1,0].set_ylabel('Correlation')
    axes[1,0].grid(True, alpha=0.3)
    
    # Summary statistics
    metrics_summary = {
        'MAE': f"{data['metrics']['overall_mae']:.4f} ± {df['mae'].std():.4f}",
        'RMSE': f"{data['metrics']['overall_rmse']:.4f} ± {df['rmse'].std():.4f}",
        'Correlation': f"{data['metrics']['overall_correlation']:.4f} ± {df['correlation'].std():.4f}",
        'Samples': data['metrics']['total_samples']
    }
    
    text_str = '\n'.join([f"{k}: {v}" for k, v in metrics_summary.items()])
    axes[1,1].text(0.1, 0.7, text_str, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1,1].set_title('Performance Summary')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "continuous_cv_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: continuous_cv_performance.png")

def create_performance_summary(data, output_dir):
    """Create performance summary plot"""
    if 'metrics' not in data:
        print("No metrics data available")
        return
    
    metrics = data['metrics']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create performance summary visualization
    performance_data = {
        'Overall MAE': metrics['overall_mae'],
        'Overall RMSE': metrics['overall_rmse'], 
        'Overall Correlation': metrics['overall_correlation']
    }
    
    if 'fold_metrics' in metrics:
        fold_df = pd.DataFrame(metrics['fold_metrics'])
        performance_data.update({
            'MAE Std': fold_df['mae'].std(),
            'RMSE Std': fold_df['rmse'].std(),
            'Correlation Std': fold_df['correlation'].std()
        })
    
    y_pos = np.arange(len(performance_data))
    values = list(performance_data.values())
    labels = list(performance_data.keys())
    
    bars = ax.barh(y_pos, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'yellow'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value')
    ax.set_title('Continuous Model Performance Summary\n(Real Data Results)')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "continuous_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: continuous_performance_summary.png")

def create_score_distribution_plot(data, output_dir):
    """Create score distribution plot using real data"""
    if 'labels' not in data:
        print("No labels data available")
        return
    
    scores = data['labels']['label'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Real Quality Scores')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(scores, vert=True)
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Real Quality Score Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if 'score_stats' in data:
        stats = data['score_stats']
        stats_text = f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}\nMedian: {stats['median']:.3f}\nMin: {stats['min']:.3f}\nMax: {stats['max']:.3f}"
        ax2.text(1.1, 0.7, stats_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "continuous_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: continuous_score_distribution.png")

def main():
    """Main function to create all continuous model plots"""
    print("📊 Creating Continuous Model Plots")
    print("="*50)
    
    # Setup output directory
    output_dir = Path("plotting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Create plots
    print("Creating calibration plot...")
    create_calibration_plot(data, output_dir)
    
    print("Creating CV performance plot...")
    create_cv_performance_plot(data, output_dir)
    
    print("Creating performance summary...")
    create_performance_summary(data, output_dir)
    
    print("Creating score distribution plot...")
    create_score_distribution_plot(data, output_dir)
    
    print(f"\n✅ All continuous model plots saved to {output_dir}")
    
    # List generated files
    generated_files = [f for f in output_dir.glob("continuous_*.png")]
    if generated_files:
        print("Generated files:")
        for f in generated_files:
            print(f"  - {f.name}")

if __name__ == "__main__":
    main() 