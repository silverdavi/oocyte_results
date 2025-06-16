#!/usr/bin/env python3
"""
Analysis of continuous blastulation quality prediction model using REAL DATA ONLY.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import json

def load_real_data():
    """Load the actual continuous labels data."""
    continuous_df = pd.read_csv('continuous_labels/blastulation_quality_scores.csv')
    print(f"Loaded {len(continuous_df)} real continuous samples")
    return continuous_df

def analyze_continuous_predictions():
    """Analyze the continuous model performance using real cross-validation results."""
    # Load real data
    continuous_df = load_real_data()
    
    # Load actual cross-validation results if they exist
    cv_results_file = 'continuous_labels/continuous_cross_validation_results.csv'
    
    if os.path.exists(cv_results_file):
        cv_results = pd.read_csv(cv_results_file)
        print(f"Found actual CV results with {len(cv_results)} predictions")
        
        # Calculate real performance metrics using correct column names
        mae = mean_absolute_error(cv_results['label'], cv_results['pred'])
        rmse = np.sqrt(mean_squared_error(cv_results['label'], cv_results['pred']))
        correlation = np.corrcoef(cv_results['label'], cv_results['pred'])[0, 1]
        
        # Calculate fold-wise performance using individual fold columns
        fold_cols = [col for col in cv_results.columns if col.startswith('fold')]
        fold_metrics = []
        
        for i, fold_col in enumerate(fold_cols):
            fold_mae = mean_absolute_error(cv_results['label'], cv_results[fold_col])
            fold_rmse = np.sqrt(mean_squared_error(cv_results['label'], cv_results[fold_col]))
            fold_corr = np.corrcoef(cv_results['label'], cv_results[fold_col])[0, 1]
            
            fold_metrics.append({
                'fold': i + 1,
                'mae': fold_mae,
                'rmse': fold_rmse,
                'correlation': fold_corr,
                'n_samples': len(cv_results)
            })
        
        performance_metrics = {
            'overall_mae': mae,
            'overall_rmse': rmse,
            'overall_correlation': correlation,
            'fold_metrics': fold_metrics,
            'total_samples': len(cv_results)
        }
    else:
        print("No cross-validation results found. Using basic data statistics.")
        # Basic statistics from the actual labels
        performance_metrics = {
            'data_mean': continuous_df['label'].mean(),
            'data_std': continuous_df['label'].std(),
            'data_min': continuous_df['label'].min(),
            'data_max': continuous_df['label'].max(),
            'total_samples': len(continuous_df),
            'note': 'No model predictions available - showing data statistics only'
        }
    
    # Analyze score distribution from real data
    scores = continuous_df['label'].values
    score_stats = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75),
        'min': np.min(scores),
        'max': np.max(scores)
    }
    
    # Save processed results
    os.makedirs('data_analysis/processed', exist_ok=True)
    
    # Save performance metrics
    with open('data_analysis/processed/continuous_performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Save score statistics
    with open('data_analysis/processed/continuous_score_statistics.json', 'w') as f:
        json.dump(score_stats, f, indent=2)
    
    # Save raw data for plotting
    continuous_df.to_csv('data_analysis/processed/continuous_labels_real.csv', index=False)
    
    if os.path.exists(cv_results_file):
        cv_results.to_csv('data_analysis/processed/continuous_cv_results_real.csv', index=False)
        
        # Calculate calibration data (real vs predicted) using correct column names
        calibration_data = {
            'true_scores': cv_results['label'].tolist(),
            'predicted_scores': cv_results['pred'].tolist(),
            'residuals': (cv_results['label'] - cv_results['pred']).tolist()
        }
        
        with open('data_analysis/processed/continuous_calibration_real.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
    
    print(f"✓ Continuous model analysis complete using real data")
    print(f"✓ Processed {len(continuous_df)} real samples")
    return performance_metrics

if __name__ == "__main__":
    analyze_continuous_predictions() 