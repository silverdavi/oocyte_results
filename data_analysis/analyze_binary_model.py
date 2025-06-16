#!/usr/bin/env python3
"""
Analysis of binary blastulation prediction model using REAL DATA ONLY.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
import json

def load_real_data():
    """Load the actual binary labels data."""
    binary_df = pd.read_csv('binary_labels/blastulation_binary_labels.csv')
    print(f"Loaded {len(binary_df)} real binary samples")
    return binary_df

def analyze_binary_predictions():
    """Analyze the binary model performance using real cross-validation results."""
    # Load real data
    binary_df = load_real_data()
    
    # Load actual cross-validation results if they exist
    cv_results_file = 'binary_labels/binary_cross_validation_results.csv'
    
    if os.path.exists(cv_results_file):
        cv_results = pd.read_csv(cv_results_file)
        print(f"Found actual CV results with {len(cv_results)} predictions")
        
        # Convert probabilities to binary predictions (threshold at 0.5)
        cv_results['predicted_binary'] = (cv_results['mean'] > 0.5).astype(int)
        
        # Calculate real performance metrics using correct column names
        accuracy = accuracy_score(cv_results['label'], cv_results['predicted_binary'])
        precision = precision_score(cv_results['label'], cv_results['predicted_binary'])
        recall = recall_score(cv_results['label'], cv_results['predicted_binary'])
        f1 = f1_score(cv_results['label'], cv_results['predicted_binary'])
        
        # Calculate AUC using probability scores
        auc = roc_auc_score(cv_results['label'], cv_results['mean'])
        # Calculate ROC curve data
        fpr, tpr, thresholds = roc_curve(cv_results['label'], cv_results['mean'])
        roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            cv_results['label'], cv_results['mean'])
        pr_data = {
            'precision': precision_curve.tolist(), 
            'recall': recall_curve.tolist(), 
            'thresholds': pr_thresholds.tolist()
        }
        
        # Confusion matrix
        cm = confusion_matrix(cv_results['label'], cv_results['predicted_binary'])
        
        # Calculate fold-wise performance using individual fold columns
        fold_cols = [col for col in cv_results.columns if col.startswith('fold')]
        fold_metrics = []
        
        for i, fold_col in enumerate(fold_cols):
            fold_pred_binary = (cv_results[fold_col] > 0.5).astype(int)
            fold_acc = accuracy_score(cv_results['label'], fold_pred_binary)
            fold_prec = precision_score(cv_results['label'], fold_pred_binary)
            fold_rec = recall_score(cv_results['label'], fold_pred_binary)
            fold_f1 = f1_score(cv_results['label'], fold_pred_binary)
            fold_auc = roc_auc_score(cv_results['label'], cv_results[fold_col])
            
            fold_metrics.append({
                'fold': i + 1,
                'accuracy': fold_acc,
                'precision': fold_prec,
                'recall': fold_rec,
                'f1': fold_f1,
                'auc': fold_auc,
                'n_samples': len(cv_results)
            })
        
        performance_metrics = {
            'overall_accuracy': accuracy,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'overall_auc': auc,
            'confusion_matrix': cm.tolist(),
            'fold_metrics': fold_metrics,
            'total_samples': len(cv_results),
            'roc_curve': roc_data,
            'pr_curve': pr_data
        }
            
    else:
        print("No cross-validation results found. Using basic data statistics.")
        # Basic statistics from the actual labels
        class_counts = binary_df['label'].value_counts()
        performance_metrics = {
            'class_distribution': class_counts.to_dict(),
            'positive_rate': class_counts.get(1, 0) / len(binary_df),
            'negative_rate': class_counts.get(0, 0) / len(binary_df),
            'total_samples': len(binary_df),
            'note': 'No model predictions available - showing data statistics only'
        }
    
    # Analyze label distribution from real data
    labels = binary_df['label'].values
    label_stats = {
        'positive_count': int(np.sum(labels == 1)),
        'negative_count': int(np.sum(labels == 0)),
        'positive_rate': float(np.mean(labels)),
        'class_balance': float(np.sum(labels == 1) / np.sum(labels == 0)) if np.sum(labels == 0) > 0 else float('inf')
    }
    
    # Save processed results
    os.makedirs('data_analysis/processed', exist_ok=True)
    
    # Save performance metrics
    with open('data_analysis/processed/binary_performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Save label statistics
    with open('data_analysis/processed/binary_label_statistics.json', 'w') as f:
        json.dump(label_stats, f, indent=2)
    
    # Save raw data for plotting
    binary_df.to_csv('data_analysis/processed/binary_labels_real.csv', index=False)
    
    if os.path.exists(cv_results_file):
        cv_results.to_csv('data_analysis/processed/binary_cv_results_real.csv', index=False)
        
        # Save confusion matrix and ROC data for plotting
        confusion_data = {
            'confusion_matrix': cm.tolist(),
            'class_names': ['Not Successful', 'Successful']
        }
        
        with open('data_analysis/processed/binary_confusion_real.pkl', 'wb') as f:
            pickle.dump(confusion_data, f)
        
        with open('data_analysis/processed/binary_roc_real.pkl', 'wb') as f:
            pickle.dump(roc_data, f)
    
    print(f"✓ Binary model analysis complete using real data")
    print(f"✓ Processed {len(binary_df)} real samples")
    return performance_metrics

if __name__ == "__main__":
    analyze_binary_predictions() 