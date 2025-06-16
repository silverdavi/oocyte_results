#!/usr/bin/env python3
"""
Analysis of parametric cycle calculator using REAL DATA ONLY.
"""

import pandas as pd
import numpy as np
import pickle
import os
import importlib.util
import json

def load_real_calculator():
    """Load the actual parametric calculator module."""
    calculator_path = 'calculator/parametric_cycle_calculator.py'
    if os.path.exists(calculator_path):
        spec = importlib.util.spec_from_file_location("calculator", calculator_path)
        calculator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calculator_module)
        return calculator_module
    else:
        print("Calculator module not found")
        return None

def load_real_patient_data():
    """Load actual patient data if available."""
    patient_data_files = [
        'calculator/patient_data.csv',
        'calculator/cycle_data.csv',
        'calculator/amh_data.csv'
    ]
    
    loaded_data = {}
    for file_path in patient_data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path).replace('.csv', '')
            loaded_data[file_name] = df
            print(f"Loaded {len(df)} records from {file_name}")
    
    return loaded_data

def analyze_calculator_performance():
    """Analyze the parametric calculator using real data and parameters."""
    
    # Load the actual calculator
    calculator = load_real_calculator()
    
    # Load real patient data
    patient_data = load_real_patient_data()
    
    if calculator is None:
        print("No calculator found, creating basic analysis")
        performance_metrics = {
            'note': 'Calculator module not found',
            'status': 'missing'
        }
    else:
        # Extract real calculator parameters if available
        calculator_params = {}
        
        # Try to get parameters from the calculator module
        if hasattr(calculator, '__dict__'):
            for attr_name, attr_value in calculator.__dict__.items():
                if not attr_name.startswith('_') and isinstance(attr_value, (int, float, list, dict)):
                    calculator_params[attr_name] = attr_value
        
        # Analyze real AMH data if available
        amh_analysis = {}
        if 'amh_data' in patient_data:
            amh_df = patient_data['amh_data']
            if 'age' in amh_df.columns and 'amh' in amh_df.columns:
                amh_analysis = {
                    'age_range': [amh_df['age'].min(), amh_df['age'].max()],
                    'amh_range': [amh_df['amh'].min(), amh_df['amh'].max()],
                    'age_amh_correlation': np.corrcoef(amh_df['age'], amh_df['amh'])[0, 1],
                    'sample_size': len(amh_df)
                }
        
        # Analyze cycle data if available
        cycle_analysis = {}
        if 'cycle_data' in patient_data:
            cycle_df = patient_data['cycle_data']
            if 'oocytes_retrieved' in cycle_df.columns:
                cycle_analysis = {
                    'oocyte_range': [cycle_df['oocytes_retrieved'].min(), cycle_df['oocytes_retrieved'].max()],
                    'mean_oocytes': cycle_df['oocytes_retrieved'].mean(),
                    'sample_size': len(cycle_df)
                }
                
                if 'live_births' in cycle_df.columns:
                    success_rate = cycle_df['live_births'].sum() / len(cycle_df)
                    cycle_analysis['overall_success_rate'] = success_rate
        
        # Test calculator functions if they exist
        function_tests = {}
        if hasattr(calculator, 'predict_oocytes'):
            try:
                # Test with sample ages
                test_ages = [25, 30, 35, 40, 45]
                oocyte_predictions = [calculator.predict_oocytes(age) for age in test_ages]
                function_tests['oocyte_prediction'] = {
                    'test_ages': test_ages,
                    'predictions': oocyte_predictions,
                    'status': 'working'
                }
            except Exception as e:
                function_tests['oocyte_prediction'] = {'status': 'error', 'error': str(e)}
        
        if hasattr(calculator, 'predict_live_birth_probability'):
            try:
                # Test with sample parameters
                test_params = [(25, 10), (35, 8), (40, 5)]
                lb_predictions = [calculator.predict_live_birth_probability(age, oocytes) for age, oocytes in test_params]
                function_tests['live_birth_prediction'] = {
                    'test_params': test_params,
                    'predictions': lb_predictions,
                    'status': 'working'
                }
            except Exception as e:
                function_tests['live_birth_prediction'] = {'status': 'error', 'error': str(e)}
        
        performance_metrics = {
            'calculator_params': calculator_params,
            'amh_analysis': amh_analysis,
            'cycle_analysis': cycle_analysis,
            'function_tests': function_tests,
            'data_files_found': list(patient_data.keys()),
            'status': 'analyzed'
        }
    
    # Save processed results
    os.makedirs('data_analysis/processed', exist_ok=True)
    
    # Save calculator analysis
    with open('data_analysis/processed/calculator_analysis.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Save real patient data for plotting
    for data_name, df in patient_data.items():
        df.to_csv(f'data_analysis/processed/{data_name}_real.csv', index=False)
    
    print(f"✓ Calculator analysis complete using real data")
    print(f"✓ Found {len(patient_data)} data files")
    print(f"✓ Calculator status: {performance_metrics.get('status', 'unknown')}")
    
    return performance_metrics

def create_age_stratified_analysis():
    """Create age-stratified analysis of real data."""
    
    patient_data = load_real_patient_data()
    
    if 'cycle_data' in patient_data:
        cycle_df = patient_data['cycle_data']
        
        if 'age' in cycle_df.columns:
            # Create age bins
            age_bins = [20, 25, 30, 35, 40, 45, 50]
            cycle_df['age_group'] = pd.cut(cycle_df['age'], bins=age_bins, labels=[f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)])
            
            # Analyze by age group
            age_analysis = cycle_df.groupby('age_group').agg({
                'oocytes_retrieved': ['mean', 'std', 'count'] if 'oocytes_retrieved' in cycle_df.columns else [],
                'live_births': ['sum', 'mean'] if 'live_births' in cycle_df.columns else []
            }).round(3)
            
            # Save age-stratified analysis
            age_analysis.to_csv('data_analysis/processed/age_stratified_analysis_real.csv')
            
            print(f"✓ Age-stratified analysis created with {len(age_analysis)} age groups")

if __name__ == "__main__":
    analyze_calculator_performance()
    create_age_stratified_analysis() 