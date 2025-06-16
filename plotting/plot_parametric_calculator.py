"""
Plot Parametric Calculator Results
=================================

This script creates publication-ready plots for the parametric cycle
calculator using processed data.

Input:  data_analysis/processed/calculator_*.csv/pkl
Output: plotting/figures/calculator_*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_figure_style():
    """Set up consistent figure styling"""
    
    # Font settings for publication
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

def load_processed_data():
    """Load all processed calculator data"""
    
    data_dir = Path("data_analysis/processed")
    
    data = {}
    
    # Load validation data
    try:
        data['validation'] = pd.read_csv(data_dir / "calculator_validation.csv")
    except FileNotFoundError:
        print("Warning: calculator_validation.csv not found")
        data['validation'] = None
    
    # Load attrition funnel data
    try:
        data['attrition'] = pd.read_csv(data_dir / "attrition_model_data.csv")
    except FileNotFoundError:
        print("Warning: attrition_model_data.csv not found")
        data['attrition'] = None
    
    # Load AMH analysis
    try:
        data['amh_analysis'] = pd.read_csv(data_dir / "amh_age_analysis.csv")
    except FileNotFoundError:
        print("Warning: amh_age_analysis.csv not found")
        data['amh_analysis'] = None
    
    # Load patient cohort
    try:
        data['patients'] = pd.read_csv(data_dir / "synthetic_patient_cohort.csv")
    except FileNotFoundError:
        print("Warning: synthetic_patient_cohort.csv not found")
        data['patients'] = None
    
    # Load outcomes
    try:
        data['outcomes'] = pd.read_csv(data_dir / "simulated_cycle_outcomes.csv")
    except FileNotFoundError:
        print("Warning: simulated_cycle_outcomes.csv not found")
        data['outcomes'] = None
    
    # Load summary
    try:
        with open(data_dir / "calculator_summary.pkl", 'rb') as f:
            data['summary'] = pickle.load(f)
    except FileNotFoundError:
        print("Warning: calculator_summary.pkl not found")
        data['summary'] = None
    
    return data

def plot_attrition_funnel(attrition_df, output_dir):
    """Create attrition funnel visualization"""
    
    if attrition_df is None or len(attrition_df) == 0:
        print("No attrition data available")
        return
    
    # Create funnel plot for different patient profiles
    profiles = attrition_df['profile'].unique()
    stages = ['Retrieved', 'Mature', 'Fertilized', 'Blastocysts', 'Euploid', 'Live Birth']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, profile in enumerate(profiles):
        if i >= len(axes):
            break
            
        ax = axes[i]
        profile_data = attrition_df[attrition_df['profile'] == profile]
        
        # Ensure stages are in correct order
        stage_order = {stage: idx for idx, stage in enumerate(stages)}
        profile_data = profile_data.sort_values('stage', key=lambda x: x.map(stage_order))
        
        x_pos = range(len(profile_data))
        counts = profile_data['expected_count'].values
        
        # Create funnel bars (decreasing width)
        bar_widths = counts / counts[0] if len(counts) > 0 and counts[0] > 0 else [1]*len(counts)
        
        bars = ax.barh(x_pos, counts, height=0.6, color=colors[i], alpha=0.7)
        
        # Add count labels
        for j, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{count:.1f}', va='center', fontweight='bold')
        
        # Add percentage labels
        for j, count in enumerate(counts):
            if j > 0 and counts[0] > 0:
                pct = (count / counts[0]) * 100
                ax.text(0.5, j + 0.15, f'{pct:.0f}%', va='center', ha='left', 
                       fontsize=9, style='italic', color='darkred')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(profile_data['stage'])
        ax.set_xlabel('Expected Count')
        ax.set_title(f'{profile}\n(Age: {profile_data["age"].iloc[0]}, AMH: {profile_data["amh"].iloc[0]:.1f})')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to show funnel from top to bottom
        ax.invert_yaxis()
    
    # Hide unused subplot
    if len(profiles) < len(axes):
        for i in range(len(profiles), len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle('IVF Attrition Funnel by Patient Profile', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "calculator_attrition_funnel.png")
    plt.close()
    
    print("✅ Saved: calculator_attrition_funnel.png")

def plot_amh_age_relationship(amh_df, patients_df, output_dir):
    """Plot AMH vs age relationship and percentiles"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AMH vs Age scatter plot
    ax1 = axes[0]
    
    if patients_df is not None:
        # Scatter plot of patient data
        ax1.scatter(patients_df['age'], patients_df['amh'], alpha=0.6, s=30, color='lightblue', 
                   label='Synthetic Patients')
        
        # Add trend line
        z = np.polyfit(patients_df['age'], patients_df['amh'], 2)
        p = np.poly1d(z)
        age_trend = np.linspace(patients_df['age'].min(), patients_df['age'].max(), 100)
        ax1.plot(age_trend, p(age_trend), 'r-', linewidth=2, label='Trend Line')
    
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('AMH (ng/mL)')
    ax1.set_title('AMH Levels by Age')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AMH percentiles by age group
    ax2 = axes[1]
    
    if amh_df is not None:
        age_groups = amh_df['age_group'].unique()
        percentiles = [10, 25, 50, 75, 90]
        
        x_pos = np.arange(len(age_groups))
        width = 0.15
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        for i, pct in enumerate(percentiles):
            pct_data = amh_df[amh_df['percentile'] == pct]
            pct_values = [pct_data[pct_data['age_group'] == ag]['amh_value'].iloc[0] 
                         if len(pct_data[pct_data['age_group'] == ag]) > 0 else 0 
                         for ag in age_groups]
            
            ax2.bar(x_pos + i*width, pct_values, width, label=f'{pct}th percentile', 
                   color=colors[i], alpha=0.7)
        
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('AMH (ng/mL)')
        ax2.set_title('AMH Percentiles by Age Group')
        ax2.set_xticks(x_pos + width*2)
        ax2.set_xticklabels(age_groups)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "calculator_amh_age_analysis.png")
    plt.close()
    
    print("✅ Saved: calculator_amh_age_analysis.png")

def plot_validation_performance(validation_df, output_dir):
    """Plot calculator validation performance"""
    
    if validation_df is None or len(validation_df) == 0:
        print("No validation data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Oocyte prediction scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(validation_df['predicted_oocytes'], validation_df['actual_oocytes'], 
               alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(validation_df['predicted_oocytes'].min(), validation_df['actual_oocytes'].min())
    max_val = max(validation_df['predicted_oocytes'].max(), validation_df['actual_oocytes'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    # Calculate correlation
    correlation = np.corrcoef(validation_df['predicted_oocytes'], validation_df['actual_oocytes'])[0, 1]
    ax1.set_xlabel('Predicted Oocytes')
    ax1.set_ylabel('Actual Oocytes')
    ax1.set_title(f'Oocyte Yield Prediction\n(r = {correlation:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Live birth probability scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(validation_df['predicted_live_birth_prob'], validation_df['actual_live_birth_prob'], 
               alpha=0.6, s=30, color='green')
    
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Prediction')
    
    lb_correlation = np.corrcoef(validation_df['predicted_live_birth_prob'], 
                                validation_df['actual_live_birth_prob'])[0, 1]
    ax2.set_xlabel('Predicted Live Birth Probability')
    ax2.set_ylabel('Actual Live Birth Probability')
    ax2.set_title(f'Live Birth Probability\n(r = {lb_correlation:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residuals plot for oocytes
    ax3 = axes[1, 0]
    residuals = validation_df['actual_oocytes'] - validation_df['predicted_oocytes']
    ax3.scatter(validation_df['predicted_oocytes'], residuals, alpha=0.6, s=30)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Predicted Oocytes')
    ax3.set_ylabel('Residuals (Actual - Predicted)')
    ax3.set_title('Oocyte Prediction Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8, label='Zero Error')
    ax4.set_xlabel('Prediction Error (Actual - Predicted)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution\n(MAE: {np.mean(np.abs(residuals)):.2f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Calculator Validation Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "calculator_validation_performance.png")
    plt.close()
    
    print("✅ Saved: calculator_validation_performance.png")

def plot_age_outcomes_relationship(outcomes_df, output_dir):
    """Plot relationship between age and IVF outcomes"""
    
    if outcomes_df is None or len(outcomes_df) == 0:
        print("No outcomes data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Age bins for analysis
    age_bins = [25, 30, 35, 40, 45]
    age_labels = ['25-30', '30-35', '35-40', '40-45']
    outcomes_df['age_group'] = pd.cut(outcomes_df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # Mean outcomes by age group
    ax1 = axes[0, 0]
    age_means = outcomes_df.groupby('age_group').agg({
        'retrieved_oocytes': 'mean',
        'blastocysts': 'mean',
        'euploid_blastocysts': 'mean'
    })
    
    x_pos = np.arange(len(age_labels))
    width = 0.25
    
    ax1.bar(x_pos - width, age_means['retrieved_oocytes'], width, label='Retrieved Oocytes', alpha=0.7)
    ax1.bar(x_pos, age_means['blastocysts'], width, label='Blastocysts', alpha=0.7)
    ax1.bar(x_pos + width, age_means['euploid_blastocysts'], width, label='Euploid Blastocysts', alpha=0.7)
    
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Mean Count')
    ax1.set_title('Mean Outcomes by Age Group')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(age_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Live birth rate by age
    ax2 = axes[0, 1]
    live_birth_rates = outcomes_df.groupby('age_group')['live_birth'].mean()
    
    bars = ax2.bar(age_labels, live_birth_rates, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Live Birth Rate')
    ax2.set_title('Live Birth Rate by Age')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, rate in zip(bars, live_birth_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Age vs Retrieved Oocytes scatter
    ax3 = axes[1, 0]
    ax3.scatter(outcomes_df['age'], outcomes_df['retrieved_oocytes'], alpha=0.6, s=20)
    
    # Trend line
    z = np.polyfit(outcomes_df['age'], outcomes_df['retrieved_oocytes'], 1)
    p = np.poly1d(z)
    age_trend = np.linspace(outcomes_df['age'].min(), outcomes_df['age'].max(), 100)
    ax3.plot(age_trend, p(age_trend), 'r-', linewidth=2)
    
    ax3.set_xlabel('Age (years)')
    ax3.set_ylabel('Retrieved Oocytes')
    ax3.set_title('Age vs Retrieved Oocytes')
    ax3.grid(True, alpha=0.3)
    
    # AMH vs Retrieved Oocytes scatter
    ax4 = axes[1, 1]
    ax4.scatter(outcomes_df['amh'], outcomes_df['retrieved_oocytes'], alpha=0.6, s=20, c=outcomes_df['age'], 
               cmap='viridis')
    
    # Trend line
    z = np.polyfit(outcomes_df['amh'], outcomes_df['retrieved_oocytes'], 1)
    p = np.poly1d(z)
    amh_trend = np.linspace(outcomes_df['amh'].min(), outcomes_df['amh'].max(), 100)
    ax4.plot(amh_trend, p(amh_trend), 'r-', linewidth=2)
    
    ax4.set_xlabel('AMH (ng/mL)')
    ax4.set_ylabel('Retrieved Oocytes')
    ax4.set_title('AMH vs Retrieved Oocytes\n(colored by age)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for age
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Age (years)')
    
    plt.suptitle('Age and AMH Impact on IVF Outcomes', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "calculator_age_outcomes_analysis.png")
    plt.close()
    
    print("✅ Saved: calculator_age_outcomes_analysis.png")

def main():
    """Main plotting pipeline"""
    
    print("📊 Creating Parametric Calculator Plots")
    print("="*50)
    
    # Setup
    setup_figure_style()
    
    # Create output directory
    output_dir = Path("plotting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Create plots
    print("Creating attrition funnel...")
    plot_attrition_funnel(data['attrition'], output_dir)
    
    print("Creating AMH age analysis...")
    plot_amh_age_relationship(data['amh_analysis'], data['patients'], output_dir)
    
    print("Creating validation performance...")
    plot_validation_performance(data['validation'], output_dir)
    
    print("Creating age outcomes analysis...")
    plot_age_outcomes_relationship(data['outcomes'], output_dir)
    
    print(f"\n✅ All calculator plots saved to {output_dir}")
    print("Generated files:")
    plot_files = list(output_dir.glob("calculator_*.png"))
    for file_path in sorted(plot_files):
        print(f"  - {file_path.name}")

if __name__ == "__main__":
    main() 