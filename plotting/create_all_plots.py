"""
Master Plotting Script
======================

This script runs all plotting modules and prepares publication-ready
figures for the LaTeX paper.

Usage: python plotting/create_all_plots.py
"""

import subprocess
import sys
from pathlib import Path
import time
import shutil

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=".")
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ SUCCESS: {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ ERROR: {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: Failed to run {description}: {str(e)}")
        return False

def check_prerequisites():
    """Check if processed data files exist"""
    
    print("Checking prerequisites...")
    
    data_dir = Path("data_analysis/processed")
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        print("Please run data analysis scripts first: python data_analysis/run_all_analyses.py")
        return False
    
    # Check for some key files
    key_files = [
        "continuous_model_metrics.pkl",
        "binary_model_metrics.pkl", 
        "calculator_summary.pkl"
    ]
    
    missing_files = []
    for file_name in key_files:
        if not (data_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print("⚠️  Warning: Some processed data files are missing:")
        for file_name in missing_files:
            print(f"  - {file_name}")
        print("Plots will be created with available data only.")
    else:
        print("✅ All key processed data files found.")
    
    # Create output directory
    output_dir = Path("plotting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}")
    
    return True

def copy_figures_to_latex():
    """Copy generated figures to LaTeX figures directory"""
    
    source_dir = Path("plotting/figures")
    dest_dir = Path("paper_latex/figures")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"❌ Source directory {source_dir} does not exist")
        return False
    
    # Copy all PNG files
    png_files = list(source_dir.glob("*.png"))
    
    if not png_files:
        print(f"⚠️  No PNG files found in {source_dir}")
        return False
    
    print(f"\n📁 Copying figures to LaTeX directory...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    copied_files = []
    for png_file in png_files:
        try:
            dest_file = dest_dir / png_file.name
            shutil.copy2(png_file, dest_file)
            copied_files.append(png_file.name)
            print(f"  ✅ Copied: {png_file.name}")
        except Exception as e:
            print(f"  ❌ Failed to copy {png_file.name}: {e}")
    
    print(f"\n📊 Successfully copied {len(copied_files)} figures to LaTeX directory")
    return len(copied_files) > 0

def create_figure_list():
    """Create a list of all generated figures for reference"""
    
    figures_dir = Path("plotting/figures")
    if not figures_dir.exists():
        return
    
    png_files = sorted(figures_dir.glob("*.png"))
    
    if not png_files:
        return
    
    # Create figure reference file
    reference_file = figures_dir / "figure_list.txt"
    
    with open(reference_file, 'w') as f:
        f.write("Generated Figures for IVF Counseling Paper\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by model type
        continuous_figs = [f for f in png_files if 'continuous' in f.name]
        binary_figs = [f for f in png_files if 'binary' in f.name]
        calculator_figs = [f for f in png_files if 'calculator' in f.name]
        
        if continuous_figs:
            f.write("Continuous Model Figures:\n")
            for fig in continuous_figs:
                f.write(f"  - {fig.name}\n")
            f.write("\n")
        
        if binary_figs:
            f.write("Binary Classification Figures:\n")
            for fig in binary_figs:
                f.write(f"  - {fig.name}\n")
            f.write("\n")
        
        if calculator_figs:
            f.write("Parametric Calculator Figures:\n")
            for fig in calculator_figs:
                f.write(f"  - {fig.name}\n")
            f.write("\n")
        
        f.write(f"Total figures: {len(png_files)}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📋 Figure reference list saved to: {reference_file}")

def main():
    """Run all plotting scripts"""
    
    print("🎨 Starting Comprehensive Figure Generation")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Exiting.")
        return False
    
    # List of plotting scripts to run
    plots = [
        {
            'script': 'plotting/plot_continuous_model.py',
            'description': 'Continuous Model Visualizations'
        },
        {
            'script': 'plotting/plot_binary_model.py', 
            'description': 'Binary Classification Visualizations'
        },
        {
            'script': 'plotting/plot_parametric_calculator.py',
            'description': 'Parametric Calculator Visualizations'
        }
    ]
    
    # Run each plotting script
    results = []
    total_start = time.time()
    
    for plot in plots:
        success = run_script(plot['script'], plot['description'])
        results.append({
            'plot': plot['description'],
            'success': success
        })
        
        if not success:
            print(f"\n⚠️  Warning: {plot['description']} failed but continuing with remaining plots...")
    
    # Post-processing
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("🎨 PLOTTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print()
    
    successful = 0
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{status}: {result['plot']}")
        if result['success']:
            successful += 1
    
    print(f"\nCompleted: {successful}/{len(results)} plotting scripts successful")
    
    # Copy figures to LaTeX directory
    if successful > 0:
        print("\n📁 Copying figures to LaTeX directory...")
        copy_success = copy_figures_to_latex()
        
        if copy_success:
            print("✅ Figures successfully copied to paper_latex/figures/")
        else:
            print("⚠️  Some figures may not have been copied")
    
    # Create figure reference list
    create_figure_list()
    
    # List generated files
    figures_dir = Path("plotting/figures")
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.png"))
        if figure_files:
            print(f"\n📊 Generated {len(figure_files)} figures:")
            for fig_file in sorted(figure_files):
                size_kb = fig_file.stat().st_size / 1024
                print(f"  - {fig_file.name} ({size_kb:.1f} KB)")
        else:
            print(f"\n⚠️  No figures found in {figures_dir}")
    
    print("\n🎯 Next steps:")
    print("1. Review generated figures in plotting/figures/")
    print("2. Check copied figures in paper_latex/figures/")
    print("3. Compile LaTeX document to include figures")
    print("4. Adjust figure references in LaTeX as needed")
    
    return successful == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 