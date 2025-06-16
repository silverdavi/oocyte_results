"""
Master Data Analysis Script
==========================

This script runs all data analyses in sequence and creates
a comprehensive set of processed data files for plotting.

Usage: python data_analysis/run_all_analyses.py
"""

import subprocess
import sys
from pathlib import Path
import time

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
    """Check if required files and directories exist"""
    
    print("Checking prerequisites...")
    
    required_files = [
        "continuous_labels/blastulation_quality_scores.csv",
        "continuous_labels/continuous_cross_validation_results.csv",
        "binary_labels/blastulation_binary_labels.csv", 
        "binary_labels/binary_cross_validation_results.csv",
        "calculator/parametric_cycle_calculator.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some input files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("Scripts will use synthetic data for demonstration.")
    else:
        print("✅ All required input files found.")
    
    # Create output directory
    output_dir = Path("data_analysis/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}")
    
    return True

def main():
    """Run all data analyses"""
    
    print("🔬 Starting Comprehensive Data Analysis")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Exiting.")
        return False
    
    # List of analyses to run
    analyses = [
        {
            'script': 'data_analysis/analyze_continuous_model.py',
            'description': 'Continuous Model Performance Analysis'
        },
        {
            'script': 'data_analysis/analyze_binary_model.py', 
            'description': 'Binary Classification Analysis'
        },
        {
            'script': 'data_analysis/analyze_parametric_calculator.py',
            'description': 'Parametric Calculator Validation'
        }
    ]
    
    # Run each analysis
    results = []
    total_start = time.time()
    
    for analysis in analyses:
        success = run_script(analysis['script'], analysis['description'])
        results.append({
            'analysis': analysis['description'],
            'success': success
        })
        
        if not success:
            print(f"\n⚠️  Warning: {analysis['description']} failed but continuing with remaining analyses...")
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print()
    
    successful = 0
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{status}: {result['analysis']}")
        if result['success']:
            successful += 1
    
    print(f"\nCompleted: {successful}/{len(results)} analyses successful")
    
    # List generated files
    output_dir = Path("data_analysis/processed")
    if output_dir.exists():
        output_files = list(output_dir.glob("*"))
        if output_files:
            print(f"\n📁 Generated files in {output_dir}:")
            for file_path in sorted(output_files):
                size_kb = file_path.stat().st_size / 1024
                print(f"  - {file_path.name} ({size_kb:.1f} KB)")
        else:
            print(f"\n⚠️  No files found in {output_dir}")
    
    print("\n🎯 Next steps:")
    print("1. Review the generated processed data files")
    print("2. Run plotting scripts to create figures")
    print("3. Generate paper figures for LaTeX document")
    
    return successful == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 