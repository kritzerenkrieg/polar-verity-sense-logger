import pandas as pd
import numpy as np
import os
import glob
import re
from collections import defaultdict

def discover_all_csv_files(root_directory):
    """
    Discover all CSV files in directory and subdirectories
    
    Parameters:
    root_directory: str - Root directory to search for CSV files
    
    Returns:
    list: List of all CSV file paths
    """
    
    print(f"🔍 Searching for CSV files in: {os.path.abspath(root_directory)}")
    
    # Search for all CSV files recursively
    search_pattern = os.path.join(root_directory, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} CSV files total")
    
    return csv_files

def check_csv_sample_count(file_path):
    """
    Check the number of samples (rows) in a CSV file
    
    Parameters:
    file_path: str - Path to CSV file
    
    Returns:
    dict: Information about the file's sample count
    """
    
    try:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        
        result = {
            'file_path': file_path,
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'success': True,
            'error': None,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'sample_density': 'High' if len(df) > 10000 else 'Medium' if len(df) > 1000 else 'Low'
        }
        
        return result
        
    except Exception as e:
        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'total_rows': 0,
            'total_columns': 0,
            'success': False,
            'error': str(e),
            'file_size_mb': 0,
            'sample_density': 'Unknown'
        }

def categorize_csv_by_type(csv_files):
    """
    Categorize CSV files by their likely type based on filename patterns
    """
    categories = {
        'hrv_time_truth': [],
        'hrv_time_lie': [],
        'hrv_freq_truth': [],
        'hrv_freq_lie': [],
        'other': []
    }
    
    for file_path in csv_files:
        filename = os.path.basename(file_path).lower()
        
        if 'truth' in filename and 'frequency' in filename:
            categories['hrv_freq_truth'].append(file_path)
        elif 'lie' in filename and 'frequency' in filename:
            categories['hrv_freq_lie'].append(file_path)
        elif 'truth' in filename and ('windowed' in filename or 'time' in filename):
            categories['hrv_time_truth'].append(file_path)
        elif 'lie' in filename and ('windowed' in filename or 'time' in filename):
            categories['hrv_time_lie'].append(file_path)
        else:
            categories['other'].append(file_path)
    
    return categories

def check_all_csv_files_samples(root_directory=".", min_samples=100):
    """
    Check sample count of all CSV files in directory and subdirectories
    
    Parameters:
    root_directory: str - Root directory to search for CSV files
    min_samples: int - Minimum number of samples considered adequate
    
    Returns:
    dict: Complete analysis results
    """
    
    print("🔍 CSV FILE SAMPLE COUNT CHECKER")
    print("=" * 50)
    print("Checking each CSV file for number of samples (rows)")
    
    # Discover all CSV files
    csv_files = discover_all_csv_files(root_directory)
    
    if not csv_files:
        print("❌ No CSV files found!")
        return None
    
    # Categorize files by type
    categorized_files = categorize_csv_by_type(csv_files)
    
    # Analyze each file
    all_results = {}
    low_sample_files = []
    error_files = []
    
    print(f"\n📊 Analyzing {len(csv_files)} CSV files...")
    print("-" * 50)
    
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n📄 [{i:3d}/{len(csv_files)}] {filename}")
        
        result = check_csv_sample_count(file_path)
        all_results[file_path] = result
        
        if not result['success']:
            error_files.append(file_path)
            print(f"  ❌ Error: {result['error']}")
            continue
        
        print(f"  📊 Samples: {result['total_rows']:,}")
        print(f"  📋 Columns: {result['total_columns']}")
        print(f"  💾 File Size: {result['file_size_mb']:.2f} MB")
        print(f"  🎯 Density: {result['sample_density']}")
        
        if result['total_rows'] < min_samples:
            low_sample_files.append(file_path)
            print(f"  ⚠️  LOW SAMPLE COUNT (< {min_samples:,})")
        else:
            print(f"  ✅ Adequate sample count")
    
    return {
        'all_results': all_results,
        'low_sample_files': low_sample_files,
        'error_files': error_files,
        'total_files': len(csv_files),
        'categorized_files': categorized_files,
        'min_samples_threshold': min_samples
    }

def generate_detailed_report(results, output_file=None):
    """
    Generate a comprehensive report of CSV file sample count analysis
    """
    
    if not results:
        print("❌ No results to report")
        return
    
    print(f"\n" + "=" * 60)
    print("📋 COMPREHENSIVE CSV SAMPLE COUNT REPORT")
    print("=" * 60)
    
    all_results = results['all_results']
    low_sample_files = results['low_sample_files']
    error_files = results['error_files']
    total_files = results['total_files']
    min_samples = results['min_samples_threshold']
    
    # Overall statistics
    successful_files = total_files - len(error_files)
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"  Total CSV files: {total_files}")
    print(f"  Successfully analyzed: {successful_files}")
    print(f"  Files with errors: {len(error_files)}")
    print(f"  Files with low sample count (< {min_samples:,}): {len(low_sample_files)}")
    
    # Sample count summary
    if successful_files > 0:
        adequate_files = successful_files - len(low_sample_files)
        total_samples = sum(result['total_rows'] for result in all_results.values() if result['success'])
        avg_samples = total_samples / successful_files if successful_files > 0 else 0
        
        print(f"\n🎯 SAMPLE COUNT SUMMARY:")
        print(f"  ✅ Files with adequate samples: {adequate_files}/{successful_files} ({adequate_files/successful_files*100:.1f}%)")
        print(f"  ⚠️  Files with low samples: {len(low_sample_files)}/{successful_files} ({len(low_sample_files)/successful_files*100:.1f}%)")
        print(f"  📈 Total samples across all files: {total_samples:,}")
        print(f"  📊 Average samples per file: {avg_samples:,.0f}")
        
        # Find files with highest and lowest sample counts
        valid_results = [(fp, res) for fp, res in all_results.items() if res['success']]
        if valid_results:
            highest_samples = max(valid_results, key=lambda x: x[1]['total_rows'])
            lowest_samples = min(valid_results, key=lambda x: x[1]['total_rows'])
            
            print(f"  🏆 Highest sample count: {os.path.basename(highest_samples[0])} ({highest_samples[1]['total_rows']:,} samples)")
            print(f"  📉 Lowest sample count: {os.path.basename(lowest_samples[0])} ({lowest_samples[1]['total_rows']:,} samples)")
    
    # Error files
    if error_files:
        print(f"\n❌ FILES WITH ERRORS:")
        print("-" * 30)
        for file_path in error_files:
            filename = os.path.basename(file_path)
            error_msg = all_results[file_path]['error']
            print(f"  📄 {filename}")
            print(f"      Error: {error_msg}")
    
    # Low sample files
    if low_sample_files:
        print(f"\n⚠️  FILES WITH LOW SAMPLE COUNTS:")
        print("-" * 40)
        
        for file_path in low_sample_files:
            result = all_results[file_path]
            filename = os.path.basename(file_path)
            
            print(f"  📄 {filename}")
            print(f"      Samples: {result['total_rows']:,}")
            print(f"      Columns: {result['total_columns']}")
            print(f"      File Size: {result['file_size_mb']:.2f} MB")
    else:
        print(f"\n🎉 ALL CSV FILES HAVE ADEQUATE SAMPLE COUNTS!")
    
    # File type summary
    categorized = results['categorized_files']
    print(f"\n📁 FILE TYPE SUMMARY:")
    for category, files in categorized.items():
        if files:
            total_samples_category = sum(all_results[f]['total_rows'] for f in files if all_results[f]['success'])
            low_in_category = [f for f in files if f in low_sample_files]
            print(f"  {category.upper()}: {len(files)} files, {total_samples_category:,} total samples")
            if low_in_category:
                print(f"    ⚠️  {len(low_in_category)} with low sample counts")
    
    # Sample density breakdown
    density_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for result in all_results.values():
        if result['success']:
            density_counts[result['sample_density']] += 1
    
    print(f"\n📊 SAMPLE DENSITY BREAKDOWN:")
    print(f"  🔥 High density (>10K samples): {density_counts['High']} files")
    print(f"  🔶 Medium density (1K-10K samples): {density_counts['Medium']} files")
    print(f"  🔸 Low density (<1K samples): {density_counts['Low']} files")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if low_sample_files:
        print(f"  1. Consider collecting more data for files with low sample counts")
        print(f"  2. Verify if low sample counts are expected for specific datasets")
        print(f"  3. For machine learning: ensure sufficient samples for train/validation/test splits")
        print(f"  4. Consider data augmentation techniques if appropriate")
        print(f"  5. Use cross-validation for small datasets")
    else:
        print(f"  ✅ Your CSV files have adequate sample counts!")
    
    if error_files:
        print(f"  6. Fix errors in {len(error_files)} files before analysis")
    
    # Save detailed report
    if output_file:
        try:
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename', 'File_Path', 'Total_Samples', 'Total_Columns', 
                               'File_Size_MB', 'Sample_Density', 'Low_Sample_Count', 
                               'Error', 'Success'])
                
                for file_path, result in all_results.items():
                    filename = os.path.basename(file_path)
                    
                    if result['success']:
                        writer.writerow([
                            filename,
                            file_path,
                            result['total_rows'],
                            result['total_columns'],
                            round(result['file_size_mb'], 2),
                            result['sample_density'],
                            file_path in low_sample_files,
                            '',
                            True
                        ])
                    else:
                        writer.writerow([
                            filename,
                            file_path,
                            0,
                            0,
                            0,
                            'Unknown',
                            False,
                            result['error'],
                            False
                        ])
            
            print(f"\n💾 Detailed report saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save report: {e}")

def main(root_directory=".", min_samples=100, save_report=True):
    """
    Main function to check CSV file sample counts
    
    Parameters:
    root_directory: str - Directory to search for CSV files
    min_samples: int - Minimum number of samples considered adequate
    save_report: bool - Whether to save detailed report
    """
    
    # Check sample counts
    results = check_all_csv_files_samples(root_directory, min_samples)
    
    if results:
        # Generate report
        output_file = "csv_sample_count_report.csv" if save_report else None
        generate_detailed_report(results, output_file)
        
        return results
    else:
        return None

# Convenience functions
def quick_sample_check(directory=".", min_samples=100):
    """Quick sample count check with default settings"""
    return main(directory, min_samples)

def strict_sample_check(directory=".", min_samples=1000):
    """More strict sample count check (1000 minimum samples)"""
    return main(directory, min_samples)

def check_specific_csv(file_path):
    """Check sample count of a specific CSV file"""
    print(f"🔍 Checking sample count of: {os.path.basename(file_path)}")
    result = check_csv_sample_count(file_path)
    
    if result['success']:
        print(f"📊 Samples: {result['total_rows']:,}")
        print(f"📋 Columns: {result['total_columns']}")
        print(f"💾 File Size: {result['file_size_mb']:.2f} MB")
        print(f"🎯 Density: {result['sample_density']}")
        
        return result
    else:
        print(f"❌ Error: {result['error']}")
        return None

if __name__ == "__main__":
    print("🚀 Starting CSV File Sample Count Check...")
    results = quick_sample_check(".")
    
    if results and results['low_sample_files']:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"Found {len(results['low_sample_files'])} CSV files with low sample counts:")
        for file_path in results['low_sample_files']:
            filename = os.path.basename(file_path)
            sample_count = results['all_results'][file_path]['total_rows']
            print(f"  ⚠️  {filename} ({sample_count:,} samples)")
    else:
        print(f"\n🎉 All CSV files have adequate sample counts!")