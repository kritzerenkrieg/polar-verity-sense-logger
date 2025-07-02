import pandas as pd
import numpy as np
import os
import glob
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def discover_csv_files(root_directory):
    """
    Automatically discover all HRV CSV files in directory and subdirectories
    
    Parameters:
    root_directory: str - Root directory to search for CSV files
    
    Returns:
    dict: Organized file paths by subject and data type
    """
    
    # Search patterns for different file types (handle both hyphenated and non-hyphenated)
    patterns = {
        'time_truth': ['*truthsequence_windowed.csv', '*truth-sequence_windowed.csv'],
        'time_lie': ['*liesequence_windowed.csv', '*lie-sequence_windowed.csv'], 
        'freq_truth': ['*truthsequence_frequency_windowed.csv', '*truth-sequence_frequency_windowed.csv'],
        'freq_lie': ['*liesequence_frequency_windowed.csv', '*lie-sequence_frequency_windowed.csv']
    }
    
    print(f"🔍 Searching for CSV files in: {os.path.abspath(root_directory)}")
    
    discovered_files = {}
    
    # Search recursively in all subdirectories
    for data_type, pattern_list in patterns.items():
        files = []
        for pattern in pattern_list:
            search_pattern = os.path.join(root_directory, '**', pattern)
            found_files = glob.glob(search_pattern, recursive=True)
            files.extend(found_files)
        discovered_files[data_type] = files
        print(f"Found {len(files)} {data_type} files")
    
    # Extract subject names from file paths
    subjects = set()
    subject_files = {}
    
    for data_type, files in discovered_files.items():
        for file_path in files:
            # Extract subject name from filename
            filename = os.path.basename(file_path)
            
            # Try different patterns to extract subject name
            patterns_to_try = [
                r'pr_(\w+)_emosi_',     # pr_bayu_emosi_...
                r'pr_(\w+)_etika_',     # pr_bayu_etika_...
                r'pr_(\w+)_kognitif_',  # pr_bayu_kognitif_...
                r'pr_(\w+)_sosial_',    # pr_bayu_sosial_...
                r'(\w+)_emosi_',        # bayu_emosi_...
                r'(\w+)_etika_',        # bayu_etika_...
                r'(\w+)_kognitif_',     # bayu_kognitif_...
                r'(\w+)_sosial_',       # bayu_sosial_...
                r'pr_(\w+)_',           # pr_bayu_...
                r'(\w+)_truth',         # bayu_truth...
                r'(\w+)_lie',           # bayu_lie...
                r'^(\w+)_',             # bayu_...
            ]
            
            subject_name = None
            for pattern in patterns_to_try:
                match = re.search(pattern, filename.lower())
                if match:
                    subject_name = match.group(1)
                    break
            
            if not subject_name:
                # If no pattern matches, use first part of filename
                subject_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
            
            subjects.add(subject_name)
            
            # Initialize subject entry if not exists
            if subject_name not in subject_files:
                subject_files[subject_name] = {}
            
            subject_files[subject_name][data_type] = file_path
    
    print(f"\n📊 Discovered subjects: {sorted(list(subjects))}")
    
    # Validate that each subject has all required file types
    complete_subjects = {}
    required_types = ['time_truth', 'time_lie', 'freq_truth', 'freq_lie']
    
    for subject, files in subject_files.items():
        missing_types = [t for t in required_types if t not in files]
        
        if not missing_types:
            complete_subjects[subject] = files
            print(f"✅ {subject}: Complete dataset")
        else:
            available_types = [t for t in required_types if t in files]
            if len(available_types) >= 2:  # At least time domain data
                complete_subjects[subject] = files
                print(f"⚠️  {subject}: Partial dataset (missing: {', '.join(missing_types)})")
            else:
                print(f"❌ {subject}: Insufficient data (only has: {', '.join(available_types)})")
    
    return complete_subjects

def analyze_hrv_features(subject_name, file_paths):
    """
    Analyze HRV features for a single subject to find effect size, direction, and percent difference
    
    Parameters:
    subject_name: str - Name of the subject (extracted from filename)
    file_paths: dict - Dictionary containing file paths for different data types
    
    Returns:
    dict: Analysis results with effect size, direction, and percent difference for each feature
    """
    
    # Load data with error handling
    data = {}
    data_loaded = {}
    
    # Define which files are required vs optional
    file_mappings = {
        'time_truth': 'Time domain truth data',
        'time_lie': 'Time domain lie data', 
        'freq_truth': 'Frequency domain truth data',
        'freq_lie': 'Frequency domain lie data'
    }
    
    print(f"\n📂 Loading data for {subject_name.upper()}:")
    
    for data_type, description in file_mappings.items():
        if data_type in file_paths:
            try:
                data[data_type] = pd.read_csv(file_paths[data_type])
                data_loaded[data_type] = True
                print(f"  ✅ {description}: {len(data[data_type])} samples")
            except Exception as e:
                print(f"  ❌ {description}: Error loading - {e}")
                data_loaded[data_type] = False
        else:
            print(f"  ⚠️  {description}: File not found")
            data_loaded[data_type] = False
    
    # Check if we have minimum required data (at least time domain)
    if not (data_loaded.get('time_truth', False) and data_loaded.get('time_lie', False)):
        print(f"  ❌ Insufficient data for {subject_name} - need at least time domain truth and lie data")
        return None
    
    # Define features to analyze based on available data
    time_features = ['nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 'sdsd', 
                    'rmssd', 'pnn20', 'pnn50', 'triangular_index']
    
    freq_features = ['vlf_power', 'lf_power', 'hf_power', 'total_power', 
                    'lf_hf_ratio', 'lf_norm', 'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak']
    
    results = {
        'subject': subject_name,
        'time_analysis': {},
        'freq_analysis': {},
        'summary': {},
        'files_loaded': data_loaded
    }
    
    def calculate_feature_stats(truth_data, lie_data, feature_name):
        """Calculate statistics for a single feature"""
        from scipy import stats  # Import here to avoid scope issues
        
        # Extract values and remove NaN/invalid entries
        truth_values = truth_data[feature_name].dropna()
        lie_values = lie_data[feature_name].dropna()
        
        if len(truth_values) == 0 or len(lie_values) == 0:
            return None
            
        # Calculate basic statistics
        truth_mean = truth_values.mean()
        lie_mean = lie_values.mean()
        truth_std = truth_values.std()
        lie_std = lie_values.std()
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((truth_std**2 + lie_std**2) / 2)
        effect_size = abs(truth_mean - lie_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate percent difference (relative to lie condition)
        percent_diff = ((truth_mean - lie_mean) / abs(lie_mean) * 100) if lie_mean != 0 else 0
        
        # Determine direction
        direction = 'truth_higher' if truth_mean > lie_mean else 'lie_higher'
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(truth_values, lie_values)
        
        return {
            'truth_mean': truth_mean,
            'lie_mean': lie_mean,
            'truth_std': truth_std,
            'lie_std': lie_std,
            'difference': truth_mean - lie_mean,
            'effect_size': effect_size,
            'percent_diff': percent_diff,
            'direction': direction,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'truth_samples': len(truth_values),
            'lie_samples': len(lie_values)
        }
    
    # Analyze time domain features
    if data_loaded.get('time_truth', False) and data_loaded.get('time_lie', False):
        print(f"\n  📊 Analyzing Time Domain Features:")
        for feature in time_features:
            if feature in data['time_truth'].columns and feature in data['time_lie'].columns:
                stats_result = calculate_feature_stats(data['time_truth'], data['time_lie'], feature)
                if stats_result:
                    results['time_analysis'][feature] = stats_result
                    
                    # Print formatted results
                    direction_symbol = "T>L" if stats_result['direction'] == 'truth_higher' else "L>T"
                    significance = "*" if stats_result['significant'] else " "
                    print(f"    {feature:15s}: Effect={stats_result['effect_size']:.3f}, "
                          f"{direction_symbol}, {stats_result['percent_diff']:+6.1f}%{significance}")
    
    # Analyze frequency domain features  
    if data_loaded.get('freq_truth', False) and data_loaded.get('freq_lie', False):
        print(f"\n  📊 Analyzing Frequency Domain Features:")
        for feature in freq_features:
            if feature in data['freq_truth'].columns and feature in data['freq_lie'].columns:
                stats_result = calculate_feature_stats(data['freq_truth'], data['freq_lie'], feature)
                if stats_result:
                    results['freq_analysis'][feature] = stats_result
                    
                    # Print formatted results
                    direction_symbol = "T>L" if stats_result['direction'] == 'truth_higher' else "L>T"
                    significance = "*" if stats_result['significant'] else " "
                    print(f"    {feature:15s}: Effect={stats_result['effect_size']:.3f}, "
                          f"{direction_symbol}, {stats_result['percent_diff']:+6.1f}%{significance}")
    
    # Create summary of top discriminative features
    all_features = {}
    all_features.update(results['time_analysis'])
    all_features.update(results['freq_analysis'])
    
    if all_features:
        # Sort by effect size
        sorted_features = sorted(all_features.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        print(f"\n  🏆 {subject_name.upper()} - Top 5 Discriminative Features:")
        for i, (feature, stats) in enumerate(sorted_features[:5]):
            direction_symbol = "T>L" if stats['direction'] == 'truth_higher' else "L>T"
            print(f"    {i+1}. {feature:15s}: Effect={stats['effect_size']:.3f}, "
                  f"{direction_symbol}, {stats['percent_diff']:+6.1f}%")
        
        results['summary'] = dict(sorted_features[:10])  # Top 10 features
    else:
        print(f"  ⚠️  No features could be analyzed for {subject_name}")
    
    return results

def analyze_multiple_subjects(root_directory):
    """
    Automatically discover and analyze all subjects in directory and subdirectories
    
    Parameters:
    root_directory: str - Root directory to search for CSV files
    
    Returns:
    dict: Combined analysis results for all subjects
    """
    
    print("=== AUTOMATED MULTI-SUBJECT HRV LIE DETECTION ANALYSIS ===")
    
    # Discover all CSV files
    subject_files = discover_csv_files(root_directory)
    
    if not subject_files:
        print("❌ No valid HRV data files found!")
        print("Expected filename patterns:")
        print("  - *truthsequence_windowed.csv (time domain truth data)")
        print("  - *liesequence_windowed.csv (time domain lie data)")
        print("  - *truthsequence_frequency_windowed.csv (frequency domain truth data)")
        print("  - *liesequence_frequency_windowed.csv (frequency domain lie data)")
        return {}
    
    all_results = {}
    
    print(f"\n🔬 Analyzing {len(subject_files)} subjects...")
    
    for subject, file_paths in subject_files.items():
        try:
            results = analyze_hrv_features(subject, file_paths)
            if results:
                all_results[subject] = results
            else:
                print(f"❌ Failed to analyze {subject}")
        except Exception as e:
            print(f"❌ Error analyzing {subject}: {e}")
    
    # Summary statistics
    total_samples = 0
    total_truth = 0
    total_lie = 0
    
    for subject, results in all_results.items():
        if 'time_analysis' in results:
            # Count samples from first available feature
            first_feature = next(iter(results['time_analysis'].values()), None)
            if first_feature:
                total_truth += first_feature.get('truth_samples', 0)
                total_lie += first_feature.get('lie_samples', 0)
    
    total_samples = total_truth + total_lie
    
    print(f"\n📈 DATASET SUMMARY:")
    print(f"  📊 Total subjects analyzed: {len(all_results)}")
    print(f"  📊 Total samples: {total_samples}")
    print(f"  📊 Truth samples: {total_truth}")
    print(f"  📊 Lie samples: {total_lie}")
    
    return all_results

def analyze_feature_consistency(all_results):
    """
    Analyze consistency of features across all subjects
    
    Parameters:
    all_results: dict - Results from analyze_multiple_subjects
    
    Returns:
    dict: Consistency analysis results
    """
    
    print("\n" + "="*60)
    print("CROSS-SUBJECT CONSISTENCY ANALYSIS")
    print("="*60)
    
    # Collect all features across subjects
    all_features = set()
    for subject_results in all_results.values():
        all_features.update(subject_results['time_analysis'].keys())
        all_features.update(subject_results['freq_analysis'].keys())
    
    consistency_results = {}
    
    for feature in all_features:
        feature_data = {
            'subjects_with_data': [],
            'effect_sizes': [],
            'directions': [],
            'percent_diffs': [],
            'truth_higher_count': 0,
            'lie_higher_count': 0
        }
        
        # Collect data for this feature across all subjects
        for subject, results in all_results.items():
            # Check both time and freq analysis
            feature_stats = None
            if feature in results['time_analysis']:
                feature_stats = results['time_analysis'][feature]
            elif feature in results['freq_analysis']:
                feature_stats = results['freq_analysis'][feature]
            
            if feature_stats:
                feature_data['subjects_with_data'].append(subject)
                feature_data['effect_sizes'].append(feature_stats['effect_size'])
                feature_data['directions'].append(feature_stats['direction'])
                feature_data['percent_diffs'].append(feature_stats['percent_diff'])
                
                if feature_stats['direction'] == 'truth_higher':
                    feature_data['truth_higher_count'] += 1
                else:
                    feature_data['lie_higher_count'] += 1
        
        # Calculate consistency metrics
        if len(feature_data['subjects_with_data']) > 0:
            total_subjects = len(feature_data['subjects_with_data'])
            max_direction_count = max(feature_data['truth_higher_count'], 
                                    feature_data['lie_higher_count'])
            consistency = max_direction_count / total_subjects
            
            feature_data.update({
                'total_subjects': total_subjects,
                'avg_effect_size': np.mean(feature_data['effect_sizes']),
                'consistency_score': consistency,
                'dominant_direction': 'truth_higher' if feature_data['truth_higher_count'] > feature_data['lie_higher_count'] else 'lie_higher',
                'reliability': 'HIGH' if consistency >= 0.8 else 'MODERATE' if consistency >= 0.6 else 'LOW'
            })
            
            consistency_results[feature] = feature_data
    
    # Sort by consistency and effect size
    sorted_features = sorted(consistency_results.items(), 
                           key=lambda x: (x[1]['consistency_score'], x[1]['avg_effect_size']), 
                           reverse=True)
    
    print(f"\nFEATURE RELIABILITY RANKING:")
    print(f"{'Feature':<15} {'Consistency':<12} {'Avg Effect':<12} {'Direction':<15} {'Reliability'}")
    print("-" * 75)
    
    for feature, data in sorted_features:
        consistency_pct = f"{data['consistency_score']*100:.0f}%"
        direction = "Truth Higher" if data['dominant_direction'] == 'truth_higher' else "Lie Higher"
        
        print(f"{feature:<15} {consistency_pct:<12} {data['avg_effect_size']:<12.3f} "
              f"{direction:<15} {data['reliability']}")
    
    # Show detailed breakdown for top features
    print(f"\nDETAILED BREAKDOWN - TOP RELIABLE FEATURES:")
    for feature, data in sorted_features[:5]:
        if data['consistency_score'] >= 0.6:  # Only show moderately reliable or better
            print(f"\n{feature.upper()}:")
            print(f"  Consistency: {data['consistency_score']*100:.0f}% "
                  f"({max(data['truth_higher_count'], data['lie_higher_count'])}/{data['total_subjects']} subjects)")
            print(f"  Average Effect Size: {data['avg_effect_size']:.3f}")
            print(f"  Pattern: {direction}")
            print(f"  Subject breakdown:")
            
            for i, subject in enumerate(data['subjects_with_data']):
                direction_symbol = "T>L" if data['directions'][i] == 'truth_higher' else "L>T"
                print(f"    {subject}: Effect={data['effect_sizes'][i]:.3f}, "
                      f"{direction_symbol}, {data['percent_diffs'][i]:+6.1f}%")
    
    return consistency_results

def create_feature_recommendations(consistency_results):
    """
    Create practical recommendations for feature selection
    """
    
    print(f"\n" + "="*60)
    print("FEATURE SELECTION RECOMMENDATIONS")
    print("="*60)
    
    # Categorize features by reliability
    tier1_features = []  # High reliability (>= 80% consistency)
    tier2_features = []  # Moderate reliability (60-79% consistency)  
    tier3_features = []  # Low reliability (< 60% consistency)
    
    for feature, data in consistency_results.items():
        if data['consistency_score'] >= 0.8:
            tier1_features.append((feature, data))
        elif data['consistency_score'] >= 0.6:
            tier2_features.append((feature, data))
        else:
            tier3_features.append((feature, data))
    
    # Sort each tier by effect size
    tier1_features.sort(key=lambda x: x[1]['avg_effect_size'], reverse=True)
    tier2_features.sort(key=lambda x: x[1]['avg_effect_size'], reverse=True)
    tier3_features.sort(key=lambda x: x[1]['avg_effect_size'], reverse=True)
    
    print("🏆 TIER 1 - HIGHLY RELIABLE FEATURES (Use with high confidence):")
    for feature, data in tier1_features:
        direction = "Truth Higher" if data['dominant_direction'] == 'truth_higher' else "Lie Higher"
        print(f"✅ {feature.upper()}: {data['consistency_score']*100:.0f}% consistency, "
              f"Effect={data['avg_effect_size']:.3f}, {direction}")
    
    print(f"\n🥈 TIER 2 - MODERATELY RELIABLE FEATURES (Use with caution):")
    for feature, data in tier2_features:
        direction = "Truth Higher" if data['dominant_direction'] == 'truth_higher' else "Lie Higher"
        print(f"⚠️  {feature.upper()}: {data['consistency_score']*100:.0f}% consistency, "
              f"Effect={data['avg_effect_size']:.3f}, {direction}")
    
    print(f"\n❌ TIER 3 - UNRELIABLE FEATURES (Avoid for general model):")
    for feature, data in tier3_features[:5]:  # Show top 5 unreliable
        print(f"❌ {feature.upper()}: {data['consistency_score']*100:.0f}% consistency - "
              f"High individual differences")
    
    # Create optimal feature set recommendation
    optimal_features = [f[0] for f in tier1_features] + [f[0] for f in tier2_features[:2]]
    
    print(f"\n🎯 RECOMMENDED FEATURE SET FOR GENERAL MODEL:")
    print(f"PRIMARY: {', '.join([f.upper() for f in optimal_features[:3]])}")
    if len(optimal_features) > 3:
        print(f"SECONDARY: {', '.join([f.upper() for f in optimal_features[3:]])}")
    print(f"TOTAL: {len(optimal_features)} features")
    
    return {
        'tier1': tier1_features,
        'tier2': tier2_features, 
        'tier3': tier3_features,
        'optimal_features': optimal_features
    }

# Main execution function
def main(root_directory=".", save_results=True):
    """
    Main function to run complete automated HRV lie detection analysis
    
    Parameters:
    root_directory: str - Root directory to search for CSV files (default: current directory)
    save_results: bool - Whether to save results to JSON file (default: True)
    """
    
    print("🚀 STARTING AUTOMATED HRV LIE DETECTION ANALYSIS")
    print("=" * 60)
    
    # Step 1: Automatically discover and analyze all subjects
    all_results = analyze_multiple_subjects(root_directory)
    
    if not all_results:
        print("❌ No valid data found. Please check that your CSV files are in the specified directory.")
        print("\n📁 Expected file naming patterns:")
        print("  • *truthsequence_windowed.csv")
        print("  • *liesequence_windowed.csv") 
        print("  • *truthsequence_frequency_windowed.csv")
        print("  • *liesequence_frequency_windowed.csv")
        print("\n💡 The script will automatically extract subject names from filenames!")
        return None
    
    # Step 2: Analyze consistency across subjects
    consistency_results = analyze_feature_consistency(all_results)
    
    # Step 3: Create recommendations
    recommendations = create_feature_recommendations(consistency_results)
    
    # Step 4: Save results to files
    if save_results:
        try:
            import json
            from datetime import datetime
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_filename = f'hrv_analysis_results_{timestamp}.json'
            with open(results_filename, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                def clean_for_json(data):
                    if isinstance(data, dict):
                        return {k: clean_for_json(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [clean_for_json(item) for item in data]
                    else:
                        return convert_numpy(data)
                
                json.dump(clean_for_json(all_results), f, indent=2)
            
            # Create summary CSV for easy viewing
            summary_filename = f'hrv_feature_summary_{timestamp}.csv'
            create_summary_csv(consistency_results, summary_filename)
            
            print(f"\n💾 Results saved:")
            print(f"  📄 Detailed results: {results_filename}")
            print(f"  📊 Feature summary: {summary_filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    print(f"\n🎉 Analysis complete! Found {len(all_results)} subjects with valid data.")
    
    return {
        'all_results': all_results,
        'consistency_results': consistency_results,
        'recommendations': recommendations
    }

def create_summary_csv(consistency_results, filename):
    """
    Create a CSV summary of feature analysis results
    """
    summary_data = []
    
    for feature, data in consistency_results.items():
        summary_data.append({
            'Feature': feature,
            'Consistency_Percent': f"{data['consistency_score']*100:.0f}%",
            'Avg_Effect_Size': f"{data['avg_effect_size']:.3f}",
            'Dominant_Direction': data['dominant_direction'].replace('_', ' ').title(),
            'Reliability': data['reliability'],
            'Subjects_With_Data': data['total_subjects'],
            'Truth_Higher_Count': data['truth_higher_count'],
            'Lie_Higher_Count': data['lie_higher_count']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Consistency_Percent', 'Avg_Effect_Size'], ascending=[False, False])
    df.to_csv(filename, index=False)

# Example usage and convenience functions
def quick_analysis(directory_path="."):
    """
    Quick analysis function - just run this!
    """
    return main(directory_path)

def analyze_single_directory(directory_path):
    """
    Analyze all CSV files in a specific directory (no subdirectory search)
    """
    return main(directory_path)

def get_file_patterns():
    """
    Show expected file naming patterns
    """
    patterns = {
        "Time domain truth": ["*truthsequence_windowed.csv", "*truth-sequence_windowed.csv"],
        "Time domain lie": ["*liesequence_windowed.csv", "*lie-sequence_windowed.csv"], 
        "Frequency domain truth": ["*truthsequence_frequency_windowed.csv", "*truth-sequence_frequency_windowed.csv"],
        "Frequency domain lie": ["*liesequence_frequency_windowed.csv", "*lie-sequence_frequency_windowed.csv"]
    }
    
    print("📁 Expected file naming patterns:")
    for description, pattern_list in patterns.items():
        print(f"  {description}: {' OR '.join(pattern_list)}")
    print("\n💡 Subject names will be automatically extracted from filenames!")
    print("Examples:")
    print("  pr_bayu_emosi_truthsequence_windowed.csv → Subject: 'bayu'")
    print("  pr_bayu_emosi_truth-sequence_windowed.csv → Subject: 'bayu'")
    print("  john_liesequence_frequency_windowed.csv → Subject: 'john'")
    print("  data_smith_truth-sequence_windowed.csv → Subject: 'smith'")

# Simple execution
if __name__ == "__main__":
    # Show file patterns first
    get_file_patterns()
    
    # Run analysis on current directory and all subdirectories
    print(f"\n🔍 Searching current directory and subdirectories for CSV files...")
    results = quick_analysis(".")
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"Use the generated CSV file for a quick overview of results.")
    else:
        print(f"\n❌ No data found. Please check your file naming and location.")
        get_file_patterns()