import os
import re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

class FeatureStatisticalAnalyzer:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        
    def load_data(self):
        """Load and combine all CSV files."""
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        file_metadata = []
        
        for filename in csv_files:
            pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence_(frequency_)?windowed\.csv'
            match = re.match(pattern, filename)
            
            if match:
                file_metadata.append({
                    'subject': match.group(1),
                    'condition': match.group(2),
                    'label': match.group(3),
                    'domain': 'frequency' if match.group(4) else 'time',
                    'filename': filename
                })
        
        all_dataframes = []
        
        for meta in file_metadata:
            filepath = os.path.join(self.data_directory, meta['filename'])
            try:
                df = pd.read_csv(filepath)
                df['subject'] = meta['subject']
                df['condition'] = meta['condition']
                df['label'] = meta['label']
                df['domain'] = meta['domain']
                df['binary_label'] = 1 if meta['label'] == 'lie' else 0
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error loading {meta['filename']}: {e}")
        
        if not all_dataframes:
            raise ValueError("No CSV files could be loaded")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        print(f"Unique subjects: {sorted(combined_df['subject'].unique())}")
        
        return combined_df
    
    def get_numeric_features(self, df):
        """Extract numeric feature columns (excluding metadata columns and window features)."""
        metadata_cols = {'subject', 'condition', 'label', 'domain', 'binary_label'}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out metadata columns and window-related features
        feature_cols = [col for col in numeric_cols 
                       if col not in metadata_cols and not col.startswith('window')]
        
        print(f"Total numeric columns: {len(numeric_cols)}")
        print(f"Features after filtering (excluding metadata and window* features): {len(feature_cols)}")
        
        return feature_cols
    
    def perform_ttest_analysis(self, df, groupby_column='binary_label'):
        """Perform independent t-tests for all numeric features grouped by specified column."""
        
        feature_cols = self.get_numeric_features(df)
        
        if not feature_cols:
            print("No numeric features found for analysis")
            return None
        
        # Get unique groups
        groups = df[groupby_column].unique()
        if len(groups) != 2:
            print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
            print("Using first two groups for comparison")
            groups = groups[:2]
        
        print(f"\nPerforming t-tests comparing groups: {groups}")
        print(f"Group 0 ({groups[0]}): {len(df[df[groupby_column] == groups[0]])} samples")
        print(f"Group 1 ({groups[1]}): {len(df[df[groupby_column] == groups[1]])} samples")
        
        # Separate data by groups
        group_0_data = df[df[groupby_column] == groups[0]]
        group_1_data = df[df[groupby_column] == groups[1]]
        
        results = []
        
        for feature in feature_cols:
            # Get feature values for each group
            group_0_values = group_0_data[feature].dropna()
            group_1_values = group_1_data[feature].dropna()
            
            # Skip if insufficient data
            if len(group_0_values) < 2 or len(group_1_values) < 2:
                continue
            
            # Perform independent t-test
            t_stat, p_value = ttest_ind(group_0_values, group_1_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group_0_values) - 1) * group_0_values.std()**2 + 
                                 (len(group_1_values) - 1) * group_1_values.std()**2) / 
                                (len(group_0_values) + len(group_1_values) - 2))
            cohens_d = (group_0_values.mean() - group_1_values.mean()) / pooled_std
            
            # Statistical significance
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            results.append({
                'feature': feature,
                'group_0_mean': group_0_values.mean(),
                'group_0_std': group_0_values.std(),
                'group_0_n': len(group_0_values),
                'group_1_mean': group_1_values.mean(),
                'group_1_std': group_1_values.std(),
                'group_1_n': len(group_1_values),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance
            })
        
        return pd.DataFrame(results)
    
    def display_results(self, results_df, sort_by='p_value', top_n=20):
        """Display statistical results in a formatted table."""
        if results_df is None or results_df.empty:
            print("No results to display")
            return
        
        # Sort results
        results_sorted = results_df.sort_values(by=sort_by).head(top_n)
        
        print(f"\n{'='*100}")
        print(f"STATISTICAL T-TEST RESULTS (Top {top_n} features sorted by {sort_by})")
        print(f"Window features excluded from analysis")
        print(f"{'='*100}")
        print(f"{'Feature':<25} {'Group0_Mean':<12} {'Group1_Mean':<12} {'T-Stat':<10} {'P-Value':<12} {'Cohen_d':<10} {'Sig':<5}")
        print("-" * 100)
        
        for _, row in results_sorted.iterrows():
            print(f"{row['feature']:<25} {row['group_0_mean']:<12.4f} {row['group_1_mean']:<12.4f} "
                  f"{row['t_statistic']:<10.4f} {row['p_value']:<12.6f} {row['cohens_d']:<10.4f} {row['significance']:<5}")
        
        print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Summary statistics
        significant_features = results_df[results_df['p_value'] < 0.05]
        print(f"\nSUMMARY:")
        print(f"Total features analyzed: {len(results_df)}")
        print(f"Significant features (p<0.05): {len(significant_features)}")
        print(f"Highly significant features (p<0.001): {len(results_df[results_df['p_value'] < 0.001])}")
        
        return results_sorted
    
    def analyze_by_domain(self, df):
        """Perform separate analysis for time and frequency domains."""
        if 'domain' not in df.columns:
            print("No domain column found, performing overall analysis")
            return self.perform_ttest_analysis(df)
        
        domains = df['domain'].unique()
        all_results = {}
        
        for domain in domains:
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR {domain.upper()} DOMAIN")
            print(f"{'='*60}")
            
            domain_df = df[df['domain'] == domain]
            results = self.perform_ttest_analysis(domain_df)
            
            if results is not None:
                all_results[domain] = results
                self.display_results(results, top_n=15)
        
        return all_results

def main():
    # Set your data directory path here
    data_directory = "./"  # Update this path
    
    # Initialize analyzer
    analyzer = FeatureStatisticalAnalyzer(data_directory)
    
    try:
        # Load data
        print("Loading data...")
        df = analyzer.load_data()
        
        # Perform overall analysis
        print("\n" + "="*60)
        print("OVERALL STATISTICAL ANALYSIS")
        print("="*60)
        
        results = analyzer.perform_ttest_analysis(df)
        overall_results = analyzer.display_results(results)
        
        # Perform domain-specific analysis
        domain_results = analyzer.analyze_by_domain(df)
        
        # Save results to CSV
        if results is not None:
            output_file = "ttest_results_no_window.csv"
            results.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        
        return results, domain_results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None, None

if __name__ == "__main__":
    # Run the analysis
    overall_results, domain_results = main()
    
    # Optional: Access specific results
    if overall_results is not None:
        print("\nMost significant features (excluding window features):")
        top_features = overall_results.head(5)
        for _, row in top_features.iterrows():
            print(f"- {row['feature']}: p={row['p_value']:.6f}, Cohen's d={row['cohens_d']:.4f}")