import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

def save_cleaning_comparison(original, cleaned, country, filename="cleaning_comparison.csv"):
    """Save before/after cleaning data to single CSV"""
    try:
        os.makedirs("clean_snapshots", exist_ok=True)
        filepath = Path("clean_snapshots") / filename
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Year': original.index,
            f'{country}_Original': original.values,
            f'{country}_Cleaned': cleaned.values
        }).set_index('Year')
        
        # Append to existing file or create new
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, index_col='Year')
            updated = existing.join(comparison, how='outer')
            updated.to_csv(filepath)
        else:
            comparison.to_csv(filepath)
            
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to save comparison: {str(e)}")
        return None

def clean_population_data(series, country):
    """Full cleaning pipeline with comparison saving"""
    try:
        original = series.copy()
        
        # 1. Remove negatives/zeros
        cleaned = series[series > 0]
        
        # 2. Handle missing values
        cleaned = cleaned.interpolate(method='time').ffill().bfill()
        
        # 3. Remove outliers (Tukey's method with rolling window)
        rolling_stats = cleaned.rolling(window=5, min_periods=1).agg(['mean', 'std'])
        upper_bound = (rolling_stats['mean'] + 3 * rolling_stats['std']).bfill()
        lower_bound = (rolling_stats['mean'] - 3 * rolling_stats['std']).bfill()
        cleaned = cleaned[(cleaned >= lower_bound) & (cleaned <= upper_bound)]
        
        # Save comparison
        comparison_path = save_cleaning_comparison(original, cleaned, country)
        
        report = {
            'country': country,
            'original_points': len(original),
            'cleaned_points': len(cleaned),
            'removed_points': len(original) - len(cleaned),
            'comparison_path': comparison_path,
            'timestamp': pd.Timestamp.now()
        }
        
        logging.info(f"Data cleaning completed for {country}")
        return cleaned, report
        
    except Exception as e:
        logging.error(f"Cleaning failed for {country}: {str(e)}")
        return series, None