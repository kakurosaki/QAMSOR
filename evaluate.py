import numpy as np
import pandas as pd
import os

def compute_rmse(actual, predicted):
    """Computes Root Mean Squared Error"""
    return np.sqrt(np.mean((actual.values - predicted.values) ** 2))

def compute_mape(actual, predicted):
    """Computes Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual.values - predicted.values) / actual.values)) * 100

def save_metrics_to_csv(country, metrics_dict, filename="metrics.csv"):
    """Saves evaluation metrics with cleaning metadata"""
    try:
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        
        metrics_df = pd.DataFrame({
            'Country': [country],
            'Timestamp': [pd.Timestamp.now()],
            **metrics_dict
        })
        
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(filepath, index=False)
        else:
            metrics_df.to_csv(filepath, index=False)
            
        return True
    except Exception as e:
        print(f"❌ Error saving metrics: {str(e)}")
        return False