import numpy as np
import pandas as pd
import os
from pathlib import Path

def compute_rmse(actual, predicted):
    """Compute RMSE with validation"""
    try:
        if len(actual) != len(predicted):
            raise ValueError("Inputs must have same length")
        return np.sqrt(np.mean((actual - predicted) ** 2))
    except Exception as e:
        print(f"❌ RMSE calculation error: {str(e)}")
        raise

def compute_mape(actual, predicted):
    """Compute MAPE with validation"""
    try:
        if len(actual) != len(predicted):
            raise ValueError("Inputs must have same length")
        if (actual == 0).any():
            raise ValueError("Actual values contain zeros")
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    except Exception as e:
        print(f"❌ MAPE calculation error: {str(e)}")
        raise

def verify_file_save(filepath):
    """Verify file was saved successfully"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not created at {filepath}")
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file")
        return True
    except Exception as e:
        print(f"❌ File verification failed: {str(e)}")
        raise

def save_metrics_to_csv(country, metrics, filename="metrics.csv"):
    """Save metrics with verification"""
    try:
        os.makedirs("metrics", exist_ok=True)
        os.chmod("metrics", 0o755)
        filepath = os.path.join("metrics", filename)
        
        metrics_df = pd.DataFrame({
            'Country': [country],
            'Timestamp': [pd.Timestamp.now()],
            **metrics
        })
        
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(filepath, index=False)
        else:
            metrics_df.to_csv(filepath, index=False)
        
        verify_file_save(filepath)
        print(f"✅ Metrics saved to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save metrics: {str(e)}")
        raise

def save_validation_report(country, train_years, val_years, val_results, filename="validation_report.csv"):
    """Save validation/test report with verification"""
    try:
        os.makedirs("metrics", exist_ok=True)
        os.chmod("metrics", 0o755)
        filepath = Path("metrics") / filename
        
        # Convert numpy arrays if present
        actuals = val_results['actuals'].tolist() if hasattr(val_results['actuals'], 'tolist') else val_results['actuals']
        predictions = val_results['predictions'].tolist() if hasattr(val_results['predictions'], 'tolist') else val_results['predictions']
        
        report = pd.DataFrame({
            'Country': [country],
            'Training_Start': [train_years[0]],
            'Training_End': [train_years[-1]],
            'Validation_Start': [val_years[0]],
            'Validation_End': [val_years[-1]],
            'Validation_Actual': [actuals],
            'Validation_Predicted': [predictions],
            'Validation_MAPE': [val_results['mape']],
            'Validation_RMSE': [val_results['rmse']],
            'Report_Type': ['Test' if 'test' in filename.lower() else 'Validation']
        })
        
        if filepath.exists():
            existing = pd.read_csv(filepath)
            updated = pd.concat([existing, report], ignore_index=True)
        else:
            updated = report
        
        updated.to_csv(filepath, index=False)
        verify_file_save(filepath)
        print(f"✅ Report saved to {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"❌ Failed to save report: {str(e)}")
        raise