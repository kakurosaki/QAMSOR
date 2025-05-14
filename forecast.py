import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import os

def ensure_directory(path):
    """Create directory with verification"""
    try:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o755)
        return path
    except Exception as e:
        print(f"❌ Failed to create directory {path}: {str(e)}")
        raise

def verify_plot_save(save_path):
    """Verify plot was saved successfully"""
    try:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Plot not created at {save_path}")
        if os.path.getsize(save_path) == 0:
            raise ValueError("Empty plot file")
        return True
    except Exception as e:
        print(f"❌ Plot verification failed: {str(e)}")
        raise

def visualize_differencing(series, country):
    """Differencing visualization with verification"""
    try:
        diff_dir = ensure_directory("Graphs/Differencing")
        plt.figure(figsize=(15,12))
        
        # Original Series
        plt.subplot(4,1,1)
        plt.plot(series.index, series, label='Original')
        adf_orig = adfuller(series)[1]
        plt.title(f'{country} - Original (ADF p={adf_orig:.4f})')
        plt.grid(True)
        
        # First Difference
        diff1 = series.diff().dropna()
        plt.subplot(4,1,2)
        plt.plot(diff1.index, diff1, color='orange')
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        adf_diff1 = adfuller(diff1)[1]
        plt.title(f'1st Difference (ADF p={adf_diff1:.4f})')
        plt.grid(True)
        
        # Second Difference
        diff2 = diff1.diff().dropna()
        plt.subplot(4,1,3)
        plt.plot(diff2.index, diff2, color='green')
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        adf_diff2 = adfuller(diff2)[1]
        plt.title(f'2nd Difference (ADF p={adf_diff2:.4f})')
        plt.grid(True)
        
        # Selected Difference
        optimal_diff = diff1 if adf_diff1 < 0.05 else diff2
        plt.subplot(4,1,4)
        plt.plot(optimal_diff.index, optimal_diff, color='purple')
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Selected (ADF p={adfuller(optimal_diff)[1]:.4f})')
        plt.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(diff_dir, f"{country}_differencing.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        verify_plot_save(save_path)
        print(f"✅ Saved differencing plot: {save_path}")
        return optimal_diff
        
    except Exception as e:
        print(f"❌ Differencing plot error: {str(e)}")
        plt.close('all')
        raise

def diagnose_residuals(residuals, country):
    """Residual diagnostics with verification"""
    try:
        res_dir = ensure_directory("Graphs/Residuals")
        plt.figure(figsize=(15,5))
        
        # Residuals Plot
        plt.subplot(1,3,1)
        plt.plot(residuals)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Residuals')
        
        # ACF Plot
        plt.subplot(1,3,2)
        plot_acf(residuals, lags=20, ax=plt.gca())
        plt.title('ACF')
        
        # PACF Plot
        plt.subplot(1,3,3)
        plot_pacf(residuals, lags=20, ax=plt.gca())
        plt.title('PACF')
        
        plt.tight_layout()
        
        save_path = os.path.join(res_dir, f"{country}_residuals.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        verify_plot_save(save_path)
        print(f"✅ Saved residuals plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Residuals plot error: {str(e)}")
        plt.close('all')
        raise

def create_forecast_plot(historical, smoothed, forecast, country):
    """Forecast visualization with verification"""
    try:
        forecast_dir = ensure_directory("Graphs/Forecasts")
        plt.figure(figsize=(12,6))
        
        # Align forecast with history
        full_series = pd.concat([
            historical,
            pd.Series([historical.iloc[-1]], index=[historical.index[-1]]),
            forecast
        ])
        
        # Plot components
        plt.plot(historical.index, historical, 'b-', label='Historical', linewidth=2)
        plt.plot(smoothed.index, smoothed, 'g-', label='Trend (5Y MA)', linewidth=2)
        plt.plot(full_series.index[-len(forecast)-1:], 
                full_series[-len(forecast)-1:], 
                'r--', label='Forecast', linewidth=2)
        
        # Formatting
        plt.axvline(x=historical.index[-1], color='gray', linestyle=':')
        plt.title(f'{country} Population Forecast')
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        save_path = os.path.join(forecast_dir, f"{country}_forecast.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        verify_plot_save(save_path)
        print(f"✅ Saved forecast plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Forecast plot error: {str(e)}")
        plt.close('all')
        raise

def plot_validation(historical, train_years, val_years, val_results, country):
    """Validation visualization with verification"""
    try:
        val_dir = ensure_directory("Graphs/Validation")
        plt.figure(figsize=(14,7))
        
        # Plot full historical data
        plt.plot(historical.index, historical, 'b-', label='Historical Data', linewidth=1.5)
        
        # Highlight validation period
        val_data = historical.loc[val_years[0]:val_years[-1]]
        plt.plot(val_data.index, val_data, 'ro-', markersize=6, label='Actual Values')
        
        # Plot predictions
        plt.plot(val_data.index, val_results['predictions'], 'gx--', 
                 markersize=8, linewidth=1.5,
                 label=f'Predictions (MAPE: {val_results["mape"]:.1f}%)')
        
        # Add annotations
        for year, actual, pred in zip(val_years, val_results['actuals'], val_results['predictions']):
            plt.text(year, actual, f"  {actual/1e6:.1f}M", ha='left', va='center', color='red')
            plt.text(year, pred, f"  {pred/1e6:.1f}M", ha='left', va='center', color='green')
        
        # Add shaded regions
        plt.axvspan(train_years[0], train_years[-1], alpha=0.1, color='blue', label='Training Period')
        plt.axvspan(val_years[0], val_years[-1], alpha=0.1, color='orange', label='Validation Period')
        
        plt.title(f"{country}\nModel Validation ({val_years[0]}-{val_years[-1]})", pad=20)
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(val_dir, f"{country}_validation.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        verify_plot_save(save_path)
        print(f"✅ Saved validation plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Validation plot error: {str(e)}")
        plt.close('all')
        raise