import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import os

def ensure_directory(path):
    """Guarantee directory exists and return path"""
    os.makedirs(path, exist_ok=True)
    return path

def visualize_differencing(series, country):
    """Complete differencing visualization with guaranteed saving"""
    try:
        # Setup directory and figure
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
        
        # Guaranteed save
        save_path = os.path.join(diff_dir, f"{country}_differencing.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved differencing plot: {save_path}")
        
        return optimal_diff
        
    except Exception as e:
        print(f"❌ Differencing plot error: {str(e)}")
        plt.close()
        return series.diff().dropna()  # Fallback

def diagnose_residuals(residuals, country):
    """Complete residual diagnostics with guaranteed saving"""
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
        print(f"✅ Saved residuals plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Residuals plot error: {str(e)}")
        plt.close()

def create_forecast_plot(historical, smoothed, forecast, country):
    """Complete forecast visualization with trend preservation"""
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
        print(f"✅ Saved forecast plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Forecast plot error: {str(e)}")
        plt.close()