import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from forecast import visualize_differencing, diagnose_residuals, create_forecast_plot
import os
import warnings
warnings.filterwarnings("ignore")
import logging

# Set up logging
logging.basicConfig(filename='population_forecast.log', level=logging.INFO)

def ensure_directories_exist():
    """Create all required directories"""
    dirs = [
        "Graphs/Differencing",
        "Graphs/Residuals", 
        "Graphs/Forecasts",
        "datas"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def optimize_arima(series, country):
    """Enhanced grid search with robust error handling"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    # Parameter ranges optimized for population data
    p_range = range(0, 3)  # AR order
    d_range = range(1, 2)  # Differencing (1 is usually best for population)
    q_range = range(0, 3)  # MA order
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for trend in ['c', None]:  # Try with and without constant trend
                    try:
                        model = ARIMA(series, order=(p,d,q), trend=trend).fit()
                        
                        # Validate model converged properly
                        if (not np.isinf(model.aic) and 
                            model.mle_retvals['converged'] and
                            not np.isnan(model.params).any()):
                            
                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_order = (p,d,q)
                                best_model = model
                                best_trend = trend
                                
                    except Exception as e:
                        logging.warning(f"Failed ARIMA({p},{d},{q}) for {country}: {str(e)}")
                        continue
    
    if best_model:
        trend_status = 'c' if best_trend else 'none'
        print(f"✅ Best model: ARIMA{best_order} | Trend: {trend_status} | AIC: {best_aic:.2f}")
        return best_model, best_order
    
    # Fallback to simple model if optimization fails
    print("⚠️ Optimization failed - trying simple fallback model")
    try:
        model = ARIMA(series, order=(1,1,1), trend='c').fit()
        print("✅ Using fallback ARIMA(1,1,1) with trend")
        return model, (1,1,1)
    except Exception as e:
        logging.error(f"Fallback model failed for {country}: {str(e)}")
        return None, None

def main():
    ensure_directories_exist()
    
    try:
        df = pd.read_csv("Asian_Countries_Population.csv")
        countries = df['Country Name'].unique()
        
        for country in countries:
            try:
                print(f"\n🔍 Processing {country}...")
                logging.info(f"Processing {country}")
                
                # Load and prepare data
                country_data = df[df['Country Name'] == country]
                years = [str(y) for y in range(1960, 2024)]
                series = country_data[years].T.dropna()
                series.index = series.index.astype(int)
                series.columns = ['Population']
                
                # Check data sufficiency
                if len(series) < 15:
                    print(f"⚠️ Insufficient data points ({len(series)})")
                    continue
                
                # Differencing with careful stationarity testing
                stationary_series = series['Population'].diff().dropna()
                adf_p = adfuller(stationary_series)[1]
                
                if adf_p > 0.05 and len(stationary_series) > 10:
                    print(f"Testing second differencing (p-value={adf_p:.3f})")
                    second_diff = stationary_series.diff().dropna()
                    if len(second_diff) > 5 and adfuller(second_diff)[1] < 0.05:
                        stationary_series = second_diff
                        print("Applied second differencing")
                
                # Model fitting with enhanced optimization
                model, order = optimize_arima(stationary_series, country)
                if not model:
                    print("❌ Could not fit any ARIMA model")
                    continue
                
                # Generate and save diagnostics
                diagnose_residuals(model.resid, country)
                
                # Forecasting
                forecast_steps = 12
                forecast = model.get_forecast(steps=forecast_steps)
                forecast_values = forecast.predicted_mean
                
                # Reconstruct trend based on differencing order
                last_value = series['Population'].iloc[-1]
                if order[1] == 1:
                    forecast_values = np.cumsum(forecast_values) + last_value
                elif order[1] == 2:
                    forecast_values = np.cumsum(np.cumsum(forecast_values)) + last_value + series['Population'].diff().iloc[-1]
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'Year': range(2024, 2024 + forecast_steps),
                    'Population': forecast_values
                })
                
                # Visualization
                create_forecast_plot(
                    historical=series['Population'],
                    smoothed=series['Population'].rolling(5).mean().dropna(),
                    forecast=forecast_df.set_index('Year')['Population'],
                    country=country
                )
                
                # Save results
                forecast_df.to_csv(f"datas/{country}_forecast.csv", index=False)
                print(f"📊 Saved forecast for {country}")
                
            except Exception as e:
                print(f"⚠️ Error processing {country}: {str(e)}")
                logging.error(f"Error processing {country}: {str(e)}")
                continue
                
        print("\n🎉 Processing completed!")
        
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        logging.critical(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()