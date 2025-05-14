import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from forecast import visualize_differencing, diagnose_residuals, create_forecast_plot
from evaluate import compute_rmse, compute_mape, save_metrics_to_csv
from data_cleaner import clean_population_data
import os
import warnings
import logging
from datetime import datetime

# Configure warnings and logging
warnings.filterwarnings("ignore")

def setup_logging():
    """Configure logging system"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    logging.basicConfig(
        filename=f"logs/population_forecast_{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def ensure_directories_exist():
    """Create all required directories"""
    dirs = [
        "Graphs/Differencing",
        "Graphs/Residuals", 
        "Graphs/Forecasts",
        "datas",
        "metrics",
        "clean_snapshots",
        "logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def optimize_arima(series, country):
    """Enhanced ARIMA grid search with error handling"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    p_range = range(0, 3)
    d_range = range(1, 2)
    q_range = range(0, 3)
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for trend in ['c', None]:
                    try:
                        model = ARIMA(series, order=(p,d,q), trend=trend).fit()
                        if (not np.isinf(model.aic) and 
                            model.mle_retvals['converged'] and 
                            not np.isnan(model.params).any()):
                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_order = (p,d,q)
                                best_model = model
                                best_trend = trend
                    except Exception as e:
                        logging.warning(f"ARIMA({p},{d},{q}) failed for {country}: {str(e)}")
                        continue
    
    if best_model:
        trend_status = 'c' if best_trend else 'none'
        logging.info(f"Best model for {country}: ARIMA{best_order} | Trend: {trend_status} | AIC: {best_aic:.2f}")
        return best_model, best_order
    
    logging.warning(f"Optimization failed for {country}, trying fallback")
    try:
        model = ARIMA(series, order=(1,1,1), trend='c').fit()
        logging.info("Using fallback ARIMA(1,1,1) with trend")
        return model, (1,1,1)
    except Exception as e:
        logging.error(f"Fallback model failed for {country}: {str(e)}")
        return None, None

def main():
    setup_logging()
    ensure_directories_exist()
    
    try:
        logging.info("Starting population forecast pipeline")
        print("🚀 Starting population forecasting...")
        
        # Load data
        df = pd.read_csv("Asian_Countries_Population.csv")
        countries = df['Country Name'].unique()
        
        for country in countries:
            try:
                logging.info(f"Processing {country}")
                print(f"\n🔍 Processing {country}...")
                
                # Load and prepare data
                country_data = df[df['Country Name'] == country]
                years = [str(y) for y in range(1960, 2024)]
                raw_series = country_data[years].T.dropna()
                raw_series.index = raw_series.index.astype(int)
                raw_series.columns = ['Population']
                series = raw_series['Population']
                
                # Data cleaning with comparison
                cleaned_series, clean_report = clean_population_data(series, country)
                if clean_report:
                    print(f"🧹 Data cleaning completed:")
                    print(f"   - Original points: {clean_report['original_points']}")
                    print(f"   - Cleaned points: {clean_report['cleaned_points']}")
                    print(f"   - Removed points: {clean_report['removed_points']}")
                    print(f"📊 Comparison saved to: {clean_report['comparison_path']}")
                
                if len(cleaned_series) < 15:
                    msg = f"Skipping {country}: Insufficient data after cleaning ({len(cleaned_series)})"
                    print(f"⚠️ {msg}")
                    logging.warning(msg)
                    continue
                
                # Stationarity handling
                stationary_series = cleaned_series.diff().dropna()
                adf_p = adfuller(stationary_series)[1]
                
                if adf_p > 0.05 and len(stationary_series) > 10:
                    logging.info(f"Testing second differencing for {country} (p={adf_p:.3f})")
                    second_diff = stationary_series.diff().dropna()
                    if len(second_diff) > 5 and adfuller(second_diff)[1] < 0.05:
                        stationary_series = second_diff
                        logging.info("Applied second differencing")
                
                # Model fitting
                model, order = optimize_arima(stationary_series, country)
                if not model:
                    logging.error(f"No valid ARIMA model for {country}")
                    continue
                
                # Diagnostics
                diagnose_residuals(model.resid, country)
                
                # Metrics calculation
                try:
                    train_pred = model.predict(start=stationary_series.index[0], 
                                             end=stationary_series.index[-1])
                    
                    if order[1] == 1:
                        train_pred_orig = np.cumsum(train_pred) + cleaned_series.iloc[1]
                        actual_values = cleaned_series.iloc[2:]
                        pred_values = train_pred_orig[:-1]
                    elif order[1] == 2:
                        train_pred_orig = np.cumsum(np.cumsum(train_pred)) + cleaned_series.iloc[2] + cleaned_series.diff().iloc[2]
                        actual_values = cleaned_series.iloc[3:]
                        pred_values = train_pred_orig[:-2]
                    
                    min_length = min(len(actual_values), len(pred_values))
                    metrics = {
                        'RMSE': compute_rmse(actual_values[-min_length:], pred_values[-min_length:]),
                        'MAPE': compute_mape(actual_values[-min_length:], pred_values[-min_length:]),
                        'Model': f"ARIMA{order}",
                        'Differencing': order[1],
                        'DataPoints': len(cleaned_series),
                        'RemovedPoints': clean_report['removed_points'] if clean_report else 0
                    }
                    
                    if save_metrics_to_csv(country, metrics):
                        print(f"📈 Model metrics saved for {country}")
                    else:
                        logging.warning(f"Failed to save metrics for {country}")
                except Exception as e:
                    logging.error(f"Metrics calculation failed for {country}: {str(e)}")
                
                # Forecasting
                forecast_steps = 12
                forecast = model.get_forecast(steps=forecast_steps)
                forecast_values = forecast.predicted_mean
                
                last_value = cleaned_series.iloc[-1]
                if order[1] == 1:
                    forecast_values = np.cumsum(forecast_values) + last_value
                elif order[1] == 2:
                    forecast_values = np.cumsum(np.cumsum(forecast_values)) + last_value + cleaned_series.diff().iloc[-1]
                
                forecast_df = pd.DataFrame({
                    'Year': range(2024, 2024 + forecast_steps),
                    'Population': forecast_values
                })
                
                # Visualization
                create_forecast_plot(
                    historical=cleaned_series,
                    smoothed=cleaned_series.rolling(5).mean().dropna(),
                    forecast=forecast_df.set_index('Year')['Population'],
                    country=country
                )
                
                forecast_df.to_csv(f"datas/{country}_forecast.csv", index=False)
                print(f"✅ Completed processing for {country}")
                logging.info(f"Completed processing for {country}")
                
            except Exception as e:
                logging.error(f"Error processing {country}: {str(e)}", exc_info=True)
                print(f"⚠️ Error processing {country}: {str(e)}")
                continue
                
        logging.info("Pipeline completed successfully")
        print("\n🎉 All countries processed successfully!")
        print(f"📂 Check these directories for results:")
        print(f"   - Forecasts: datas/")
        print(f"   - Cleaning comparisons: clean_snapshots/")
        print(f"   - Metrics: metrics/")
        print(f"   - Graphs: Graphs/")
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n💥 Critical error: {str(e)}")

if __name__ == "__main__":
    main()