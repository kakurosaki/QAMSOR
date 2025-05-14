import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from forecast import visualize_differencing, diagnose_residuals, create_forecast_plot, plot_validation
from evaluate import compute_rmse, compute_mape, save_metrics_to_csv, save_validation_report
from data_cleaner import clean_population_data
import os
import warnings
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

def setup_logging():
    """Configure logging system"""
    try:
        os.makedirs("logs", exist_ok=True)
        os.chmod("logs", 0o755)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        logging.basicConfig(
            filename=f"logs/population_forecast_{timestamp}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    except Exception as e:
        print(f"❌ Failed to setup logging: {str(e)}")
        raise

def ensure_directories_exist():
    """Create all required directories with verification"""
    dirs = [
        "Graphs/Differencing",
        "Graphs/Residuals", 
        "Graphs/Forecasts",
        "Graphs/Validation",
        "datas",
        "metrics",
        "clean_snapshots",
        "logs"
    ]
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o755)
        except Exception as e:
            logging.error(f"Directory creation failed: {dir_path} - {str(e)}")
            raise

def optimize_arima(series, country):
    """ARIMA optimization with proper error handling"""
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

def hybrid_validation(series, country, order):
    """Validation with detailed tracking"""
    try:
        val_size = min(5, max(2, int(len(series)*0.15)))  # 2-5 years
        if len(series) < 10:
            return float('nan'), None, None, None
            
        train = series.iloc[:-val_size]
        val = series.iloc[-val_size:]
        
        actuals = val.values
        predictions = []
        temp_model = ARIMA(train, order=order).fit()
        
        for i in range(1, len(val)+1):
            pred = temp_model.get_forecast(steps=i).predicted_mean.iloc[-1]
            predictions.append(pred)
            if i < len(val):
                temp_model = temp_model.append([val.iloc[i-1]])
        
        # Ensure we return proper lists
        actuals_list = actuals.tolist() if hasattr(actuals, 'tolist') else list(actuals)
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        mape = compute_mape(pd.Series(actuals_list), pd.Series(predictions_list))
        rmse = compute_rmse(pd.Series(actuals_list), pd.Series(predictions_list))
        
        return mape, {
            'actuals': actuals_list,
            'predictions': predictions_list,
            'mape': mape,
            'rmse': rmse
        }, train.index, val.index
        
    except Exception as e:
        logging.warning(f"Validation failed for {country}: {str(e)}")
        return float('nan'), None, None, None
def main():
    setup_logging()
    ensure_directories_exist()
    
    try:
        logging.info("Starting population forecast pipeline")
        df = pd.read_csv("Asian_Countries_Population.csv")
        countries = df['Country Name'].unique()
        
        for country in countries:
            try:
                logging.info(f"Processing {country}")
                print(f"\n🔍 Processing {country}...")
                
                # Data loading and cleaning
                country_data = df[df['Country Name'] == country]
                years = [str(y) for y in range(1960, 2024)]
                raw_series = country_data[years].T.dropna()
                raw_series.index = raw_series.index.astype(int)
                raw_series.columns = ['Population']
                series = raw_series['Population']
                
                cleaned_series, clean_report = clean_population_data(series, country)
                if clean_report:
                    print(f"🧹 Cleaned {country}: Removed {clean_report['removed_points']} points")
                
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
                
                # Hybrid validation
                val_mape, val_results, train_years, val_years = hybrid_validation(cleaned_series, country, order)
                
                if val_results:
                    print(f"📊 Validation MAPE ({val_years[0]}-{val_years[-1]}): {val_mape:.2f}%")
                    save_validation_report(
                        country=country,
                        train_years=train_years,
                        val_years=val_years,
                        val_results=val_results
                    )
                    plot_validation(
                        historical=cleaned_series,
                        train_years=train_years,
                        val_years=val_years,
                        val_results=val_results,
                        country=country
                    )
                
                # Generate and save diagnostics
                diagnose_residuals(model.resid, country)
                visualize_differencing(cleaned_series, country)
                
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
                        'Validation_MAPE': val_mape,
                        'DataPoints': len(cleaned_series),
                        'RemovedPoints': clean_report['removed_points'] if clean_report else 0
                    }
                    
                    if not save_metrics_to_csv(country, metrics):
                        raise Exception("Failed to save metrics")
                except Exception as e:
                    logging.error(f"Metrics calculation failed for {country}: {str(e)}")
                    raise
                
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
                
                # Save forecast data
                forecast_path = f"datas/{country}_forecast.csv"
                forecast_df.to_csv(forecast_path, index=False)
                print(f"💾 Saved forecast data: {forecast_path}")
                
                # Create forecast plot
                create_forecast_plot(
                    historical=cleaned_series,
                    smoothed=cleaned_series.rolling(5).mean().dropna(),
                    forecast=forecast_df.set_index('Year')['Population'],
                    country=country
                )
                
                logging.info(f"✅ Completed processing for {country}")
                print(f"✅ Completed {country}")
                
            except Exception as e:
                logging.error(f"Error processing {country}: {str(e)}", exc_info=True)
                print(f"❌ Error processing {country}: {str(e)}")
                continue
                
        logging.info("Pipeline completed successfully")
        print("\n🎉 Processing completed!")
        print("Output directories:")
        print(f"- Forecast data: datas/")
        print(f"- Metrics: metrics/")
        print(f"- Graphs: Graphs/")
        print(f"- Logs: logs/")
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n💥 Critical error: {str(e)}")

if __name__ == "__main__":
    main()