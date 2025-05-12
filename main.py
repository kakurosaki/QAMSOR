import os
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_population_data, get_all_countries
from stationarity_test import check_stationarity, apply_differencing
from train_arima import grid_search_arima, time_series_cv
from forecast import forecast_population
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure directories exist
os.makedirs("Graphs", exist_ok=True)
os.makedirs("datas", exist_ok=True)

# Auto-detect all country names
countries = get_all_countries("Asian_Countries_Population.csv")
print("\n🔎 Detected Countries:", countries)

# Dictionary to store all forecasts
all_forecasts = {}

# Loop through each country and forecast
for country in countries:
    try:
        print(f"\n🔹 Processing: {country}")

        # Step 1: Load data
        df_country = load_population_data("Asian_Countries_Population.csv", country)

        if df_country.empty or len(df_country) < 5:  # Ensure minimum data length
            print(f"⚠️ Not enough data for {country}, skipping...")
            continue

        print(f"📊 {country} - Data Loaded: {df_country.shape}")

        # Step 2: Check Stationarity and Apply Differencing if Needed
        is_stationary, p_value = check_stationarity(df_country["Population"])
        if not is_stationary:
            print(f"📉 {country}'s data is non-stationary (p-value: {p_value:.5f}). Differencing needed.")
            df_country["Population"] = apply_differencing(df_country["Population"], order=1)  # Apply first differencing

        # Step 3: Train ARIMA Model using Grid Search
        model, best_order = grid_search_arima(df_country["Population"])

        if model is None:
            print(f"⚠️ No valid model trained for {country}, skipping...")
            continue

        print(f"✅ ARIMA Model for {country} - Best Order: {best_order}")

        # Step 4: Check Residuals
        residual_diagnostics(model)

        # Step 5: Evaluate Model with Time Series Cross-Validation
        cv_rmse = time_series_cv(df_country["Population"], ARIMA, n_splits=5)
        print(f"Time Series Cross-Validation RMSEs: {cv_rmse}")
        print(f"Average RMSE: {np.mean(cv_rmse)}")

        # Step 6: Forecast Future Population (2024–2035)
        forecast_years = list(range(2024, 2036))
        forecasted_values = forecast_population(model, df_country, forecast_years, country)

        # Step 7: Save Forecast Results in 'datas' folder
        forecast_df = pd.DataFrame({"Year": forecast_years, "Forecasted Population": forecasted_values})
        forecast_df.to_csv(f"datas/Forecast_{country}.csv", index=False)
        print(f"📂 Saved forecast CSV for {country} in 'datas/' folder.")

        # Store forecast in a dictionary
        all_forecasts[country] = forecasted_values

    except Exception as e:
        print(f"❌ Error processing {country}: {e}")

# Step 8: Save All Forecasts in One File
if all_forecasts:
    all_forecasts_df = pd.DataFrame(all_forecasts, index=forecast_years)
    all_forecasts_df.index.name = "Year"
    all_forecasts_df.to_csv("datas/All_Countries_Forecast.csv")
    print("\n✅ All forecasts saved in 'datas/All_Countries_Forecast.csv'!")
else:
    print("\n⚠️ No forecasts were generated. Check the data.")
