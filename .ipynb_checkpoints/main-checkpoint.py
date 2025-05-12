from load_data import load_population_data, get_all_countries
from stationarity_test import check_stationarity
from train_arima import rolling_window_arima
from forecast import forecast_population
import pandas as pd

# Auto-detect all country names
countries = get_all_countries("Asian_Countries_Population.csv")

# Dictionary to store all forecasts
all_forecasts = {}

# Loop through each country and forecast
for country in countries:
    print(f"\n🔹 Processing: {country}")

    # Step 1: Load data
    df_country = load_population_data("Asian_Countries_Population.csv", country)
    
    if df_country.empty:
        print(f"⚠️ No valid data for {country}, skipping...")
        continue
    
    # Step 2: Check Stationarity
    is_stationary, p_value = check_stationarity(df_country["Population"])
    if not is_stationary:
        print(f"📉 {country}'s data is non-stationary (p-value: {p_value:.5f}). Differencing needed.")
    
    # Step 3: Train ARIMA Model with Rolling Window Cross-Validation
    model, best_order, avg_rmse, avg_mape = rolling_window_arima(df_country["Population"], num_splits=5)
    print(f"✅ Final ARIMA Model for {country} - Order: {best_order}, Avg RMSE: {avg_rmse:.2f}, Avg MAPE: {avg_mape:.2f}%")

    # Step 4: Forecast Future Population (2024–2034) & Generate Plot
    forecast_years = list(range(2024, 2035))
    forecasted_values = forecast_population(model, df_country, forecast_years, country)

    # Step 5: Save Forecast Results
    forecast_df = pd.DataFrame({"Year": forecast_years, "Forecasted Population": forecasted_values})
    forecast_df.to_csv(f"Forecast_{country}.csv", index=False)
    print(f"📂 Saved forecast for {country}!")

    # Store forecast in a dictionary
    all_forecasts[country] = forecasted_values

# Step 6: Save All Forecasts in One File
all_forecasts_df = pd.DataFrame(all_forecasts, index=forecast_years)
all_forecasts_df.index.name = "Year"
all_forecasts_df.to_csv("All_Countries_Forecast.csv")

print("\n✅ All forecasts saved in 'All_Countries_Forecast.csv'!")
