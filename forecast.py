# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# def forecast_population(model, df_country, forecast_years, country):
#     """
#     Forecasts the population for future years using the trained ARIMA model
#     and visualizes both historical and forecasted trends.

#     Args:
#     - model: Trained ARIMA model.
#     - df_country (pd.DataFrame): Historical population data.
#     - forecast_years (list): List of years to forecast.
#     - country (str): Country name.

#     Returns:
#     - List of forecasted population values.
#     """
#     try:
#         # Forecast future population
#         steps = len(forecast_years)
#         forecast = model.forecast(steps=steps)

#         # Ensure directory exists
#         os.makedirs("Graphs", exist_ok=True)

#         # Plot historical and forecasted population
#         plt.figure(figsize=(12, 6))
#         sns.lineplot(x=df_country.index, y=df_country["Population"], label="Historical Population", marker="o")
#         sns.lineplot(x=forecast_years, y=forecast, label="Forecasted Population", marker="o", linestyle="--", color="red")

#         plt.xlabel("Year")
#         plt.ylabel("Population")
#         plt.title(f"Population Forecast for {country}")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()

#         # Save the plot in 'Graphs' folder
#         graph_path = f"Graphs/Forecast_Plot_{country}.png"
#         plt.savefig(graph_path)
#         print(f"📊 Saved forecast plot for {country} in 'Graphs/' folder.")

#         plt.show()

#         return forecast.tolist()

#     except Exception as e:
#         print(f"❌ Error forecasting {country}: {e}")
#         return []


# import matplotlib.pyplot as plt
# import pandas as pd

# def forecast_population(model, df_country, forecast_years, country_name):
#     """
#     Uses the trained ARIMA model to forecast population for future years and plots the results
#     along with historical, smoothed, and forecasted data as a continuous trend.
#     """
#     # Generate forecasts
#     forecast = model.forecast(steps=len(forecast_years))

#     # Calculate smoothed data (moving average as example)
#     smoothed = df_country["Population"].rolling(window=3, center=True).mean()

#     # Get last valid smoothed value
#     last_smoothed = smoothed.dropna().iloc[-1]

#     # Create index for forecast
#     forecast_index = pd.Index(forecast_years)

#     # Concatenate smoothed and forecast for plotting
#     smoothed_forecast = pd.concat([pd.Series([last_smoothed], index=[df_country.index[-1]]), pd.Series(forecast.values, index=forecast_index)])

#     # Plotting
#     plt.figure(figsize=(10, 5))

#     # Plot historical
#     plt.plot(df_country.index, df_country["Population"], label="Historical Data", color="gray", linewidth=2)

#     # Plot smoothed
#     plt.plot(smoothed.index, smoothed, label="Smoothed Data", color="red", linewidth=2)

#     # Plot forecast from smoothed trend
#     plt.plot(smoothed_forecast.index, smoothed_forecast.values, label="Forecasted Data", color="blue", linestyle="--", linewidth=2)

#     # Labels and formatting
#     plt.xlabel("Year")
#     plt.ylabel("Population")
#     plt.title(f"Population Forecast for {country_name} (2024–2034)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # Save plot
#     plot_filename = f"Forecast_Plot_{country_name}.png"
#     plt.savefig(plot_filename)
#     print(f"📊 Saved forecast plot for {country_name} as {plot_filename}")

#     # Show plot
#     plt.show()

#     return forecast




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima

def forecast_population(df_country, forecast_years, country_name):
    """
    Forecast population using STL, log-transform, and auto_arima.
    Returns forecast and generates a combined plot with historical, smoothed, and forecasted data.
    """
    # Log transform to stabilize variance
    df_country["Log_Pop"] = np.log(df_country["Population"])

    # STL decomposition for smoothing
    stl = STL(df_country["Log_Pop"], period=1)
    res = stl.fit()
    smoothed_log = res.trend

    # Use auto_arima for optimal ARIMA parameters
    model = auto_arima(df_country["Log_Pop"], seasonal=False, stepwise=True, suppress_warnings=True)

    # Forecast future values in log scale
    forecast_log = model.predict(n_periods=len(forecast_years))

    # Convert forecast back from log scale
    forecast = np.exp(forecast_log)

    # Last smoothed value (exponentiated)
    last_smoothed = np.exp(smoothed_log.dropna().iloc[-1])

    # Indexes
    forecast_index = pd.Index(forecast_years)
    smoothed_forecast = pd.concat([
        pd.Series([last_smoothed], index=[df_country.index[-1]]),
        pd.Series(forecast, index=forecast_index)
    ])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_country.index, df_country["Population"], label="Historical Data", color="gray", linewidth=2)
    plt.plot(smoothed_log.index, np.exp(smoothed_log), label="Smoothed Data (STL)", color="red", linewidth=2)
    plt.plot(smoothed_forecast.index, smoothed_forecast.values, label="Forecasted Data", color="blue", linestyle="--", linewidth=2)

    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title(f"Population Forecast for {country_name} (2024–2034)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f"Forecast_Plot_{country_name}.png"
    plt.savefig(plot_filename)
    print(f"Saved forecast plot for {country_name} as {plot_filename}")
    plt.show()

    return pd.Series(forecast, index=forecast_index)
