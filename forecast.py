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




# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.seasonal import STL
# from pmdarima import auto_arima

# def forecast_population(df_country, forecast_years, country_name):
#     """
#     Forecast population using STL, log-transform, and auto_arima.
#     Returns forecast and generates a combined plot with historical, smoothed, and forecasted data.
#     """
#     # Log transform to stabilize variance
#     df_country["Log_Pop"] = np.log(df_country["Population"])
#     df_country["Diff_Pop"] = df_country["Log_Pop"].diff().dropna()


#     # STL decomposition for smoothing
#     stl = STL(df_country["Log_Pop"], period=1)
#     res = stl.fit()
#     smoothed_log = res.trend

#     # Use auto_arima for optimal ARIMA parameters
#     model = auto_arima(df_country["Diff_Pop"].dropna(), seasonal=False, stepwise=True, suppress_warnings=True)
#     # Forecast future values in log scale
#     forecast_log = model.predict(n_periods=len(forecast_years))

#     # Convert forecast back from log scale
#     forecast = np.exp(forecast_log)

#     # Last smoothed value (exponentiated)
#     last_smoothed = np.exp(smoothed_log.dropna().iloc[-1])

#     # Indexes
#     forecast_index = pd.Index(forecast_years)
#     smoothed_forecast = pd.concat([
#         pd.Series([last_smoothed], index=[df_country.index[-1]]),
#         pd.Series(forecast, index=forecast_index)
#     ])

#   # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(df_country.index, df_country["Population"], label="Historical Data", color="blue", linewidth=2)
#     plt.plot(smoothed_log.index, np.exp(smoothed_log), label="Smoothed Data", color="green", linewidth=2)
#     plt.plot(smoothed_forecast.index, smoothed_forecast.values, label="Forecasted Data", color="red", linestyle="--", linewidth=2)

#     plt.xlabel("Year")
#     plt.ylabel("Population")
#     plt.title(f"Population Forecast for {country_name} (2024–2034)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # Save the plot and CSV file
#     plot_filename = f"Graphs/Forecast_Plot_{country_name}.png"
#     df_country.to_csv(f"Data/{country_name}_Population.csv")  # Save the CSV as well
#     plt.savefig(plot_filename)
#     print(f"Saved forecast plot and CSV for {country_name}")
#     plt.show()
#     return pd.Series(forecast, index=forecast_index)


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.seasonal import STL
# from pmdarima import auto_arima

# def forecast_population(df_country, forecast_years, country_name):
#     """
#     Forecast population using STL smoothing and auto_arima, and save both plot and CSV.
#     Historical data: Blue, Smoothed: Green, Forecast continuation: Red dashed.
#     """
#     # Create necessary directories
#     os.makedirs("Graphs", exist_ok=True)
#     os.makedirs("datas", exist_ok=True)

#     # Log transform to stabilize variance
#     df_country["Log_Pop"] = np.log(df_country["Population"])

#     # STL decomposition
#     stl = STL(df_country["Log_Pop"], period=1)
#     res = stl.fit()
#     smoothed_log = res.trend

#     # Train ARIMA on log-transformed population (not differenced manually)
#     model = auto_arima(df_country["Log_Pop"], seasonal=False, stepwise=True, suppress_warnings=True)

#     # Forecast in log scale
#     forecast_log = model.predict(n_periods=len(forecast_years))
#     forecast = np.exp(forecast_log)  # Back to normal scale

#     # Get forecast index
#     forecast_index = pd.Index(forecast_years)

#     # Join last smoothed value with forecast to form a smooth continuation
#     last_smoothed = np.exp(smoothed_log.dropna().iloc[-1])
#     smoothed_forecast = pd.concat([
#         pd.Series([last_smoothed], index=[df_country.index[-1]]),
#         pd.Series(forecast, index=forecast_index)
#     ])

#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(df_country.index, df_country["Population"], label="Historical Data", color="blue", linewidth=2)
#     plt.plot(smoothed_log.index, np.exp(smoothed_log), label="Smoothed Trend", color="green", linewidth=2)
#     plt.plot(smoothed_forecast.index, smoothed_forecast.values, label="Forecasted Trend", color="red", linestyle="--", linewidth=2)

#     plt.xlabel("Year")
#     plt.ylabel("Population")
#     plt.title(f"Population Forecast for {country_name} (2024–2035)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # Save plot and CSV
#     plt.savefig(f"Graphs/Forecast_Plot_{country_name}.png")
#     df_country[["Population"]].to_csv(f"datas/{country_name}_Population.csv")
#     print(f"📊 Saved forecast plot and CSV for {country_name}")
#     plt.close()

#     return pd.Series(forecast, index=forecast_index)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def forecast_population(model, df_country, forecast_years, country_name):
    """
    Your original function with explicit smoothing-forecast connection
    """
    # 1. Historical Data
    historical = df_country["Population"]
    
    # 2. Smoothing (STL)
    stl = STL(np.log(historical), period=1)
    res = stl.fit()
    smoothed = np.exp(res.trend)
    
    # 3. Forecasting 
    forecast = model.forecast(steps=len(forecast_years))
    
    # Create smooth transition between last smoothed value and forecast
    last_smoothed = smoothed.iloc[-1]
    forecast_index = pd.Index(forecast_years)
    smoothed_forecast = pd.concat([
        pd.Series([last_smoothed], index=[df_country.index[-1]]),
        pd.Series(forecast, index=forecast_index)
    ])

    # Plotting (your original style)
    plt.figure(figsize=(10, 5))
    plt.plot(df_country.index, historical, label="Historical Data", color="blue", linewidth=2)
    plt.plot(smoothed.index, smoothed, label="Smoothed Data", color="green", linewidth=2)
    plt.plot(smoothed_forecast.index, smoothed_forecast, 
             label="Forecasted Data", color="red", linestyle="--", linewidth=2)

    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title(f"Population Forecast for {country_name}")
    plt.legend()
    plt.grid(True)
    
    # Save outputs
    plt.savefig(f"Graphs/Forecast_Plot_{country_name}.png")
    pd.DataFrame({
        'Year': smoothed_forecast.index,
        'Population': smoothed_forecast.values
    }).to_csv(f"datas/{country_name}_Forecast.csv")
    
    plt.close()
    
    return smoothed_forecast