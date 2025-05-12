import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def forecast_population(model, df_country, forecast_years, country):
    """
    Forecasts the population for future years using the trained ARIMA model
    and visualizes both historical and forecasted trends.

    Args:
    - model: Trained ARIMA model.
    - df_country (pd.DataFrame): Historical population data.
    - forecast_years (list): List of years to forecast.
    - country (str): Country name.

    Returns:
    - List of forecasted population values.
    """
    try:
        # Forecast future population
        steps = len(forecast_years)
        forecast = model.forecast(steps=steps)

        # Ensure directory exists
        os.makedirs("Graphs", exist_ok=True)

        # Plot historical and forecasted population
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df_country.index, y=df_country["Population"], label="Historical Population", marker="o")
        sns.lineplot(x=forecast_years, y=forecast, label="Forecasted Population", marker="o", linestyle="--", color="red")

        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.title(f"Population Forecast for {country}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot in 'Graphs' folder
        graph_path = f"Graphs/Forecast_Plot_{country}.png"
        plt.savefig(graph_path)
        print(f"📊 Saved forecast plot for {country} in 'Graphs/' folder.")

        plt.show()

        return forecast.tolist()

    except Exception as e:
        print(f"❌ Error forecasting {country}: {e}")
        return []
