from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def check_stationarity(series, country_name=""):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check if a time series is stationary,
    and plots the series and its differenced version (if non-stationary).

    Args:
    - series (pd.Series): The population data series.
    - country_name (str): Optional name of the country for visualization title.

    Returns:
    - bool: True if stationary, False if non-stationary.
    - float: p-value from the test.
    """
    result = adfuller(series.dropna())
    p_value = result[1]

    # Plot original series
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=series, label="Original")
    plt.title(f"Original Series - {country_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if p_value > 0.05:
        # Plot differenced series
        diff_series = series.diff().dropna()
        plt.figure(figsize=(10, 4))
        sns.lineplot(data=diff_series, label="First Difference", color="red")
        plt.title(f"First Differenced Series - {country_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return p_value <= 0.05, p_value  # Return True if stationary


def apply_differencing(series, order=1):
    """
    Apply differencing to the series to remove trends.

    Args:
    - series (pd.Series): The population data series.
    - order (int): The differencing order (1 for first difference, 2 for second difference).

    Returns:
    - pd.Series: Differenced series.
    """
    return series.diff(periods=order).dropna()
