from statsmodels.tsa.arima.model import ARIMA
import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def grid_search_arima(series, max_p=5, max_d=2, max_q=5, max_P=2, max_Q=2, max_D=1):
    """
    Performs grid search to find the best ARIMA model parameters.

    Args:
    - series (pd.Series): Population data series.
    - max_p, max_d, max_q: ARIMA parameters.

    Returns:
    - best_model: Fitted ARIMA model with the best parameters.
    - best_order: The optimal order (p, d, q, P, D, Q).
    """
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    seasonal_p_values = range(0, max_P + 1)
    seasonal_q_values = range(0, max_Q + 1)

    best_rmse = np.inf
    best_model = None
    best_order = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q in itertools.product(seasonal_p_values, seasonal_q_values):
            try:
                # Fit ARIMA model with current parameters
                model = ARIMA(series, order=(p, d, q),
                              seasonal_order=(P, D, Q, 12))  # 12 for monthly seasonality (adjust if needed)
                model_fit = model.fit()

                # Evaluate using RMSE
                predictions = model_fit.predict(start=len(series), end=len(series) + len(series) - 1)
                rmse = np.sqrt(np.mean((predictions - series[-len(predictions):]) ** 2))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_fit
                    best_order = (p, d, q, P, D, Q)
            except Exception as e:
                continue

    return best_model, best_order


def time_series_cv(series, model_class, n_splits=5):
    """
    Perform cross-validation for time series data.

    Args:
    - series (pd.Series): Time series data.
    - model_class (class): ARIMA or other model class.
    - n_splits (int): Number of splits for cross-validation.

    Returns:
    - List of RMSE for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []

    for train_idx, test_idx in tscv.split(series):
        train, test = series[train_idx], series[test_idx]
        model = model_class(order=(1, 1, 1))  # You can customize the order here
        model_fit = model.fit(train)

        # Evaluate RMSE
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
        rmse = np.sqrt(np.mean((predictions - test) ** 2))
        rmses.append(rmse)

    return rmses
