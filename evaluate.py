import numpy as np

def compute_rmse(actual, predicted):
    """
    Computes Root Mean Squared Error (RMSE).

    Args:
    - actual (pd.Series): Actual population values.
    - predicted (pd.Series): Predicted population values.

    Returns:
    - float: RMSE value.
    """
    return np.sqrt(np.mean((actual.values - predicted.values) ** 2))

def compute_mape(actual, predicted):
    """
    Computes Mean Absolute Percentage Error (MAPE).

    Args:
    - actual (pd.Series): Actual population values.
    - predicted (pd.Series): Predicted population values.

    Returns:
    - float: MAPE value.
    """
    return np.mean(np.abs((actual.values - predicted.values) / actual.values)) * 100