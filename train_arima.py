from statsmodels.tsa.arima.model import ARIMA
import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def grid_search_arima(series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_model = None
    best_order = None
    
    for p, d, q in itertools.product(
        range(max_p + 1),
        range(max_d + 1), 
        range(max_q + 1)
    ):
        try:
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_model = model_fit
                best_order = (p, d, q)
        except:
            continue
            
    return best_model, best_order

def time_series_cv(series, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    
    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        try:
            model = ARIMA(train, order=(1,1,1))
            model_fit = model.fit()
            preds = model_fit.get_forecast(steps=len(test)).predicted_mean
            rmse = np.sqrt(np.mean((preds.values - test.values) ** 2))
            rmses.append(rmse)
        except Exception as e:
            print(f"CV error: {str(e)}")
            continue
            
    return rmses