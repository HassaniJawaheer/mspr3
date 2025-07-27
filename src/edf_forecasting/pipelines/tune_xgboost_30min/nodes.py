"""
This is a boilerplate pipeline 'tune_xgboost_30min'
generated using Kedro 0.19.12
"""
import numpy as np
from edf_forecasting.components.eco2mix_tune_xgboost_30min import XGBoostTuner

def create_windows(df_data, params):
    df = df_data.copy()
    window_size = params["window_size"]
    target_col = params["target_col"]

    if target_col not in df.columns:
        raise ValueError(f"Target colums '{target_col}' not found.")
    
    values = df[target_col].values
    if len(values) <= window_size:
        raise ValueError("Insufficient data size to create at least one window.")

    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]

    return X, y

# Tuning (with Optuna)
def tune(X, y, params):
    tuner = XGBoostTuner(
        n_trials=params["n_trials"],
        timeout=params["timeout"],
        cv = params["cv"],
        seed=params["seed"]
    )

    best_params, _ = tuner.run(X,y)
    return best_params