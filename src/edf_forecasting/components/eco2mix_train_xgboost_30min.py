import logging
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainGBoost30min:
    def __init__(self, df_train, params):
        self.df_train = df_train
        self.params = params
        self.X_train = None
        self.y_train = None
    
    def _create_windows(self, params):
        windows_size = params["windows_size"]
        target_col = params["target_col"]

        if target_col not in self.df_train.columns:
            raise ValueError(f"Target column '{target_col}' not found in train dataframe.")
        
        values = self.df_train[target_col].values

        if len(values) <= windows_size:
            raise ValueError("Insufficient data to create at least one windows.")
        
        self.X_train = np.lib.stride_tricks.sliding_window_view(values, window_shape=windows_size)[:-1]
        self.y_train = values[windows_size:]

    def run(self):
        model = XGBRegressor(**self.params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_train)
        scores = {
            "r2_score": float(r2_score(self.y_train, y_pred)),
            "rmse": float(root_mean_squared_error(self.y_train, y_pred))
        }

        metadata = {
            "model": "XGBoostRegressor",
            "params_used": self.params,
            "n_samples": len(self.X)
        }

        return model, scores, metadata
