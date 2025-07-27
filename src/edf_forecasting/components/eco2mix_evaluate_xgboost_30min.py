import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGEvaluate30min:
    def __init__(self, model, df_test, q_inf, q_sup, quantile):
        self.model = model
        self.df_test = df_test
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile = quantile

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.lower = None
        self.upper = None

    def _create_windows(self, params):
        windows_size = params["windows_size"]
        target_col = params["target_col"]

        if target_col not in self.df_test.columns:
            raise ValueError(f"Target column '{target_col}' not found in test dataframe.")

        values = self.df_test[target_col].values

        if len(values) <= windows_size:
            raise ValueError("Insufficient data to create at least one window.")

        self.X_test = np.lib.stride_tricks.sliding_window_view(values, window_shape=windows_size)[:-1]
        self.y_test = values[windows_size:]

    def _predict(self):
        if self.X_test is None:
            raise RuntimeError("X_test is not prepared. Call _create_windows first.")

        self.y_pred = self.model.predict(self.X_test)
        self.lower = self.y_pred + self.q_inf
        self.upper = self.y_pred + self.q_sup

    def _pinball_loss(self, y_true, y_pred, quantile):
        delta = y_true - y_pred
        return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))

    def run(self, params):
        self._create_windows(params)
        self._predict()

        y_true = self.y_test
        y_pred = self.y_pred
        lower = self.lower
        upper = self.upper

        return {
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "coverage": np.mean((y_true >= lower) & (y_true <= upper)),
            "interval_width": np.mean(upper - lower),
            "pinball_loss_lower": self._pinball_loss(y_true, lower, quantile=self.quantile / 2),
            "pinball_loss_upper": self._pinball_loss(y_true, upper, quantile=1 - self.quantile / 2),
            "overflow_rate": np.mean(y_true > upper)
        }
