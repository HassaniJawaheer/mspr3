import os
import yaml
import joblib
import logging
import datetime
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainGBoostDay:
    def __init__(self, x_path, y_path, params_path, output_root):
        self.x_path = x_path
        self.y_path = y_path
        self.params_path = params_path
        self.output_root = output_root

        self.run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_dir = os.path.join(self.output_root, self.run_time)
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_data(self):
        X = pd.read_csv(self.x_path).values
        y = pd.read_csv(self.y_path).values
        return X, y

    def _load_params(self):
        with open(self.params_path, 'r') as f:
            raw = yaml.safe_load(f)
        return raw.get("best_params", raw)

    def _save_yaml(self, data, filename):
        path = os.path.join(self.model_dir, filename)
        with open(path, 'w') as f:
            yaml.dump(data, f)

    def _evaluate(self, model, X, y):
        y_pred = model.predict(X)
        return {
            'r2_score': float(r2_score(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred)))
        }

    def run(self):
        logging.info("Loading training data...")
        X, y = self._load_data()

        logging.info("Loading training parameters...")
        params = self._load_params()

        seed = params.get("random_state", 42)
        model = MultiOutputRegressor(XGBRegressor(**params))

        logging.info("Training model...")
        model.fit(X, y)

        logging.info("Evaluating model...")
        scores = self._evaluate(model, X, y)

        joblib.dump(model, os.path.join(self.model_dir, "model.joblib"))
        self._save_yaml(params, "params_used.yml")
        self._save_yaml(scores, "scores.yml")

        training_metadata = {
            'run_time': self.run_time,
            'date': datetime.datetime.now().isoformat(),
            'seed': seed,
            'model_name': 'XGBoost_MultiOutput',
            'input_data': {
                'x_train': self.x_path,
                'y_train': self.y_path
            },
            'param_source': self.params_path
        }
        self._save_yaml(training_metadata, "training.yml")

        logging.info(f"Training completed. Artifacts saved to {self.model_dir}")
