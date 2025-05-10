import os
import yaml
import joblib
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixEvaluateGBoostDay:
    def __init__(self, x_test_path, y_test_path, model_path, output_root, random_seed=None, n_days=3):
        self.x_test_path = x_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path
        self.output_root = output_root
        self.random_seed = random_seed
        self.n_days = n_days
        self.eval_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.eval_dir = os.path.join(self.output_root, f"evaluation_{self.eval_time}")
        os.makedirs(self.eval_dir, exist_ok=True)

    def _load_data(self):
        X_test = pd.read_csv(self.x_test_path).values
        y_test = pd.read_csv(self.y_test_path).values
        return X_test, y_test

    def _load_model(self):
        return joblib.load(self.model_path)

    def _evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {"r2_score": float(r2), "rmse": float(rmse)}

    def _save_yaml(self, data, filename):
        path = os.path.join(self.eval_dir, filename)
        with open(path, 'w') as f:
            yaml.dump(data, f)

    def _plot_predictions(self, model, X_test, y_test):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        indexes = np.random.choice(len(X_test), size=self.n_days, replace=False)
        y_pred = model.predict(X_test[indexes])
        y_true = y_test[indexes]

        for i, idx in enumerate(indexes):
            plt.figure(figsize=(10, 4))
            plt.plot(y_true[i], label="True", linewidth=2)
            plt.plot(y_pred[i], label="Predicted", linestyle='--')
            plt.title(f"Prediction vs True - Sample {i+1}")
            plt.xlabel("15-minute intervals")
            plt.ylabel("Consumption")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fig_path = os.path.join(self.eval_dir, f"plot_{i+1}.png")
            plt.savefig(fig_path)
            plt.close()

    def run(self):
        logging.info("Loading test data and model...")
        X_test, y_test = self._load_data()
        model = self._load_model()

        logging.info("Evaluating model on test data...")
        scores = self._evaluate(model, X_test, y_test)
        self._save_yaml(scores, "scores.yml")

        logging.info("Generating prediction plots...")
        self._plot_predictions(model, X_test, y_test)

        logging.info(f"Evaluation complete. Results saved in {self.eval_dir}")
