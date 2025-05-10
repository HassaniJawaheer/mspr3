import os
import yaml
import joblib
import logging
import datetime
import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)

class XGBoostTuner:
    """Hyperparameter tuning for XGBoost on daily Eco2mix data."""

    def __init__(self, x_path, y_path, output_base_dir="data/07_models/eco2mix/xgboost_day/tunings", 
                 n_trials=20, timeout=None, seed=42):
        self.x_path = x_path
        self.y_path = y_path
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed

        self.X = pd.read_csv(self.x_path)
        self.y = pd.read_csv(self.y_path)

        timestamp = datetime.datetime.now().strftime("tuning_%Y-%m-%d_%H-%M")
        self.output_dir = os.path.join(output_base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.study_path = os.path.join(self.output_dir, "optuna_study.db")
        self.trials_csv = os.path.join(self.output_dir, "optuna_trials.csv")
        self.params_yaml = os.path.join(self.output_dir, "best_params.yml")

    def run(self):
        """Run Optuna tuning and export results."""
        fixed_params = {
            'n_jobs': -1,
            'random_state': self.seed
        }

        def objective(trial):
            params = {
                **fixed_params,
                "n_estimators": trial.suggest_int("n_estimators", 50, 80),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_int("min_child_weight", 1, 20),
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"])
            }
            model = MultiOutputRegressor(XGBRegressor(**params))
            score = cross_val_score(model, self.X, self.y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
            return -score.mean()

        study = optuna.create_study(
            direction='minimize',
            study_name="xgb_tuning",
            storage=f"sqlite:///{self.study_path}",
            load_if_exists=True
        )

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        self._export_results(study)
        logging.info(f"Tuning completed. Outputs saved in: {self.output_dir}")

    def _export_results(self, study):
        """Export best parameters, study trials and metadata."""
        best_params = study.best_params
        best_score = study.best_value

        info = {
            'best_params': best_params,
            'best_score_rmse': best_score,
            'date': datetime.datetime.now().isoformat(),
            'x_file': os.path.basename(self.x_path),
            'y_file': os.path.basename(self.y_path),
            'seed': self.seed,
            'n_trials': self.n_trials
        }

        with open(self.params_yaml, 'w') as f:
            yaml.dump(info, f)

        df_trials = study.trials_dataframe()
        df_trials.to_csv(self.trials_csv, index=False)
        logging.info("Best parameters and study exported.")

    @staticmethod
    def load_study(study_path, study_name="xgb_tuning"):
        """Load Optuna study from SQLite DB."""
        return optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")

    @staticmethod
    def plot_study(study):
        """Display interactive Optuna visualizations."""
        try:
            optuna.visualization.plot_optimization_history(study).show()
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_parallel_coordinate(study).show()
        except Exception as e:
            logging.warning(f"Failed to generate plots: {e}")
