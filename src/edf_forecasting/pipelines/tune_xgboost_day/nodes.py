from edf_forecasting.components.eco2mix_aggregate import Eco2mixAggregate
from edf_forecasting.components.eco2mix_add_tempo import Eco2MixAddTempo
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesDay
from edf_forecasting.components.eco2mix_preprocess_gboost_day import Eco2mixPreprocessGBoostDay
from edf_forecasting.components.eco2mix_tune_gboost_day import XGBoostTuner
from edf_forecasting.components.eco2mix_generate_tuning_plots_gboost_day import generate_tuning_plots
import os
import optuna

# Aggregate data 30min daily data
def aggregate_data(df):
    aggregator = Eco2mixAggregate()
    return aggregator.aggregate(df)

# Add tempo data (Bleu/Blanc/Rouge)
def add_tempo(df_data, df_tempo, params):
    tempo_adder = Eco2MixAddTempo(mode=params["mode"])
    return tempo_adder.add_tempo(df_data, df_tempo)

# Added weather, calendar, etc.
def add_features(df, params):
    extractor = Eco2mixFeaturesDay(df)
    return extractor.run(include=params["include"])

# Prepares X/y train/test datasets
def preprocess_data(df, params):
    processor = Eco2mixPreprocessGBoostDay(
        target_col_prefix=params["target_col_prefix"],
        window_size=params["window_size"],
        seed=params["seed"],
        features_to_include=params["features_to_include"],
        target_features_to_include=params["target_features_to_include"],
        test_size=params["test_size"],
        shuffle=params["shuffle"]
    )
    X_train, X_test, y_train, y_test = processor.run(df)
    return X_train, X_test, y_train, y_test

# 8. Tuning (with Optuna)
def tune_model(X, y, params):
    tuner = XGBoostTuner(
        n_trials=params["n_trials"],
        timeout=params["timeout"],
        seed=params["seed"]
    )
    best_params, summary = tuner.run(X, y)
    return best_params, summary

def generate_plots_from_study(_, study: optuna.Study, params) -> dict:
    return generate_tuning_plots(study, params["output_dir"])
