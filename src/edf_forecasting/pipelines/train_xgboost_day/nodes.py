from edf_forecasting.components.eco2mix_scraper import Eco2MixScraper
from edf_forecasting.components.eco2mix_prestructuration_data import Eco2MixDataPreparator
from edf_forecasting.components.eco2mix_clean_data import Eco2mixCleaner
from edf_forecasting.components.eco2mix_aggregate import Eco2mixAggregate
from edf_forecasting.components.eco2mix_add_tempo import Eco2MixAddTempo
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesDay
from edf_forecasting.components.eco2mix_preprocess_gboost_day import Eco2mixPreprocessGBoostDay
from edf_forecasting.components.eco2mix_tune_gboost_day import XGBoostTuner
from edf_forecasting.components.eco2mix_train_gboost_day import Eco2mixTrainGBoostDay
from edf_forecasting.components.eco2mix_evaluate_gboost_day import Eco2mixEvaluateGBoostDay
from edf_forecasting.components.eco2mix_generate_tuning_plots_gboost_day import generate_tuning_plots
import os

# Download raw data: definitive + tempo
def scrape_data(params):
    output_dir = params["output_dir"]
    start_year_def = params["start_year_definitive"]
    end_year_def = params["end_year_definitive"]
    start_year_tempo = params["start_year_tempo"]
    end_year_tempo = params["end_year_tempo"]

    scraper = Eco2MixScraper(output_dir=output_dir)
    scraper.scrape_definitive_data(start_year_def, end_year_def)
    scraper.scrape_tempo_data(start_year_tempo, end_year_tempo)

# Prestructure raw data
def prestructure_data(params):
    raw_dir = params["raw_dir"]
    output_dir = params["output_dir"]
    start_year = params["start_year"]
    end_year = params["end_year"]

    preparator = Eco2MixDataPreparator(raw_dir, output_dir)
    preparator.prepare_consumption_data(start_year, end_year)
    preparator.prepare_tempo_calendar(start_year, end_year)

# Cleans structured data
def clean_data(df_definitive, df_tempo, params):
    cleaner = Eco2mixCleaner(
        columns_to_keep=params["columns_to_keep"],
        tempo_column_name=params["tempo_column_name"],
        new_tempo_column_name=params["new_tempo_column_name"],
        consumption_col=params["consumption_col"]
    )
    df_def_cleaned = cleaner.clean_definitive(df_definitive)
    df_tempo_cleaned = cleaner.clean_tempo(df_tempo)
    return df_def_cleaned, df_tempo_cleaned

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

def generate_plots_from_study(study_path: str) -> dict:
    output_dir = os.path.dirname(study_path).replace("optuna_study.db", "plots")
    return generate_tuning_plots(study_path, output_dir)
