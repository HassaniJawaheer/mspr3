"""
This is a boilerplate pipeline 'preprocess_data_30min'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_add_tempo import Eco2MixAddTempo
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute
import pandas as pd

def add_tempo_min(df_data, df_tempo, params):
    mode = params["mode"]
    adder = Eco2MixAddTempo(mode=mode)
    return adder.add_tempo(df_data, df_tempo)

def add_features_min(df, params):
    include = params.get("include", [])
    engineer = Eco2mixFeaturesMinute(df)
    return engineer.run(include=include)

def check_frequency(df_data, params):
    df = df_data.copy()
    dt_col = params["datetime_col"]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)

    # Check for uniform frequency
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq != params["freq"]:
        raise ValueError(f"Inconsistent time step: expected {params['freq']}, got {inferred_freq}")
    
    return df

def split_train_cal_test(df, params):
    """
    Split the DataFrame into train, calibration, and test sets based on year boundaries.
    """
    df_train = df[df.index.year < params["cal_year"]]
    df_cal = df[df.index.year == params["cal_year"]]
    df_test = df[df.index.year == params["test_year"]]
    return df_train, df_cal, df_test