"""
This is a boilerplate pipeline 'preprocess_data_30min'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_add_tempo import Eco2MixAddTempo
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute

def add_tempo_min(df_data, df_tempo, params):
    mode = params["mode"]
    adder = Eco2MixAddTempo(mode=mode)
    return adder.add_tempo(df_data, df_tempo)

def add_features_min(df, params):
    include = params.get("include", [])
    engineer = Eco2mixFeaturesMinute(df)
    return engineer.run(include=include)

