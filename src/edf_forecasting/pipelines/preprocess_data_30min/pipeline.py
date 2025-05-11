"""
This is a boilerplate pipeline 'preprocess_data_30min'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, pipeline
from .nodes import add_tempo_min, add_features_min

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=add_tempo_min,
            inputs=["cleaned_consumption_data", "cleaned_tempo_calendar", "params:preprocess_params"],
            outputs="tempo_consumption_data",
            name="add_tempo_min"
        ),

        node(
            func=add_features_min,
            inputs=["tempo_consumption_data", "params:feature_params"],
            outputs="tempo_consumption_enriched_data",
            name="add_features_min"
        )
    ])
