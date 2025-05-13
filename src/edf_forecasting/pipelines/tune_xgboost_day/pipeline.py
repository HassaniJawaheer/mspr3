"""
This is a boilerplate pipeline 'tune_xgboost_day'
generated using Kedro 0.19.12
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import aggregate_data, add_tempo, add_features, preprocess_data, tune_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=aggregate_data,
            inputs="cleaned_consumption_data",
            outputs="aggregated_consumption_data",
            name="aggregate_data"
        ),

        node(
            func=add_tempo,
            inputs=[
                "aggregated_consumption_data",
                "cleaned_tempo_calendar",
                "params:add_tempo"
            ],
            outputs="tempo_aggregated_consumption_data",
            name="add_tempo"
        ),
        node(
            func=add_features,
            inputs=[
                "tempo_aggregated_consumption_data",
                "params:add_features"
            ],
            outputs="tempo_aggregated_consumption_enriched_data",
            name="add_features"
        ),
    
        node(
            func=preprocess_data,
            inputs=[
                "tempo_aggregated_consumption_enriched_data",
                "params:preprocessing"
            ],
            outputs=[
                "X_train_agg_day",
                "X_test_agg_day",
                "y_train_agg_day",
                "y_test_agg_day"
            ],
            name="preprocess_data"
        ),

        node(
            func=tune_model,
            inputs=[
                "X_train_agg_day",
                "y_train_agg_day",
                "params:tuning"
            ],
            outputs="xgboost_best_params",
            name="tune_model"
        )
    ])
