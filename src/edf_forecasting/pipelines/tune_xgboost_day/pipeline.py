"""
This is a boilerplate pipeline 'tune_xgboost_day'
generated using Kedro 0.19.12
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import aggregate_data, add_tempo, add_features, preprocess_data, tune_model, generate_plots_from_study

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
            outputs="features_day",
            name="add_features"
        ),
    
        node(
            func=preprocess_data,
            inputs=[
                "features_day",
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
            outputs=[
                "xgboost_best_params",
                "xgboost_tuning_summary"
            ],
            name="tune_model"
        ),

        node(
            func=generate_plots_from_study,
            inputs=[
                "xgboost_tuning_summary",
                "xgboost_tuning_study",
                "params:plot_tuning_dir"
            ],
            outputs="tuning_plot_paths",
            name="generate_plots_from_study"
        )
    ])
