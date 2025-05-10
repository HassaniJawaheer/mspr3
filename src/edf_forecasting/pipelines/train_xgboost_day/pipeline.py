"""
This is a boilerplate pipeline 'train_xgboost_day'
generated using Kedro 0.19.12
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import generate_plots_from_study, scrape_data, prestructure_data, clean_data, aggregate_data, add_tempo, add_features, preprocess_data, tune_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scrape_data,
            inputs="params:parameters_xgboost_training@scraping",
            outputs=None,
            name="scrape_data"
        ),

        node(
            func=prestructure_data,
            inputs="params:parameters_xgboost_training@prestructuring",
            outputs=None,
            name="prestructure_data"
        ),

        node(
            func=clean_data,
            inputs=[
                "consumption_data",
                "tempo_calendar",
                "params:parameters_xgboost_training@cleaning"
            ],
            outputs=["cleaned_consumption_data", "cleaned_tempo_calendar"],
            name="clean_data"
        ),

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
                "params:parameters_xgboost_training@add_tempo"
            ],
            outputs="tempo_aggregated_consumption_data",
            name="add_tempo"
        ),
        node(
            func=add_features,
            inputs=[
                "tempo_aggregated_consumption_data",
                "params:parameters_xgboost_training@add_features"
            ],
            outputs="features_day",
            name="add_features"
        ),
    
        node(
            func=preprocess_data,
            inputs=[
                "features_day",
                "params:parameters_xgboost_training@preprocessing"
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
                "params:parameters_xgboost_training@tuning"
            ],
            outputs=[
                "xgboost_best_params",
                "xgboost_tuning_summary"
            ],
            name="tune_model"
        ),

        node(
            func=generate_plots_from_study,
            inputs="xgboost_tuning_study_path",
            outputs="xgboost_tuning_plots_dir",
            name="generate_tuning_plots_node"
        ),

])
