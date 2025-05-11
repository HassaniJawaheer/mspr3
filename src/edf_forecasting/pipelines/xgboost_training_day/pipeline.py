"""
This is a boilerplate pipeline 'xgboost_training_day'
generated using Kedro 0.19.12
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, cross_validate_model, evaluate_model, generate_plots

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=[
                "X_train_agg_day",
                "y_train_agg_day",
                "params:parameters_xgboost_training_day@training_params"
            ],
            outputs=[
                "xgboost_model",
                "xgboost_training_scores",
                "xgboost_training_metadata"
            ],
            name="train_model"
        ),

        node(
            func=cross_validate_model,
            inputs=[
                "X_train_agg_day",
                "y_train_agg_day",
                "params:parameters_xgboost_training_day@training_params",
                "params:parameters_xgboost_training_day@validate_params"
            ],
            outputs="xgboost_crossval_scores",
            name="cross_validate_model"
        ),

        node(
            func=evaluate_model,
            inputs=[
                "xgboost_model",
                "X_test_agg_day",
                "y_test_agg_day"
            ],
            outputs="xgboost_evaluation_scores",
            name="evaluate_model"
        ),
        node(
            func=generate_plots,
            inputs=[
                "xgboost_model",
                "X_test_agg_day",
                "y_test_agg_day",
                "params:parameters_xgboost_training_day@plots_params"
             ],
            outputs=None,
            name="generate_prediction_plots"
        ),

    ])
