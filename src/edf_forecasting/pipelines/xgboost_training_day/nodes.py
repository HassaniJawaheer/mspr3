"""
This is a boilerplate pipeline 'xgboost_training_day'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_train_gboost_day import Eco2mixTrainGBoostDay
from edf_forecasting.components.eco2mix_cross_validation_xgboost_day import Eco2mixCrossValidationXGBoostDay
from edf_forecasting.components.eco2mix_evaluate_gboost_day import Eco2mixEvaluateGBoostDay
from edf_forecasting.components.eco2mix_generate_prediction_plots_gboost_day import generate_prediction_plots

def train_model(X_train, y_train, training_params):
    trainer = Eco2mixTrainGBoostDay(
        X_train=X_train,
        y_train=y_train,
        params=training_params
    )
    return trainer.run()

def cross_validate_model(X_train, y_train, training_params, validation_params):
    validator = Eco2mixCrossValidationXGBoostDay(
        X=X_train,
        y=y_train,
        training_params=training_params,
        cv_params=validation_params
    )
    return validator.run()

def evaluate_model(model, X_test, y_test):
    evaluator = Eco2mixEvaluateGBoostDay(model=model)
    return evaluator.run(X_test, y_test)


def generate_plots(model, X_test, y_test, params):
    generate_prediction_plots(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_dir=params["plot_repo"],
        n_days=params["n_days"],
        random_seed=params["random_seed"]
    )


