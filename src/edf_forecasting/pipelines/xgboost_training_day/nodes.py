"""
This is a boilerplate pipeline 'xgboost_training_day'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_train_gboost_day import Eco2mixTrainGBoostDay
from edf_forecasting.components.eco2mix_evaluate_gboost_day import Eco2mixEvaluateGBoostDay
from edf_forecasting.components.eco2mix_generate_prediction_plots_gboost_day import generate_prediction_plots

def train_model(X_train, y_train, training_params):
    trainer = Eco2mixTrainGBoostDay(
        X_train=X_train,
        y_train=y_train,
        params=training_params
    )
    model, scores, metadata = trainer.run()
    return model, scores, metadata


def evaluate_model(model, X_test, y_test):
    evaluator = Eco2mixEvaluateGBoostDay(model=model)
    scores = evaluator.run(X_test, y_test)
    return scores

def generate_plots(_, model, X_test, y_test, params):
    generate_prediction_plots(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_dir=params["plot_repo"],
        n_days=params["n_days"],
        random_seed=params["random_seed"]
    )


