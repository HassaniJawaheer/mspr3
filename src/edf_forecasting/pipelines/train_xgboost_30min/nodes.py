"""
This is a boilerplate pipeline 'train_xgboost_30min'
generated using Kedro 0.19.12
"""

from edf_forecasting.components.eco2mix_evaluate_xgboost_30min import XGBEvaluate30min
from edf_forecasting.components.eco2mix_calibrate_xgboost_30min import XGBCalibrator30min
from edf_forecasting.components.eco2mix_train_xgboost_30min import Eco2mixTrainGBoost30min

def train(df_train, training_params, params):
    trainer = Eco2mixTrainGBoost30min(
        df_train=df_train,
        training_params=training_params,
        windows_size=params["windows_size"],
        target_col=params["target_col"]

    )

    model, scores, metadata = trainer.run()
    return model, scores, metadata

def calibrate(df_data, model, params):
    calibrator = XGBCalibrator30min(
        df_cal=df_data,
        model=model,
        error_type=params["error_type"],
        windows_size = params["windows_size"],
        target_col=params["target_col"]
    )
    
    q_inf, q_sup = calibrator.run(alpha=params["alpha"])
    return q_inf, q_sup

def evaluate(model, df_test, q_inf, q_sup, params):
    evaluator = XGBEvaluate30min(
        model,
        df_test,
        q_inf,
        q_sup,
        params["quantile"],
        params["windows_size"],
        params["target_col"]
    )

    results = evaluator.run()
    return results