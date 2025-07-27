"""
This is a boilerplate pipeline 'train_xgboost_30min'
generated using Kedro 0.19.12
"""

from edf_forecasting.components.eco2mix_evaluate_xgboost_30min import XGEvaluate30min
from edf_forecasting.components.eco2mix_calibrate_xgboost_30min import XGBCalibrator30min
from edf_forecasting.components.eco2mix_train_xgboost_30min import Eco2mixTrainGBoost30min

def train(df_train, training_params):
    trainer = Eco2mixTrainGBoost30min(
        df_train=df_train,
        params=training_params
    )

    model, scores, metadata = trainer.run()
    return model, scores, metadata

def calibrate(df_data, model, error_type):
    calibrator = XGBCalibrator30min(
        df_cal=df_data,
        model=model,
        error_type=error_type
    )
    
    q_inf, q_sup = calibrator.run()
    return q_inf, q_sup

def evaluate(model, df_test, q_inf, q_sup, quantile):
    evaluator = XGEvaluate30min(
        model,
        df_test,
        q_inf,
        q_sup,
        quantile
    )

    results = evaluator.run()
    return results