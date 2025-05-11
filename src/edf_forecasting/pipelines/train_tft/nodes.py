"""
This is a boilerplate pipeline 'train_tft'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_prepare_tft_data import Eco2mixPrepareTFTData
from edf_forecasting.components.eco2mix_create_tsdataset_tft import Eco2mixCreateTSDatasetsTFT
from edf_forecasting.components.eco2mix_train_model_tft import Eco2mixTrainModelTFT
from edf_forecasting.components.eco2mix_evaluate_model_tft import Eco2mixEvaluateModelTFT

def prepare_tft_data(df, params):
    preparer = Eco2mixPrepareTFTData(
        series_id_value=params["series_id"],
        columns_to_drop=params.get("columns_to_drop", [])
    )
    return preparer.run(df)

def create_tsdatasets(df, params):
    creator = Eco2mixCreateTSDatasetsTFT(
        test_cutoff_years=params["test_cutoff_years"],
        val_duration_years=params["val_duration_years"],
        max_encoder_length=params["max_encoder_length"],
        max_prediction_length=params["max_prediction_length"],
        known_reals=params["known_reals"],
        unknown_reals=params["unknown_reals"],
        known_categoricals=params.get("known_categoricals", []),
        unknown_categoricals=params.get("unknown_categoricals", []),
        static_reals=params.get("static_reals", []),
        static_categoricals=params.get("static_categoricals", [])
    )
    return creator.run(df)

def train_tft(training_ds, validation_ds, params):
    trainer = Eco2mixTrainModelTFT(params)
    return trainer.run(training_ds, validation_ds)

def evaluate_tft(model_path: str, test_dataset, params):
    evaluator = Eco2mixEvaluateModelTFT(batch_size=params.get("batch_size", 256))
    return evaluator.run(model_path, test_dataset)

