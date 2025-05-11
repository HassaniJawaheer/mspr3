"""
This is a boilerplate pipeline 'train_tft'
generated using Kedro 0.19.12
"""
from kedro.pipeline import node, pipeline
from .nodes import prepare_tft_data, create_tsdatasets, train_tft, evaluate_tft

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=prepare_tft_data,
            inputs=["tempo_consumption_enriched_data:", "params:preprocess_tft"],
            outputs="prepared_tft_data",
            name="prepare_tft_data"
        ),

        node(
            func=create_tsdatasets,
            inputs=["prepared_tft_data", "params:tsdataset_tft"],
            outputs=[
                "tft_training_dataset",
                "tft_validation_dataset",
                "tft_test_dataset"
            ],
            name="create_tsdatasets"
        ),

        node(
            func=train_tft,
            inputs=["tft_training_dataset", "tft_validation_dataset", "params:train_tft"],
            outputs=["tft_model_artifact", "tft_training_metadata", "tft_training_scores"],
            name="train_tft"
        ),
        node(
            func=evaluate_tft,
            inputs=["tft_model_artifact", "tft_test_dataset", "params:train_tft.evaluate_tft"],
            outputs="tft_evaluation_scores",
            name="evaluate_tft_node"
        )
    ])
