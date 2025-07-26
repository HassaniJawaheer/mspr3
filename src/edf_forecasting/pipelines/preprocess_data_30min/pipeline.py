"""
This is a boilerplate pipeline 'preprocess_data_30min'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, pipeline
from .nodes import *

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=check_frequency,
            inputs=["cleaned_consumption_data", "params:check_frequency"],
            outputs="checked_consumption_data",
            name="check_frequency"
        ),

        node(
            func=split_train_cal_test,
            inputs=["checked_consumption_data", "params:split_train_cal_test"],
            outputs=["train_checked_consumption_data", "cal_checked_consumption_data", "test_checked_consumption_data"],
            name="split_train_cal_test"
        )
    ])
