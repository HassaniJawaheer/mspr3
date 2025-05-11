"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 0.19.12
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import scrape_data, prestructure_data, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scrape_data,
            inputs="params:prepare_data@scraping",
            outputs=None,
            name="scrape_data"
        ),

        node(
            func=prestructure_data,
            inputs="params:prepare_data@prestructuring",
            outputs=None,
            name="prestructure_data"
        ),

        node(
            func=clean_data,
            inputs=[
                "consumption_data",
                "tempo_calendar",
                "params:prepare_data@cleaning"
            ],
            outputs=["cleaned_consumption_data", "cleaned_tempo_calendar"],
            name="clean_data"
        )
    ])