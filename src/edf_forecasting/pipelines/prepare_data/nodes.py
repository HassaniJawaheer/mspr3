"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 0.19.12
"""
from edf_forecasting.components.eco2mix_scraper import Eco2MixScraper
from edf_forecasting.components.eco2mix_prestructuration_data import Eco2MixDataPreparator
from edf_forecasting.components.eco2mix_clean_data import Eco2mixCleaner

# Download raw data: definitive + tempo
def scrape_data(params):
    output_dir = params["output_dir"]
    start_year_def = params["start_year_definitive"]
    end_year_def = params["end_year_definitive"]
    start_year_tempo = params["start_year_tempo"]
    end_year_tempo = params["end_year_tempo"]

    scraper = Eco2MixScraper(output_dir=output_dir)
    scraper.scrape_definitive_data(start_year_def, end_year_def)
    scraper.scrape_tempo_data(start_year_tempo, end_year_tempo)

# Prestructure raw data
def prestructure_data(params):
    raw_dir = params["raw_dir"]
    output_dir = params["output_dir"]
    start_year = params["start_year"]
    end_year = params["end_year"]

    preparator = Eco2MixDataPreparator(raw_dir, output_dir)
    preparator.prepare_consumption_data(start_year, end_year)
    preparator.prepare_tempo_calendar(start_year, end_year)

# Cleans structured data
def clean_data(df_definitive, df_tempo, params):
    cleaner = Eco2mixCleaner(
        columns_to_keep=params["columns_to_keep"],
        tempo_column_name=params["tempo_column_name"],
        new_tempo_column_name=params["new_tempo_column_name"],
        consumption_col=params["consumption_col"]
    )
    df_def_cleaned = cleaner.clean_definitive(df_definitive)
    df_tempo_cleaned = cleaner.clean_tempo(df_tempo)
    return df_def_cleaned, df_tempo_cleaned
