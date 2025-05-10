import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Eco2mixCleaner:
    """Class for cleaning and preparing Eco2mix energy data."""

    def __init__(self,
        columns_to_keep,
        definitive_data_name='consumption_data.csv',
        tempo_data_name='tempo_calendar.csv',
        tempo_column_name='Type de jour Tempo',
        new_tempo_column_name='tempo',
        consumption_col='Consommation',
        base_input_dir='data/02_intermediate/eco2mix',
        base_output_dir='data/03_primary/eco2mix'
    ):
        """Initialize the cleaner with file names, column names, and parameters."""
        self.columns_to_keep = columns_to_keep
        self.definitive_data_name = definitive_data_name
        self.tempo_data_name = tempo_data_name
        self.tempo_column_name = tempo_column_name
        self.new_tempo_column_name = new_tempo_column_name
        self.consumption_col = consumption_col
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir

    def clean_definitive(self):
        """Clean the definitive consumption data."""
        try:
            input_path = os.path.join(self.base_input_dir, 'definitive', self.definitive_data_name)
            df = self._load_csv(input_path)

            df = df.dropna(subset=[self.consumption_col])
            df = df[self.columns_to_keep]

            output_dir = os.path.join(self.base_output_dir, 'definitive')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"cleaned_{self.definitive_data_name}")

            self._save_csv(df, output_path)
            logger.info(f"Definitive data cleaned and saved at {output_path}")

        except Exception as e:
            logger.error(f"Failed to clean definitive data: {e}")
            raise

    def clean_tempo(self):
        """Clean the tempo calendar data."""
        try:
            input_path = os.path.join(self.base_input_dir, 'tempo', self.tempo_data_name)
            df = self._load_csv(input_path)

            df = self._fill_missing_dates(df)
            df[self.tempo_column_name] = self._fill_missing_values(df, self.tempo_column_name)
            df = df.rename(columns={self.tempo_column_name: self.new_tempo_column_name})

            output_dir = os.path.join(self.base_output_dir, 'tempo')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"cleaned_{self.tempo_data_name}")

            self._save_csv(df, output_path)
            logger.info(f"Tempo data cleaned and saved at {output_path}")

        except Exception as e:
            logger.error(f"Failed to clean tempo data: {e}")
            raise

    def _load_csv(self, path):
        """Load a CSV file into a DataFrame."""
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _save_csv(self, df, path):
        """Save a DataFrame to CSV, overwriting if necessary."""
        df.to_csv(path, index=False)

    def _fill_missing_dates(self, df):
        """Fill missing dates by incrementing previous dates."""
        if 'Date' not in df.columns:
            logger.error("Missing 'Date' column in Tempo data.")
            raise ValueError("Missing 'Date' column in Tempo data.")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        missing_count_before = df['Date'].isna().sum()

        for idx in range(1, len(df)):
            if pd.isna(df.at[idx, 'Date']):
                if not pd.isna(df.at[idx - 1, 'Date']):
                    df.at[idx, 'Date'] = df.at[idx - 1, 'Date'] + pd.Timedelta(days=1)

        missing_count_after = df['Date'].isna().sum()

        if missing_count_after > 0:
            logger.warning(f"{missing_count_after} dates could not be filled after reconstruction.")

        return df

    def _fill_missing_values(self, df, column):
        """Fill missing values in a column with previous known value."""
        if column not in df.columns:
            logger.error(f"Missing column: {column}")
            raise ValueError(f"Missing column: {column}")

        return df[column].ffill()
