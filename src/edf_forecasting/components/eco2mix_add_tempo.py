import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2MixAddTempo:
    """Class for adding tempo information to Eco2mix data."""

    def __init__(self,
        definitive_data_name,
        tempo_data_name,
        base_input_dir='data/04_aggregate/eco2mix',
        tempo_input_dir='data/03_primary/eco2mix/tempo',
        base_output_dir='data/05_tempo/eco2mix',
        mode='aggregate_day'  # or 'aggregate_minute'
    ):
        """Initialize paths and parameters."""
        self.definitive_data_name = definitive_data_name
        self.tempo_data_name = tempo_data_name
        self.base_input_dir = base_input_dir
        self.tempo_input_dir = tempo_input_dir
        self.base_output_dir = base_output_dir
        self.mode = mode

    def add_tempo(self):
        """Main method to add tempo information."""
        try:
            input_path = os.path.join(self.base_input_dir, 'definitive', self.definitive_data_name)
            tempo_path = os.path.join(self.tempo_input_dir, self.tempo_data_name)

            df_data = self._load_csv(input_path)
            df_tempo = self._load_csv(tempo_path)

            if self.mode == 'aggregate_day':
                df_final = self._add_tempo_day(df_data, df_tempo)
            elif self.mode == 'aggregate_minute':
                df_final = self._add_tempo_minute(df_data, df_tempo)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            output_dir = os.path.join(self.base_output_dir, self.mode)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"tempo_{self.definitive_data_name}")

            self._save_csv(df_final, output_path)
            logging.info(f"tempo added and saved at {output_path}")

        except Exception as e:
            logging.error(f"Failed to add tempo: {e}")
            raise

    def _load_csv(self, path):
        """Load a CSV file."""
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _save_csv(self, df, path):
        """Save a DataFrame to CSV."""
        if os.path.exists(path):
            os.remove(path) 
        df.to_csv(path, index=False)

    def _add_tempo_day(self, df_data, df_tempo):
        """Add tempo to aggregated daily data."""
        df_tempo['Date'] = pd.to_datetime(df_tempo['Date']).dt.date
        df_data['Date'] = pd.to_datetime(df_data['Date']).dt.date
        return df_data.merge(df_tempo[['Date', 'tempo']], how='left', on='Date')

    def _add_tempo_minute(self, df_data, df_tempo):
        """Add tempo to raw 30-min data."""
        df_tempo['Date'] = pd.to_datetime(df_tempo['Date']).dt.date
        df_data['date'] = pd.to_datetime(df_data['Datetime']).dt.date
        df_final = df_data.merge(df_tempo[['Date', 'tempo']], how='left', left_on='date', right_on='Date')
        df_final = df_final.drop(columns=['date', 'Date'])
        return df_final
