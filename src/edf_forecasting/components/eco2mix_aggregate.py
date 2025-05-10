import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2mixAggregate:
    """Class for aggregating Eco2mix 30-min data into daily vectors."""

    def __init__(self,
        definitive_data_name='consumption_data.csv',
        base_input_dir='data/03_primary/eco2mix',
        base_output_dir='data/04_aggregate/eco2mix'
    ):
        """Initialize the aggregator with file names and directories."""
        self.definitive_data_name = definitive_data_name
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir

    def aggregate(self):
        """Main method to aggregate data and save."""
        try:
            input_path = os.path.join(self.base_input_dir, 'definitive', self.definitive_data_name)
            df = self._load_csv(input_path)

            df_aggregated = self._aggregate_daywise(df)

            output_dir = os.path.join(self.base_output_dir, 'definitive')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"aggregated_{self.definitive_data_name}")

            self._save_csv(df_aggregated, output_path)
            logging.info(f"Aggregated data saved at {output_path}")

        except Exception as e:
            logging.error(f"Failed to aggregate definitive data: {e}")
            raise

    def _load_csv(self, path):
        """Load a CSV file into a DataFrame."""
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _save_csv(self, df, path):
        """Save a DataFrame to CSV."""
        df.to_csv(path, index=False)

    def _aggregate_daywise(self, df):
        """Aggregate 30-min data into one row per day with 48 half-hour features."""
        if 'Datetime' not in df.columns:
            logging.error("Missing 'Datetime' column in definitive data.")
            raise ValueError("Missing 'Datetime' column in definitive data.")

        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Date'] = df['Datetime'].dt.date
        df['HourMinute'] = df['Datetime'].dt.strftime('%Hh%M')

        df = df.drop(columns=['Datetime'])

        feature_cols = [col for col in df.columns if col not in ['Date', 'HourMinute']]

        daily_records = []

        for date, group in df.groupby('Date'):
            flat_record = {'Date': date}
            group = group.reset_index(drop=True)

            for idx, row in group.iterrows():
                time_label = row['HourMinute']
                for feature in feature_cols:
                    flat_record[f"{feature}_{time_label}"] = row[feature]

            daily_records.append(flat_record)

        df_final = pd.DataFrame(daily_records)

        return df_final