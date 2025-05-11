import pandas as pd
from typing import List

class Eco2mixPrepareTFTData:
    def __init__(self, series_id_value: str, columns_to_drop: List[str] = None):
        self.series_id_value = series_id_value
        self.columns_to_drop = columns_to_drop or []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

        df["time_idx"] = range(len(df))
        df["series_id"] = self.series_id_value

        df.drop(columns=self.columns_to_drop, inplace=True, errors="ignore")
        return df

       