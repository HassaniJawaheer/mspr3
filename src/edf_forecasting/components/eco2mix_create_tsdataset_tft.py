import pandas as pd
from typing import List, Optional
from datetime import timedelta
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

class Eco2mixCreateTSDatasetsTFT:
    def __init__(
        self,
        test_cutoff_years: float,
        val_duration_years: float,
        max_encoder_length: int,
        max_prediction_length: int,
        known_reals: List[str],
        unknown_reals: List[str],
        known_categoricals: Optional[List[str]] = None,
        unknown_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        add_relative_time_idx: bool = True,
        add_target_scales: bool = True,
        add_encoder_length: bool = True,
        target_normalizer=None
    ):
        self.test_cutoff_years = test_cutoff_years
        self.val_duration_years = val_duration_years
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.known_reals = known_reals
        self.unknown_reals = unknown_reals
        self.known_categoricals = known_categoricals or []
        self.unknown_categoricals = unknown_categoricals or []
        self.static_reals = static_reals or []
        self.static_categoricals = static_categoricals or []
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.target_normalizer = target_normalizer or GroupNormalizer(groups=["series_id"])

    def run(self, df: pd.DataFrame):
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        latest_date = df["Datetime"].max()

        test_start = latest_date - timedelta(days=round(365 * self.test_cutoff_years))
        val_start = test_start - timedelta(days=round(365 * self.val_duration_years))

        df_train = df[df["Datetime"] < val_start].copy()
        df_val = df[(df["Datetime"] >= val_start) & (df["Datetime"] < test_start)].copy()
        df_test = df[df["Datetime"] >= test_start].copy()

        training = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="Consommation",
            group_ids=["series_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=self.known_reals,
            time_varying_unknown_reals=self.unknown_reals,
            time_varying_known_categoricals=self.known_categoricals,
            time_varying_unknown_categoricals=self.unknown_categoricals,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            target_normalizer=self.target_normalizer,
            add_relative_time_idx=self.add_relative_time_idx,
            add_target_scales=self.add_target_scales,
            add_encoder_length=self.add_encoder_length
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, df_val, predict=True, stop_randomization=True
        )

        test = TimeSeriesDataSet.from_dataset(
            training, df_test, predict=True, stop_randomization=True
        )

        return training, validation, test
