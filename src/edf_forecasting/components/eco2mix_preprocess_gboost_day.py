import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2mixPreprocessGBoostDay:
    """Prepares daily Eco2mix data for XGBoost training using sliding windows."""

    def __init__(self, 
        input_path,
        output_dir,
        target_col_prefix='Consommation_',
        window_size=7,
        seed=42,
        features_to_include=None,
        target_features_to_include=None,
        test_size=0.2,
        shuffle=False
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col_prefix = target_col_prefix
        self.window_size = window_size
        self.seed = seed
        self.features_to_include = features_to_include or []
        self.target_features_to_include = target_features_to_include or []
        self.test_size = test_size
        self.shuffle = shuffle

    def run(self):
        df = self._read_df(self.input_path)
        df, mappings = self._encode_categorical(df, columns=self.features_to_include + self.target_features_to_include)

        X, y, dates = self._create_dataset_sliding(
            df,
            window_size=self.window_size,
            target_col_prefix=self.target_col_prefix,
            date_col='Date',
            features_to_include=self.features_to_include,
            target_features_to_include=self.target_features_to_include
        )

        self._save_split(X, y)
        logging.info("Preprocessing completed and files saved.")

    def _read_df(self, path):
        return pd.read_csv(path)

    def _create_dataset_sliding(self, df, window_size, target_col_prefix, date_col,
                                features_to_include, target_features_to_include):
        target_cols = [col for col in df.columns if col.startswith(target_col_prefix)]
        X_list, y_list, date_list = [], [], []

        for i in range(window_size, len(df)):
            window_df = df.iloc[i - window_size:i]
            features = []

            for j in range(window_size):
                row = window_df.iloc[j]
                if features_to_include:
                    features.extend(row[features_to_include].values.tolist())
                features.extend(row[target_cols].values.tolist())

            if target_features_to_include:
                features.extend(df.iloc[i][target_features_to_include].values.tolist())

            y_values = df.iloc[i][target_cols].values.tolist()
            X_list.append(np.array(features))
            y_list.append(y_values)
            date_list.append(df.iloc[i][date_col])

        return np.array(X_list), np.array(y_list), date_list

    def _encode_categorical(self, df, columns, start_at=0):
        df = df.copy()
        mappings = {}

        for col in columns:
            if self._is_boolean_column(df, col):
                df[col] = df[col].astype(int)
            else:
                unique_vals = sorted(df[col].dropna().unique())
                map_dict = {val: i + start_at for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(map_dict)
                mappings[col] = map_dict
        return df, mappings

    def _is_boolean_column(self, df, col):
        unique_vals = set(df[col].dropna().unique())
        return unique_vals <= {0, 1} or unique_vals <= {True, False}

    def _save_split(self, X, y):
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = self.test_size, shuffle = self.shuffle, random_state=self.seed
        )

        os.makedirs(self.output_dir, exist_ok=True)

        np.savetxt(os.path.join(self.output_dir, f"X_train_seed{self.seed}.csv"), X_train, delimiter=",")
        np.savetxt(os.path.join(self.output_dir, f"X_test_seed{self.seed}.csv"), X_test, delimiter=",")
        np.savetxt(os.path.join(self.output_dir, f"y_train_seed{self.seed}.csv"), y_train, delimiter=",")
        np.savetxt(os.path.join(self.output_dir, f"y_test_seed{self.seed}.csv"), y_test, delimiter=",")

        logging.info(f"Files saved to {self.output_dir} with seed {self.seed}")