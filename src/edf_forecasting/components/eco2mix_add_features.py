import os
import pandas as pd
import requests
import logging
import holidays

logging.basicConfig(level=logging.INFO)

def read_df(path):
    return pd.read_csv(path)

def check_missing_values(df, verbose=True):
    nan_counts = df.isna().sum()
    total_missing = nan_counts.sum()
    if total_missing == 0:
        if verbose:
            print("No NaN detected in the DataFrame.")
    else:
        if verbose:
            print("NaNs are still present in the following columns:")
            print(nan_counts[nan_counts > 0])

class Eco2mixFeaturesDay:
    """Adds external features (temperature, sunshine, etc.) to daily Eco2mix data."""

    def __init__(self, input_path, filename):
        self.input_path = input_path
        self.filename = filename
        self.df = read_df(input_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def add_temperature(self, latitude=48.85, longitude=2.35):
        start_date = self.df['Date'].min().strftime("%Y-%m-%d")
        end_date = self.df['Date'].max().strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join([
                "apparent_temperature_min", "apparent_temperature_max", "apparent_temperature_mean"
            ]),
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            temp_df = pd.DataFrame(data["daily"])
            temp_df["Date"] = pd.to_datetime(temp_df["time"])
            temp_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(temp_df, on="Date", how="left")
            logging.info("Temperature features successfully added.")
        except requests.RequestException as e:
            logging.error(f"[Erreur API Open-Meteo] : {e}")

    def add_sunshine(self, latitude=48.85, longitude=2.35):
        start_date = self.df['Date'].min().strftime("%Y-%m-%d")
        end_date = self.df['Date'].max().strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "sunshine_duration",
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            sun_df = pd.DataFrame(data["daily"])
            sun_df["Date"] = pd.to_datetime(sun_df["time"])
            sun_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(sun_df, on="Date", how="left")
            logging.info("Sunshine duration successfully added.")
        except requests.RequestException as e:
            logging.error(f"[Erreur API Open-Meteo] : {e}")

    def add_weekday(self):
        jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        self.df["weekday"] = self.df["Date"].dt.dayofweek.apply(lambda x: jours_fr[x])
        logging.info("Weekday column added.")

    def add_month(self):
        mois_fr = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        self.df['month'] = self.df['Date'].dt.month.apply(lambda x: mois_fr[x - 1])
        logging.info("Month column added.")

    def add_season(self):
        def get_season(month):
            if month in [12, 1, 2]:
                return "hiver"
            elif month in [3, 4, 5]:
                return "printemps"
            elif month in [6, 7, 8]:
                return "été"
            else:
                return "automne"
        self.df['season'] = self.df['Date'].dt.month.apply(get_season)
        logging.info("Season column added.")

    def add_vacation(self):
        years = self.df['Date'].dt.year.unique()
        public_holidays = holidays.FR(years=years)

        zone_vacations_by_year = {
            "winter": ("02-05", "03-07"),
            "spring": ("04-09", "05-09"),
            "summer": ("07-06", "08-31"),
            "Toussaint": ("10-17", "11-02"),
            "christmas": ("12-18", "12-31")
        }

        vacations = set()
        for year in years:
            for start_suffix, end_suffix in zone_vacations_by_year.values():
                start = f"{year}-{start_suffix}"
                end = f"{year}-{end_suffix}"
                vacations.update(pd.date_range(start=start, end=end).date)

        self.df['is_vacation'] = self.df['Date'].dt.date.apply(lambda d: int(d in public_holidays or d in vacations))
        logging.info("Vacation flag added.")

    def save(self):
        folder = "data/06_features/eco2mix/day"
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_path = os.path.join(folder, self.filename)
        self.df.to_csv(output_path, index=False)
        logging.info(f"Data saved to {output_path}")

class Eco2mixFeaturesMinute:
    """Adds external features to 30-min interval Eco2mix data."""

    def __init__(self, input_path, filename):
        self.input_path = input_path
        self.filename = filename
        self.df = pd.read_csv(input_path)
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])

    def add_temperature(self, latitude=48.85, longitude=2.35):
        self.df['Hour'] = self.df['Datetime'].dt.floor('h')
        start_date = self.df['Hour'].min().strftime("%Y-%m-%d")
        end_date = self.df['Hour'].max().strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "apparent_temperature",
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            temp_df = pd.DataFrame(data["hourly"])
            temp_df["Hour"] = pd.to_datetime(temp_df["time"])
            temp_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(temp_df, on="Hour", how="left")
            logging.info("Temperature features successfully added.")
        except requests.RequestException as e:
            logging.error(f"[Open-Meteo API Error - Temperature]: {e}")

    def add_sunshine(self, latitude=48.85, longitude=2.35):
        self.df['Date'] = self.df['Datetime'].dt.date
        start_date = min(self.df['Date']).strftime("%Y-%m-%d")
        end_date = max(self.df['Date']).strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "sunshine_duration",
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            sun_df = pd.DataFrame(data["daily"])
            sun_df["Date"] = pd.to_datetime(sun_df["time"]).dt.date
            sun_df.drop(columns=["time"], inplace=True)
            self.df = self.df.merge(sun_df, on="Date", how="left")
            logging.info("Sunshine features successfully added.")
        except requests.RequestException as e:
            logging.error(f"[Open-Meteo API Error - Sunshine]: {e}")

    def add_weekday(self):
        self.df['Date'] = self.df['Datetime'].dt.date
        jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        self.df["weekday"] = pd.to_datetime(self.df["Date"]).dt.dayofweek.apply(lambda x: jours_fr[x])
        logging.info("Weekday column added.")

    def add_month(self):
        self.df['Date'] = self.df['Datetime'].dt.date
        mois_fr = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        self.df['month'] = pd.to_datetime(self.df['Date']).dt.month.apply(lambda x: mois_fr[x - 1])
        logging.info("Month column added.")

    def add_season(self):
        self.df['Date'] = self.df['Datetime'].dt.date

        def get_season(month):
            if month in [12, 1, 2]:
                return "hiver"
            elif month in [3, 4, 5]:
                return "printemps"
            elif month in [6, 7, 8]:
                return "été"
            else:
                return "automne"

        self.df['season'] = pd.to_datetime(self.df['Date']).dt.month.apply(get_season)
        logging.info("Season column added.")

    def add_vacation(self):
        self.df['Date'] = self.df['Datetime'].dt.date
        years = pd.to_datetime(self.df['Date']).dt.year.unique()
        public_holidays = holidays.FR(years=years)

        zone_vacations_by_year = {
            "winter": ("02-05", "03-07"),
            "spring": ("04-09", "05-09"),
            "summer": ("07-06", "08-31"),
            "Toussaint": ("10-17", "11-02"),
            "christmas": ("12-18", "12-31")
        }

        vacations = set()
        for year in years:
            for start_suffix, end_suffix in zone_vacations_by_year.values():
                start = f"{year}-{start_suffix}"
                end = f"{year}-{end_suffix}"
                vacations.update(pd.date_range(start=start, end=end).date)

        self.df['is_vacation'] = self.df['Date'].apply(lambda d: int(d in public_holidays or d in vacations))
        logging.info("Vacation flag added.")

    def save(self):
        folder = "data/06_features/eco2mix/minute"
        if not os.path.exists(folder):
            os.makedirs(folder)
        if 'Hour' in self.df.columns:
            self.df.drop(columns=['Hour'], inplace=True)
        if 'Date' in self.df.columns:
            self.df.drop(columns=['Date'], inplace=True)
        output_path = os.path.join(folder, self.filename)
        self.df.to_csv(output_path, index=False)
        logging.info(f"Data saved to {output_path}")
