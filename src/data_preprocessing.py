import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import configparser
import os


class WineDataPreprocessor:
    def __init__(self, config_path='configs/config.ini'):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config.read(config_path)

    def load_data(self) -> pd.DataFrame:
        try:
            red_path = self.config['DATA']['red_wine_path']
            white_path = self.config['DATA']['white_wine_path']
        except KeyError as e:
            raise KeyError(f"Missing expected key in config file: {e}")

        if not os.path.exists(red_path):
            raise FileNotFoundError(f"Red wine data file not found at: {red_path}")
        if not os.path.exists(white_path):
            raise FileNotFoundError(f"White wine data file not found at: {white_path}")

        red_wine = pd.read_csv(red_path, sep=';')
        white_wine = pd.read_csv(white_path, sep=';')
        red_wine['wine_type'] = 'red'
        white_wine['wine_type'] = 'white'
        df = pd.concat([red_wine, white_wine], ignore_index=True)
        df = df.drop_duplicates()
        return df

    def preprocess_data(self, df: pd.DataFrame, target_col='quality'):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        df['quality_binary'] = (df[target_col] > 6).astype(int)
        feature_columns = [col for col in df.columns
                           if col not in [target_col, 'quality_binary', 'wine_type']]

        X = df[feature_columns]
        y = df['quality_binary']
        return X, y, feature_columns

    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    preprocessor = WineDataPreprocessor()
    df = preprocessor.load_data()
    X, y, features = preprocessor.preprocess_data(df)
    print(f"Features: {len(features)}")
    print(f"Samples: {len(X)}")