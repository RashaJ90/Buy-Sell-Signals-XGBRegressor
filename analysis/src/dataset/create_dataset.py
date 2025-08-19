import os
from typing import Any, Optional, Type
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
import quantstats as qs
from sklearn.preprocessing import MinMaxScaler

from analysis.conf import settings
from analysis.src.features import build_features



def create_target_price(df: pd.DataFrame, look_forward_period: int = 1) -> pd.DataFrame:
    """
    Creates the target variable (future price) for regression.
    """
    df['future_close_price'] = df['close'].shift(-look_forward_period)
    df.dropna(subset=['future_close_price'], inplace=True)
    return df


class StocksRegressionDataset(Dataset):
    """
    PyTorch Dataset for stock time-series data, accepting pre-processed sequences.
    This version avoids data leakage by not performing scaling internally.
    """
    def __init__(self, X_sequences: np.ndarray, y_targets: np.ndarray):
        # Convert numpy arrays to torch tensors
        self.X = torch.tensor(X_sequences, dtype=torch.float32)
        self.y = torch.tensor(y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# adjust per model
def create_dataset(
    df: pd.DataFrame,
    feature_builder_class: Type[build_features.Features], # should be a class that inherits from build_features.Features
    lookback_period: int = 1, window: int=7,
    **feature_params: Any
) -> pd.DataFrame:
    """
    Creates a refined dataset using a provided feature engineering class.

    """
    # 1. Initialize the provided feature builder class
    feature_builder = feature_builder_class(df, **feature_params)
    features_df = feature_builder.add_all()

    # 2. Add additional required columns
    features_df['return'] = features_df['close'].pct_change(periods=lookback_period)

    if 'dividends' in features_df.columns:
        features_df['dividend_signal'] = (features_df['dividends'] > 0).astype(int)
    else:
        features_df['dividend_signal'] = 0

    atr = build_features.calculate_atr(features_df['open'], features_df['high'], features_df['low'], window=window)
    features_df['atr_normalized'] = (atr - atr.min()) / (atr.max() - atr.min())

    # 3. Define the final set of columns for the dataset
    # The custom features are now dynamically pulled from the class property
    final_feature_columns = [
        'close',
        'return',
        'dividend_signal',
        'atr_normalized',
    ] + feature_builder.feature_columns

    final_df = features_df[[col for col in final_feature_columns if col in features_df.columns]]

    final_df = final_df.dropna()

    return final_df
