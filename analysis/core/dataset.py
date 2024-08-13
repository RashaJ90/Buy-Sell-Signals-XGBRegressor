from typing import List, Dict, Any
import pandas as pd 

from analysis.core.features import (
    calculate_accumulation_distribution_indicator,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_bollinger_width,
    calculate_ema,
    calculate_macd,
    calculate_dmi,
    calculate_stoch,
    calculate_rsi,
    calculate_williams
)

# Financial Data
import yfinance as yf
import quantstats as qs


def extract_data(ticker_symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    return yf.download(ticker_symbol, start=start, end=end, interval=interval)


def create_feature_dataset(
    df: pd.DataFrame,
    indicators: List[str],
    lookback_period: int = 1,
    indicator_params: Dict[str, Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Create a dataset with technical indicators for XGBoost model training.

    Parameters:
    - df: Input DataFrame with 'open', 'high', 'low', 'close', and 'volume' columns
    - indicators: List of indicator names to include in the dataset
    - lookback_period: Number of periods to look back for calculating returns (default is 1)
    - indicator_params: Dictionary of indicator-specific parameters (optional)

    Returns:
    - dataset: Pandas DataFrame containing the features and target variable
    """
    if indicator_params is None:
        indicator_params = {}

    # Ensure required columns are present
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Create a copy of the input DataFrame to avoid modifying the original
    dataset = df.copy()

    # Calculate returns
    dataset['return'] = dataset['close'].pct_change(periods=lookback_period)

    # Calculate and add requested indicators
    for indicator in indicators:
        if indicator == 'ADI':
            dataset['ADI'] = calculate_accumulation_distribution_indicator(
                dataset['close'].values, dataset['high'].values, dataset['low'].values, dataset['volume'].values
            )
        elif indicator == 'ATR':
            window = indicator_params.get('ATR', {}).get('window', 14)
            dataset['ATR'] = calculate_atr(
                dataset['open'].values, dataset['high'].values, dataset['low'].values, window
            )
        elif indicator == 'BB':
            window = indicator_params.get('BB', {}).get('window', 20)
            num_std = indicator_params.get('BB', {}).get('num_std', 2)
            upper, middle, lower = calculate_bollinger_bands(dataset['close'].values, window, num_std)
            dataset['BB_upper'] = upper
            dataset['BB_middle'] = middle
            dataset['BB_lower'] = lower
            dataset['BB_width'] = calculate_bollinger_width(upper, middle, lower)
        elif indicator == 'EMA':
            window = indicator_params.get('EMA', {}).get('window', 14)
            dataset['EMA'] = calculate_ema(dataset['close'].values, window)
        elif indicator == 'MACD':
            short_window = indicator_params.get('MACD', {}).get('short_window', 12)
            long_window = indicator_params.get('MACD', {}).get('long_window', 26)
            signal_window = indicator_params.get('MACD', {}).get('signal_window', 9)
            macd_hist, macd, macd_signal = calculate_macd(
                dataset['close'].values, short_window, long_window, signal_window
            )
            dataset['MACD'] = macd
            dataset['MACD_signal'] = macd_signal
            dataset['MACD_hist'] = macd_hist
        elif indicator == 'DMI':
            di_plus, di_minus, adx = calculate_dmi(
                dataset['high'].values, dataset['low'].values, dataset['open'].values
            )
            dataset['DI_plus'] = di_plus
            dataset['DI_minus'] = di_minus
            dataset['ADX'] = adx
        elif indicator == 'STOCH':
            n_fast_k = indicator_params.get('STOCH', {}).get('n_fast_k', 14)
            n_fast_d = indicator_params.get('STOCH', {}).get('n_fast_d', 3)
            n_slow_d = indicator_params.get('STOCH', {}).get('n_slow_d', 3)
            k, d_fast, d_slow = calculate_stoch(
                dataset['close'].values, dataset['low'].values, dataset['high'].values,
                n_fast_k, n_fast_d, n_slow_d
            )
            dataset['STOCH_K'] = k
            dataset['STOCH_D_fast'] = d_fast
            dataset['STOCH_D_slow'] = d_slow
        elif indicator == 'RSI':
            window = indicator_params.get('RSI', {}).get('window', 14)
            dataset['RSI'] = calculate_rsi(dataset['close'].values, window)
        elif indicator == 'WILLIAMS':
            window = indicator_params.get('WILLIAMS', {}).get('window', 14)
            dataset['WILLIAMS'] = calculate_williams(
                dataset['close'].values, dataset['low'].values, dataset['high'].values, window
            )

    # Remove rows with NaN values
    dataset = dataset.dropna()

    return dataset
