import numpy as np
import pandas as pd

def calculate_accumulation_distribution_indicator(close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate Accumulation/Distribution Indicator (ADL or ADO) for a given security.

    Parameters:
    - close_prices: Close prices series (numpy array)
    - high_prices: High prices series (numpy array)
    - low_prices: Low prices series (numpy array)
    - volume: Volume series (numpy array)

    Returns:
    - adl: Accumulation/Distribution Indicator series (numpy array)
    """
    money_flow_multiplier = ((close_prices - low_prices) - (high_prices - close_prices)) / (high_prices - low_prices)
    money_flow_volume = money_flow_multiplier * volume
    adl = np.cumsum(money_flow_volume)
    return adl

def calculate_atr(open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Average True Range (ATR) for a given security.

    Parameters:
    - open_prices: Open prices series (numpy array)
    - high_prices: High prices series (numpy array)
    - low_prices: Low prices series (numpy array)
    - window: Size of the moving average window

    Returns:
    - atr: Average True Range series (numpy array)
    """
    daily_variation = (high_prices - low_prices) / open_prices
    return pd.Series(daily_variation).rolling(window=window).mean().values

def calculate_bollinger_bands(close_prices: np.ndarray, window: int, num_std: int = 2):
    """
    Calculate the Bollinger Bands for a given set of closing prices.

    Parameters:
    - close_prices: Close prices series (numpy array)
    - window: Size of the moving average window
    - num_std: Number of standard deviations for the bands (default is 2)

    Returns:
    - bollinger_upper: Upper Bollinger Band values (numpy array)
    - sma: Simple Moving Average values (numpy array)
    - bollinger_lower: Lower Bollinger Band values (numpy array)
    """
    close_series = pd.Series(close_prices)
    sma = close_series.rolling(window=window).mean()
    rolling_std = close_series.rolling(window=window).std()

    bollinger_upper = sma + (rolling_std * num_std)
    bollinger_lower = sma - (rolling_std * num_std)

    return np.round(bollinger_upper, 2), np.round(sma, 2), np.round(bollinger_lower, 2)

def calculate_bollinger_width(bollinger_upper: np.ndarray, sma: np.ndarray, bollinger_lower: np.ndarray) -> np.ndarray:
    """
    Calculate Bollinger Band Width.

    Parameters:
    - bollinger_upper: Upper Bollinger Band values (numpy array)
    - sma: Simple Moving Average values (numpy array)
    - bollinger_lower: Lower Bollinger Band values (numpy array)

    Returns:
    - bollinger_width: Bollinger Band Width values (numpy array)
    """
    return (bollinger_upper - bollinger_lower) / sma

def calculate_ema(close_prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    Parameters:
    - close_prices: Close prices series (numpy array)
    - window: Size of the EMA window

    Returns:
    - ema: Exponential Moving Average values (numpy array)
    """
    return pd.Series(close_prices).ewm(span=window, adjust=False).mean().values

def calculate_macd(close_prices: np.ndarray, short_window: int, long_window: int, signal_window: int):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Parameters:
    - close_prices: Close prices series (numpy array)
    - short_window: Short-term EMA window
    - long_window: Long-term EMA window
    - signal_window: Signal line EMA window

    Returns:
    - macd_histogram: MACD histogram values (numpy array)
    - macd: MACD line values (numpy array)
    - macd_signal: MACD signal line values (numpy array)
    """
    short_ema = pd.Series(close_prices).ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = pd.Series(close_prices).ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd_histogram.values, macd.values, macd_signal.values

def calculate_dmi(high_prices: np.ndarray, low_prices: np.ndarray, open_prices: np.ndarray, window: int):
    """
    Calculate Directional Movement Index (DMI).

    Parameters:
    - high_prices: High prices series (numpy array)
    - low_prices: Low prices series (numpy array)
    - open_prices: Open prices series (numpy array)

    Returns:
    - di_plus: Positive Directional Indicator values (numpy array)
    - di_minus: Negative Directional Indicator values (numpy array)
    - adx: Average Directional Movement Index values (numpy array)
    """
    high_shifted = pd.Series(high_prices).shift(1)
    low_shifted = pd.Series(low_prices).shift(1)
    dm_plus = pd.Series(high_prices) - high_shifted
    dm_minus = low_shifted - pd.Series(low_prices)
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    atr = calculate_atr(open_prices, high_prices, low_prices, window=window)
    di_plus = 100 * calculate_ema(dm_plus.values, window=window) / atr 
    di_minus = 100 * (calculate_ema(dm_minus.values, window=window) / atr)
    adx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    return di_plus, di_minus, adx

def calculate_stoch(close_prices: np.ndarray, low_prices: np.ndarray, high_prices: np.ndarray, n_fast_k: int = 14, n_fast_d: int = 3, n_slow_d: int = 3):
    """
    Calculate Stochastic Oscillator (STOCH).

    Parameters:
    - close_prices: Close prices series (numpy array)
    - low_prices: Low prices series (numpy array)
    - high_prices: High prices series (numpy array)
    - n_fast_k: Fast %K period (default is 14)
    - n_fast_d: Fast %D period (default is 3)
    - n_slow_d: Slow %D period (default is 3)

    Returns:
    - k: Fast %K values (numpy array)
    - d_fast: Fast %D values (numpy array)
    - d_slow: Slow %D values (numpy array)
    """
    l14 = pd.Series(low_prices).rolling(window=n_fast_k).min()
    h14 = pd.Series(high_prices).rolling(window=n_fast_k).max()
    k = 100 * (pd.Series(close_prices) - l14) / (h14 - l14) 
    d_fast = k.rolling(window=n_fast_d).mean()  
    d_slow = d_fast.rolling(window=n_slow_d).mean()
    return k.values, d_fast.values, d_slow.values

def calculate_rsi(close_prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
    - close_prices: Close prices series (numpy array)
    - window: RSI period (default is 14)

    Returns:
    - rsi: Relative Strength Index values (numpy array)
    """
    delta = pd.Series(close_prices).diff()
    delta = delta[1:] 
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, min_periods=window).mean()
    ema_down = down.ewm(com=window-1, min_periods=window).mean()
    return (100 * ema_up / (ema_down + ema_up)).values

def calculate_williams(close_prices: np.ndarray, low_prices: np.ndarray, high_prices: np.ndarray, window: int ) -> np.ndarray:
    """
    Calculate Williams %R.

    Parameters:
    - close_prices: Close prices series (numpy array)
    - low_prices: Low prices series (numpy array)
    - high_prices: High prices series (numpy array)
    - window: Williams %R period (default is 14)

    Returns:
    - williams: Williams %R values (numpy array)
    """
    highest_high = pd.Series(high_prices).rolling(window).max()
    lowest_low = pd.Series(low_prices).rolling(window).min()
    williams = ((highest_high - pd.Series(close_prices)) / (highest_high - lowest_low)) * -100
    return williams.values