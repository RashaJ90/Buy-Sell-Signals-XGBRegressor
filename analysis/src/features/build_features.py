import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# TECHNICAL INDICATOR CALCULATION FUNCTIONS
# ------------------------------------------------------------------------------


def calculate_accumulation_distribution_indicator(close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Indicator (ADL or ADO) for a given security.[volume indicator]
    https://corporatefinanceinstitute.com/resources/equities/accumulation-distribution-indicator-a-d/#:~:text=The%20accumulation%20distribution%20indicator%20(AD,stock's%20price%20and%20volume%20flow.

    Parameters:
    - close_prices: Close prices series (pandas Series)
    - high_prices: High prices series (pandas Series)
    - low_prices: Low prices series (pandas Series)
    - volume: Volume series (pandas Series)

    Returns:
    - adl: Accumulation/Distribution Indicator series (pandas Series)
    """
    money_flow_multiplier = ((close_prices - low_prices) - (high_prices - close_prices)) / (high_prices - low_prices)
    money_flow_volume = money_flow_multiplier * volume
    adl = money_flow_volume.cumsum()
    return adl

def calculate_atr(open_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Average True Range (ATR) for a given security.[Volatility indicator]
    https://www.investopedia.com/terms/a/atr.asp

    Parameters:
    - open_prices: Open prices series (pandas Series)
    - high_prices: High prices series (pandas Series)
    - low_prices: Low prices series (pandas Series)
    - window: Size of the moving average window

    Returns:
    - atr: Average True Range series (pandas Series)
    """
    daily_variation = (high_prices - low_prices) / open_prices
    return daily_variation.rolling(window=window).mean()

def calculate_bollinger_bands(close_prices: pd.Series, window: int, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Bollinger Bands for a given set of closing prices.
    https://www.investopedia.com/terms/b/bollingerbands.asp

    Parameters:
    - close_prices: Close prices series (pandas Series)
    - window: Size of the moving average window
    - num_std: Number of standard deviations for the bands (default is 2)

    Returns:
    - bollinger_upper: Upper Bollinger Band values (pandas Series)
    - sma: Simple Moving Average values (pandas Series)
    - bollinger_lower: Lower Bollinger Band values (pandas Series)
    """
    sma = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()

    bollinger_upper = sma + (rolling_std * num_std)
    bollinger_lower = sma - (rolling_std * num_std)

    return np.round(bollinger_upper, 2), np.round(sma, 2), np.round(bollinger_lower, 2)

def calculate_bollinger_width(bollinger_upper: pd.Series, sma: pd.Series, bollinger_lower: pd.Series) -> pd.Series:
    """
    Calculate Bollinger Band Width.[Volatility indicator]
    https://www.investopedia.com/terms/b/bollingerbands.asp

    Parameters:
    - bollinger_upper: Upper Bollinger Band values (pandas Series)
    - sma: Simple Moving Average values (pandas Series)
    - bollinger_lower: Lower Bollinger Band values (pandas Series)

    Returns:
    - bollinger_width: Bollinger Band Width values (pandas Series)
    """
    return (bollinger_upper - bollinger_lower) / sma

def calculate_ema(close_prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).[trend indicator]
    https://www.investopedia.com/terms/e/ema.asp

    Parameters:
    - close_prices: Close prices series (pandas Series)
    - window: Size of the EMA window

    Returns:
    - ema: Exponential Moving Average values (pandas Series)
    """
    return close_prices.ewm(span=window, adjust=False).mean()

def calculate_macd(close_prices: pd.Series, short_window: int, long_window: int, signal_window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).[momentum | trend indicator]
    https://www.investopedia.com/terms/m/macd.asp

    Parameters:
    - close_prices: Close prices series (pandas Series)
    - short_window: Short-term EMA window
    - long_window: Long-term EMA window
    - signal_window: Signal line EMA window

    Returns:
    - macd_histogram: MACD histogram values (pandas Series)
    - macd: MACD line values (pandas Series)
    - macd_signal: MACD signal line values (pandas Series)
    """
    short_ema = close_prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd_histogram, macd, macd_signal

def calculate_dmi(high_prices: pd.Series, low_prices: pd.Series, open_prices: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Directional Movement Index (DMI).[trend indicator]
    https://www.investopedia.com/terms/d/dmi.asp

    Parameters:
    - high_prices: High prices series (pandas Series)
    - low_prices: Low prices series (pandas Series)
    - open_prices: Open prices series (pandas Series)

    Returns:
    - di_plus: Positive Directional Indicator values (pandas Series)
    - di_minus: Negative Directional Indicator values (pandas Series)
    - adx: Average Directional Movement Index values (pandas Series)
    """
    high_shifted = high_prices.shift(1)
    low_shifted = low_prices.shift(1)
    dm_plus = high_prices - high_shifted
    dm_minus = low_shifted - low_prices
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    atr = calculate_atr(open_prices, high_prices, low_prices, window=window)
    di_plus = 100 * calculate_ema(dm_plus, window=window) / atr
    di_minus = 100 * (calculate_ema(dm_minus, window=window) / atr)
    adx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    return di_plus, di_minus, adx

def calculate_stoch(close_prices: pd.Series, low_prices: pd.Series, high_prices: pd.Series, n_fast_k: int = 14, n_fast_d: int = 3, n_slow_d: int = 3) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (STOCH).[momentum oscillator]
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    Some sources suggest it is less volatile than Williams %R and may provide more reliable signals
    Parameters:
    - close_prices: Close prices series (pandas Series)
    - low_prices: Low prices series (pandas Series)
    - high_prices: High prices series (pandas Series)
    - n_fast_k: Fast %K period (default is 14)
    - n_fast_d: Fast %D period (default is 3)
    - n_slow_d: Slow %D period (default is 3)

    Returns:
    - k: Fast %K values (pandas Series)
    - d_fast: Fast %D values (pandas Series)
    - d_slow: Slow %D values (pandas Series)
    """
    l14 = low_prices.rolling(window=n_fast_k).min()
    h14 = high_prices.rolling(window=n_fast_k).max()
    k = 100 * (close_prices - l14) / (h14 - l14)
    d_fast = k.rolling(window=n_fast_d).mean()
    d_slow = d_fast.rolling(window=n_slow_d).mean()
    return k, d_fast, d_slow

def calculate_rsi(close_prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).[momentum oscillator]
    https://www.investopedia.com/terms/r/rsi.asp

    Parameters:
    - close_prices: Close prices series (pandas Series)
    - window: RSI period (default is 14)

    Returns:
    - rsi: Relative Strength Index values (pandas Series)
    """
    delta = close_prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, min_periods=window).mean()
    ema_down = down.ewm(com=window-1, min_periods=window).mean()
    return (100 * ema_up / (ema_down + ema_up))

def calculate_williams(close_prices: pd.Series, low_prices: pd.Series, high_prices: pd.Series, window: int ) -> pd.Series:
    """
    Calculate Williams %R.[momentum oscillator]
    https://www.investopedia.com/terms/w/williamsr.asp
    Some analysis suggest it is more volatile and prone to giving false signals than the Stochastic %K.
    Parameters:
    - close_prices: Close prices series (pandas Series)
    - low_prices: Low prices series (pandas Series)
    - high_prices: High prices series (pandas Series)
    - window: Williams %R period (default is 14)

    Returns:
    - williams: Williams %R values (pandas Series)
    """
    highest_high = high_prices.rolling(window).max()
    lowest_low = low_prices.rolling(window).min()
    williams = ((highest_high - close_prices) / (highest_high - lowest_low)) * -100
    return williams


def calculate_obv(close_prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates On-Balance Volume (OBV). [volume indicator]
    https://www.investopedia.com/terms/o/onbalancevolume.asp

    OBV measures buying and selling pressure. Divergence between price and OBV
    is a powerful predictor of potential trend reversals.
    """
    obv = (np.sign(close_prices.diff()) * volume).fillna(0).cumsum()
    return obv.rename("obv")

def calculate_chaikin_money_flow(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculates Chaikin Money Flow (CMF).
    https://www.investopedia.com/terms/c/chaikinmoneyflow.asp

    CMF measures money flow volume. Values > 0 suggest buying pressure, while
    values < 0 suggest selling pressure. It confirms trend strength.
    """
    mfv = ((close_prices - low_prices) - (high_prices - close_prices)) / (high_prices - low_prices).replace(0, np.nan) * volume
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return cmf.rename("cmf")

class Features: # should be a father class if any other class inherits from it
    def __init__(self, df: pd.DataFrame, window: int = 14, macd_short: int = 12, macd_long: int = 26, macd_signal: int = 9):
        self.df = df.copy()
        self.window = window
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal = macd_signal
        self._calculate_base_indicators()

    @property
    def feature_columns(self) -> list[str]:
        """Returns the list of custom feature column names."""
        return [
            'macd_crossover_signal',
            'stoch_crossover_signal',
            'rsi_relative_to_bb',
            'normalized_volume',
            'price_vs_rsi_divergence',
            'trend_strength_and_momentum',
            'directional_momentum',
            'bb_position'
        ]

    def _calculate_base_indicators(self):
        """Calculates all the necessary base indicators required for the feature engineering."""
        # MACD
        _, macd, macd_signal = calculate_macd(self.df['close'], self.macd_short, self.macd_long, self.macd_signal)
        self.df['macd'] = macd
        self.df['macd_signal'] = macd_signal

        # Stochastic Oscillator
        k, d_fast, _ = calculate_stoch(self.df['close'], self.df['low'], self.df['high'], n_fast_k=self.window)
        self.df['stoch_k'] = k
        self.df['stoch_d_fast'] = d_fast

        # RSI
        self.df['rsi'] = calculate_rsi(self.df['close'], window=self.window)

        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(self.df['close'], window=self.window)
        self.df['bb_upper'] = upper
        self.df['bb_lower'] = lower
        self.df['bb_width'] = calculate_bollinger_width(upper, middle, lower)

        # DMI / ADX
        di_plus, di_minus, adx = calculate_dmi(self.df['high'], self.df['low'], self.df['open'], window=self.window)
        self.df['di_plus'] = di_plus
        self.df['di_minus'] = di_minus
        self.df['adx'] = adx

    def add_all(self):
        """Adds all custom features to the DataFrame."""
        self.df['macd_crossover_signal'] = self.df['macd'] - self.df['macd_signal']
        self.df['stoch_crossover_signal'] = self.df['stoch_k'] - self.df['stoch_d_fast']
        self.df['rsi_relative_to_bb'] = (self.df['rsi'] - 50) / self.df['bb_width']
        self.df['normalized_volume'] = self.df['volume'] / calculate_ema(self.df['volume'], window=self.window)

        rolling_max_close = self.df['close'].rolling(window=self.window).max()
        rolling_max_rsi = self.df['rsi'].rolling(window=self.window).max()
        self.df['price_vs_rsi_divergence'] = rolling_max_close - rolling_max_rsi

        self.df['trend_strength_and_momentum'] = (self.df['adx'] / 100) * (self.df['rsi'] - 50)
        self.df['directional_momentum'] = self.df['di_plus'] - self.df['di_minus']
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])

        return self.df