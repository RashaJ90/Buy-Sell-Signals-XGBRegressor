import numpy as np
import pandas as pd

# For Data Distribution
from scipy import stats

# For visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


#Simple Moving Average:
#returns the dataframe with additional coumn of simple moving average
def calculate_sma(df: pd.DataFrame, column: str = 'Adj Close', window_size: int = 15) -> pd.DataFrame:
    """
    Calculate the Simple Moving Average (SMA) for a given column in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the financial data.
        column (str): Name of the column for which to calculate SMA. Default is 'close'.
        window_size (int): Size of the moving window. Default is 15.

    Returns:
        pd.DataFrame: DataFrame with SMA column added.
    """
    # Calculate SMA
    sma = df[column].rolling(window=window_size, min_periods=1).mean()
    
    # Create a DataFrame to store SMA
    df['SMA'] = sma
    return df

#Weighted Moving Average
def calculate_wma(df: pd.DataFrame, column: str = 'Adj Close', window_size: int = 15) -> pd.DataFrame:
    """
    Calculate the Weighted Moving Average (WMA) for a given column in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the financial data.
        column (str): Name of the column for which to calculate WMA. Default is 'close'.
        window_size (int): Size of the moving window. Default is 15.

    Returns:
        pd.DataFrame: DataFrame with WMA and WMA signal columns added.
    """
    # Generate the weights
    weights = np.arange(1, window_size + 1)
    data = df[column]
    
    # Calculate the WMA using convolution
    wma = data.rolling(window=window_size).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    # Create a DataFrame to store WMA
    df['WMA'] = wma
    
    # Add WMA signal column
    df['WMA_signal'] = df[column] - wma

    
    return df

#MACD
def calculate_macd(df: pd.DataFrame, short_window:int=12, long_window:int=26, signal_window:int=9, column: str = 'Adj Close') -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        short_window (int): The short-term window size for the short EMA.
        long_window (int): The long-term window size for the long EMA.
        signal_window (int): The window size for the signal line EMA.

    Returns:
        DataFrame: DataFrame with additional columns for MACD and signal line.
    """
    # Calculate short-term EMA
    short_ema = df[column].ewm(span=short_window, min_periods=1, adjust=False).mean()
    
    # Calculate long-term EMA
    long_ema = df[column].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = short_ema - long_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    # Store MACD and signal line in the DataFrame
    df['MACD'] = macd_line
    df['Signal Line'] = signal_line
    df['macd_signal'] = macd_line - signal_line
    return df

#Stochastic_oscillator
def calculate_stochastic_oscillator(df, k_fast_period=14, k_slow_period=3, d_slow_period=3, column: str = 'Adj Close'):
    """
    Calculate the Stochastic Oscillator and its corresponding moving averages (K and D lines).

    Parameters:
        df (DataFrame): DataFrame containing the data.
        k_fast_period (int): The period for the fast %K line.
        k_slow_period (int): The period for the slow %K line.
        d_slow_period (int): The period for the slow %D line.

    Returns:
        DataFrame: DataFrame with additional columns for %K_fast, %K_slow, %D_fast, and %D_slow.
    """
    # Calculate highest high and lowest low over the period
    HH = df['High'].rolling(window=k_fast_period).max()
    LL = df['Low'].rolling(window=k_fast_period).min()

    # Calculate %K_fast
    df['%K_fast'] = ((df[column] - LL) / 
                     (HH - LL)) * 100
    
    # Calculate %K_slow (smoothed %K_fast)
    df['%K_slow'] = df['%K_fast'].rolling(window=k_slow_period).mean()
    
    # Calculate %D_fast (3-day SMA of %K_slow)
    df['%D_fast'] = df['%K_slow'].rolling(window=d_slow_period).mean()
    
    # Calculate %D_slow (3-day SMA of %D_fast)
    df['%D_slow'] = df['%D_fast'].rolling(window=d_slow_period).mean()
    
    return df

#RSI
def calculate_rsi(df, window_size=14, column: str = 'Adj Close'):
    """
    Calculate the Relative Strength Index (RSI) for a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        window (int): The window size for calculating RSI.

    Returns:
        DataFrame: DataFrame with an additional column for RSI.
    """
    # Calculate price changes
    delta = df[column].diff()
    
    # Define up and down moves
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    
    # Calculate the relative strength (RS)
    rs = gain / loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Store RSI in the DataFrame
    df['RSI'] = rsi
    
    return df

#WPR
def calculate_williams_percent_r(df, window=14, column: str = 'Adj Close'):
    """
    Calculate the Williams %R (WPR) for a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        window (int): The window size for calculating WPR.

    Returns:
        DataFrame: DataFrame with an additional column for WPR.
    """
    # Calculate highest high and lowest low over the window
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    
    # Calculate Williams %R
    wpr = ((highest_high - df[column]) / (highest_high - lowest_low)) * -100
    
    # Store WPR in the DataFrame
    df['WPR'] = wpr
    
    return df

#Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std_dev=2, column: str = 'Adj Close'):
    """
    Calculate Bollinger Bands for a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        window (int): The window size for the moving average.
        num_std_dev (int): The number of standard deviations for the bands.

    Returns:
        DataFrame: DataFrame with additional columns for Bollinger Bands.
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    
    # Store Bollinger Bands in the DataFrame
    df['Bollinger Upper'] = upper_band
    df['Bollinger Lower'] = lower_band
    df['Bollinger Diff'] = upper_band - lower_band
    
    return df

#On-Balance Volume (OBV)
def calculate_obv(df, column: str = 'Adj Close'):
    """
    Calculate On-Balance Volume (OBV) for a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.

    Returns:
        DataFrame: DataFrame with additional column for OBV.
    """
    obv_values = []
    prev_obv = 0.0

    for i in range(1, len(df)):
        if df[column].iloc[i] > df[column].iloc[i - 1]:
            obv = prev_obv + df['Volume'].iloc[i]
        elif df[column].iloc[i] < df[column].iloc[i - 1]:
            obv = prev_obv - df['Volume'].iloc[i]
        else:
            obv = prev_obv

        obv_values.append(obv)
        prev_obv = obv

    # Add initial OBV value as 0
    obv_values = [0] + obv_values

    # Store OBV in the DataFrame
    df['OBV'] = obv_values

    # Convert OBV column to int64
    df['OBV'] = df['OBV']

    return df

#Average True Range (ATR)
def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) of a stock dataset.

    Parameters:
        df (DataFrame): DataFrame containing stock data, with 'High', 'Low', and 'Close' columns representing high, low, and closing prices respectively.
        period (int): Number of periods for which to calculate the ATR (default is 14).

    Returns:
        DataFrame: DataFrame with 'ATR' column containing the calculated ATR values.
    """
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    
    # True Range (TR) calculation
    df['TR'] = df[['High', 'Low', 'Adj Close']].apply(lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Adj Close']), abs(row['Low'] - row['Adj Close'])), axis=1)
    
    # ATR calculation
    df['ATR'] = df['TR'].rolling(period).mean()
    
    # Drop the TR column if not needed
    df.drop('TR', axis=1, inplace=True)
    
    return df

#rice Rate of Change (ROC)
def calculate_roc(df, n_periods=12, column='Adj Close'):
    """
    Calculate the Price Rate of Change (ROC) of a stock dataset.

    Parameters:
        df (DataFrame): DataFrame containing stock data, with 'Adj Close' column representing closing prices.
        n_periods (int): Number of periods for which to calculate the ROC. # It can be anything such as 12, 25,
        or 200. Short-term trader traders typically use a smaller number while longer-term investors use a larger
        number.

    Returns:
        DataFrame: DataFrame with 'ROC' column containing the calculated ROC values.
    """
    close_prices = df[column]
    close_prices_shifted = close_prices.shift(n_periods)
    
    roc = ((close_prices - close_prices_shifted) / close_prices_shifted) * 100
    
    df['ROC'] = roc
    return df

#Money Flow Index - MFI
def calculate_mfi(df, period=14):
    """
    Calculate the Money Flow Index (MFI) of a stock dataset.

    Parameters:
        df (DataFrame): DataFrame containing stock data, with 'High', 'Low', 'Close', and 'Volume' columns representing high, low, closing prices, and volume respectively.
        period (int): Number of periods for which to calculate the MFI (default is 14).

    Returns:
        DataFrame: DataFrame with 'MFI' column containing the calculated MFI values.
    """
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    volume = df['Volume']
    
    # Typical Price calculation
    tp = (high + low + close) / 3
    
    # Raw Money Flow calculation
    mf = tp * volume
    
    # Determine whether the typical price is higher or lower than the previous period
    tp_shifted = tp.shift(1)
    positive_flow = (tp > tp_shifted)
    negative_flow = (tp < tp_shifted)
    
    # Calculate positive and negative money flow
    positive_mf = positive_flow * mf
    negative_mf = negative_flow * mf
    
    # Calculate the Money Flow Ratio (MFR)
    mfr = positive_mf.rolling(window=period).sum() / negative_mf.rolling(window=period).sum()
    
    # Calculate the Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + mfr))
    
    df['MFI'] = mfi
    
    return df

#chaikin_oscillator
def calculate_chaikin_oscillator(df, short_period=3, long_period=10):
    """
    Calculate the Chaikin Oscillator of a stock dataset.

    Parameters:
        df (DataFrame): DataFrame containing stock data, with 'High', 'Low', 'Adj Close', and 'Volume' columns representing high, low, closing prices, and volume respectively.
        short_period (int): Number of periods for the short EMA (default is 3).
        long_period (int): Number of periods for the long EMA (default is 10).

    Returns:
        DataFrame: DataFrame with 'Chaikin_Oscillator' column containing the calculated Chaikin Oscillator values.
    """
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    volume = df['Volume']
    
    # Money Flow Multiplier calculation
    mfm = ((close - low) - (high - close)) / (high - low)
    
    # Money Flow Volume calculation
    mfv = mfm * volume
    
    # Accumulation/Distribution Line (ADL) calculation
    adl = mfv.cumsum()
    
    # Calculate the EMA for ADL
    ema_short = adl.ewm(span=short_period, min_periods=short_period, adjust=False).mean()
    ema_long = adl.ewm(span=long_period, min_periods=long_period, adjust=False).mean()
    
    # Calculate the Chaikin Oscillator
    chaikin_oscillator = ema_short - ema_long
    
    df['Chaikin_Oscillator'] = chaikin_oscillator
    
    return df

#Bulid the technical indicators: features
def technical_indicators(df):
    df = calculate_sma(df)
    df = calculate_wma(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_stochastic_oscillator(df)
    df = calculate_bollinger_bands(df)
    df = calculate_williams_percent_r(df)
    df = calculate_obv(df)
    df = calculate_roc(df)
    df = calculate_atr(df)
    df = calculate_mfi(df)
    df = calculate_chaikin_oscillator(df)
    return df  

# Transformation Function
# Technical analysis indicators need to be rescaled before being fed to the models. [-1,1]
# X'(t) = (X(t) - min(x)) / (max(x) - min(x))*2 -1

def feature_transform(df):
    """
    Transform all columns in the DataFrame as the following formula
    X'(t) = (X(t) - min(x)) / (max(x) - min(x))*2 -1

    Parameters:
        df (DataFrame): DataFrame containing the calculated technical indicators.

    Returns:
        DataFrame: DataFrame with all columns transformed.
    """
    max_x = df.max(skipna=True)  # Compute maximum values excluding NaN
    min_x = df.min(skipna=True)  # Compute minimum values excluding NaN

    df_transformed = (df - min_x) / (max_x - min_x) * 2 - 1

    return df_transformed

#truncate the dataframe from the biggining so the walk forward splits will continue untill the last date
def truncate_before_wf(df, in_sample_size, out_sample_size):
    drop_index = (len(df) - in_sample_size) % out_sample_size
    return (df.iloc[drop_index:, :])

#the function returns a list of tuples, where each tuple contains one in-sample 
#DataFrame and one out-of-sample DataFrame, representing the data splits for each 
#iteration of the walk-forward validation process.

def walk_forward_validation(df, in_sample_size, out_sample_size):
    """
    Perform walk-forward validation on a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        in_sample_size (int): Number of periods to use for in-sample data.
        out_sample_size (int): Number of periods to use for out-of-sample data.

    Returns:
        List: List of tuples that contains the in-sample and out-of-sample data.
    """
    total_rows = len(df)
    n_subsets = (total_rows - in_sample_size) // out_sample_size
    splits = []
        
    for i in range(n_subsets):
        start_index = i * out_sample_size
        end_index = start_index + in_sample_size + out_sample_size
        
        if end_index > total_rows:
            break
        
        in_sample = df.iloc[start_index : start_index + in_sample_size]
        out_of_sample = df.iloc[start_index + in_sample_size : end_index]
        
        splits.append((in_sample, out_of_sample))
    return (splits)

# Compute the correlation coefficients between each feature and the return & print it

def correlation(df, target_name, key):

    correlation_with_target = np.abs(df.corrwith(df[target_name]))

    # Display the correlation coefficients
    print(f'Index {key} - Correlation with {target_name}:')
    print(correlation_with_target.sort_values(ascending=False))
    correlation_with_target.sort_values().plot.barh(color = 'blue',title = f'Index {key} Strength of Correlation')
    plt.show()

# Compute the rolling correlation for each pair of selected features
def rolling_correlation(df, target_name, key, wf_ratio, wf_splits):
    window_size = len(df) // (wf_splits + wf_ratio)
    correlation_with_target = df.rolling(window=window_size).corr(df[target_name])
    # Create traces for each feature
    traces = []
    for feature in df.columns:
        trace = go.Scatter(
            x=correlation_with_target.index,
            y=correlation_with_target[feature],
            mode='lines',
            name=feature
        )
        traces.append(trace)

    # Create layout for the plot
        layout = go.Layout(
        title=f'Index {key} - Rolling Correlation of Features with {target_name}',
        xaxis=dict(title='Index'),
        yaxis=dict(title=f'Rolling Correlation with {target_name}'),
        hovermode='closest',
        autosize=True
    )

    # Create figure object
    fig = go.Figure(data=traces, layout=layout)

    # Show plot
    fig.show()


#Pair plot
def features_paiplot(df, key):
    pairplot = sns.pairplot(df, height=1.5)

    # Set the title
    pairplot.figure.suptitle(f'Index {key} - Pairplot of features', y=1.02)

    # Show the plot
    plt.show()

#count out liers
def count_outliers_iqr_df(df, k=1.5):
    """
    Count the number of outliers in each column of the DataFrame using the Interquartile Range (IQR) method.
    
    Parameters:
    - df: The input DataFrame.
    - k: The multiplier for the IQR. Typically set to 1.5 to 3.
    
    Returns:
    - A dictionary where keys are column names and values are the number of outliers detected in each column.
    """
    outliers_counts = {}
    for col in df.columns:
        data = df[col]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (k * iqr)
        upper_bound = quartile_3 + (k * iqr)
        outliers = (data < lower_bound) | (data > upper_bound)
        outliers_counts[col] = np.sum(outliers)
    return outliers_counts

#Truncate NaN Data
def drop_nan(df):
    # Remove rows with NaN values
    cleaned_df = df.dropna()

    return cleaned_df
#check distribution 
def check_distribution(df, key, currency, column_name='Adj Close'):
    """
    Check the distribution of a column in a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to check the distribution for (default is 'Adj Close').

    Returns:
        None (displays descriptive statistics and visualizations)
    """
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(df[column_name].describe())

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True)
    plt.title(f'Index {key} - Distribution of {column_name}')
    plt.xlabel(f'{column_name} [{currency}]')
    plt.ylabel('Frequency')
    plt.show()

def is_normal(df, alpha=0.05):
    """
    Test if the data is normally distributed using Z-score.
    
    Parameters:
    - data: The input data array.
    - alpha: The significance level for the test.
    
    Returns:
    - True if the data is normally distributed, False otherwise.
    """
    normal_col = {}
    for col in df.columns:
        data = df[col]
        z_score, p_value = stats.normaltest(data)
        normal_col[col] = p_value > alpha
    return normal_col
# Models Evaluatuin
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0, periods_per_year=252):
    """
    Calculate the Sharpe Ratio of an investment.

    Parameters:
        daily_returns (pd.Series or np.array): Daily returns of the investment.
        risk_free_rate (float): Annual risk-free rate of return (default is 0).
        periods_per_year (int): Number of trading days in a year (default is 252).

    Returns:
        float: Sharpe Ratio of the investment.
    """
    # Calculate excess return (mean return - risk-free rate)
    excess_return = np.mean(daily_returns) - risk_free_rate

    # Calculate standard deviation of returns
    std_dev = np.std(daily_returns)

    # Calculate Sharpe Ratio
    sharpe_ratio = (excess_return / std_dev) * np.sqrt(periods_per_year)

    return sharpe_ratio

#win/loss ratio
def calculate_win_loss_ratio(true_returns, predicted_returns):
    """
    Calculate the Win/Loss Ratio of a trading strategy.

    Parameters:
        true_returns (pd.Series or np.array): True daily returns of the investment.
        predicted_returns (pd.Series or np.array): Predicted daily returns of the investment.

    Returns:
        float: Win/Loss Ratio of the trading strategy.
    """
    # Calculate the difference between predicted and true returns
    differences = predicted_returns - true_returns

    # Count the number of positive and negative differences
    num_positive = (differences > 0).sum()
    num_negative = (differences < 0).sum()

    # Calculate the Win/Loss Ratio
    if num_negative == 0:
        win_loss_ratio = float('inf')  # Avoid division by zero
    else:
        win_loss_ratio = num_positive / num_negative

    return win_loss_ratio

def calculate_mdd(returns):
    """
    Calculate the Maximum Drawdown (MDD) of a time series of returns.

    Parameters:
    - returns: Array-like object containing historical returns.

    Returns:
    - max_drawdownfloat: Maximum Drawdown of the investment as a percentage.

    """
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate the maximum value seen up to each point
    max_seen = cum_returns.cummax()
    
    # Calculate drawdowns
    drawdowns = (cum_returns - max_seen) / max_seen
    
    # Find the maximum drawdown
    max_drawdown = drawdowns.min()

    # Convert to percentage
    max_drawdown_percentage = max_drawdown * 100
    
    return max_drawdown_percentage

def calculate_accuracy(true_series, predicted_series):
    """
    Calculate the accuracy of a model that predicts buy, hold, or sell signals.

    Parameters:
    true_series (Series): Pandas Series containing true labels.
    predicted_series (Series): Pandas Series containing predicted labels.

    Returns:
    accuracy (float): Accuracy of the model.
    """
    if len(true_series) != len(predicted_series):
        raise ValueError("The lengths of true_series and predicted_series must be equal.")

    correct_predictions = sum(1 for true_label, predicted_label in zip(true_series,
                             predicted_series) if true_label == predicted_label)
    total_predictions = len(true_series)
    accuracy = correct_predictions / total_predictions

    return accuracy

#Results Proccessing
def calculate_signal(df, return_column='Predicted Returns', signal_column='Predicted Signal', q1=None, q3=None):
    """
    Calculate trading signals based on specified quantiles of returns in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - return_column: Name of the column containing returns (default is 'Predicted Returns').
    - signal_column: Name of the column to store the calculated signals (default is 'Predicted Signal').
    - q1: The percentile to calculate for the first quartile (default is None, calculated as 0.25 quantile if not provided).
    - q3: The percentile to calculate for the third quartile (default is None, calculated as 0.75 quantile if not provided).

    Returns:
    - df: DataFrame with the added signal_column containing the calculated signals.
    """
    if q1 is None or q3 is None:
        q1 = df[return_column].quantile(0.25)
        q3 = df[return_column].quantile(0.75)

    df[signal_column] = 0  # Default signal

    df.loc[df[return_column] >= q3, signal_column] = 1
    df.loc[df[return_column] <= q1, signal_column] = -1

    return df

def calculate_returns(df, column_to_diff='Predictions', column='Predicted Returns'):
    """
    Calculate returns from adjusted close prices.

    Parameters:
    df (DataFrame): DataFrame containing adjusted close prices.

    Returns:
    returns (DataFrame): DataFrame containing the calculated returns in df[column].
    """
    
    df[column] = df[column_to_diff].pct_change()
    df = df.dropna()

    return df

def calculate_quantiles(df, column='Predected Return', q1=0.25, q2=0.5, q3=0.75):
    """
    Calculate specified quantiles (percentiles) of a column in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - column: Name of the column for which quantiles are to be calculated (default is 'Predicted Return').
    - q1: The percentile to calculate for the first quartile (default is 0.25, corresponding to the 25th percentile).
    - q2: The percentile to calculate for the second quartile (default is 0.5, corresponding to the median).
    - q3: The percentile to calculate for the third quartile (default is 0.75, corresponding to the 75th percentile).

    Returns:
    - quantiles: A tuple containing the specified quantiles (Q1, Q2, Q3).
    """
    
    Q1 = df[column].quantile(q1)
    Q2 = df[column].quantile(q2)  # Median
    Q3 = df[column].quantile(q3)

    return (Q1, Q2, Q3)

# Features
def features_creation(df, col_names:list, y_col_name='Adj Close',):
    df = calculate_sma(df.copy())
    df = calculate_wma(df.copy())
    df = calculate_macd(df.copy())
    df = calculate_rsi(df.copy())
    df = calculate_stochastic_oscillator(df.copy())
    df = calculate_bollinger_bands(df.copy())
    df = calculate_williams_percent_r(df.copy())
    df = calculate_obv(df.copy())
    df = calculate_roc(df.copy())
    df = calculate_atr(df.copy())
    df = calculate_mfi(df.copy())
    df = calculate_chaikin_oscillator(df.copy())

    #drop NA data
    df = df.dropna()

    #Choose the specific features
    df = df[col_names]

    #transform features [-1,1]
    dfX = df.copy()
    dfX = dfX.drop(columns=y_col_name)
    dfX = feature_transform(dfX)
    df = pd.concat([dfX, df[y_col_name]],axis= 1)
    return df

def apply_feature_creation_to_splits(splits, col_names:list, y_col_name='Adj Close'):
    """
    Apply feature creation function to each split and return a list of tuples containing processed 
    in-sample DataFrame along with the corresponding out-of-sample DataFrame.

    Parameters:
        splits (list): List of tuples containing in-sample and out-of-sample DataFrames.
        col_names (list): List of column names to select from the DataFrame.
        y_col_name (str): Name of the target column.

    Returns:
        list: List of tuples containing processed in-sample DataFrame and corresponding out-of-sample 
        DataFrame.
    """
    processed_splits = []  # List to store processed splits
    
    # Iterate over each split
    for split in splits:
        in_sample, out_of_sample = split  # Unpack the split
        
        # Apply features_creation to the in-sample data
        in_sample_features = features_creation(in_sample, col_names=col_names, y_col_name=y_col_name)
        
        # Apply features_creation to the in-sample data
        out_of_sample_features = features_creation(out_of_sample, col_names=col_names, y_col_name=y_col_name)
        
        # Append the processed in-sample DataFrame along with the original out-of-sample DataFrame to the list
        processed_splits.append((in_sample_features, out_of_sample_features))
    
    return processed_splits