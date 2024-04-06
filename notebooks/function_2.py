# %%
import os
import glob
import numpy as np
import pandas as pd
import requests
import time
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error



# for visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import seaborn as sns


# %%
spx_test = r"C:\Users\Nagham\Investor\Data\Test\spx_d_test2.csv"




def read_csv_file(file_path: str, delimiter: str = '\t') -> pd.DataFrame:
    """
    Read a TXT file and convert it to tabular data.

    Parameters:
        file_path (str): The path to the TXT file.
        delimiter (str): The delimiter used in the TXT file. Default is '\t' (tab).

    Returns:
        pandas.DataFrame: The tabular data.
    """
    try:
        # Read the TXT file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None
    
    
df_spx_t = read_csv_file(spx_test, ",")# Extract Test data

df_spx_t.info()

# %%
# Renaming the columns by removing the '<' and '>' characters
new_column_names = {col: col.strip('<>').upper() for col in df_spx_t.columns}
df_spx_t = df_spx_t.rename(columns=new_column_names)

# %%
df_t = df_spx_t.copy()

# %%
# Calculate discrete returns
df_t['discrete_return'] = (df_t['CLOSE'] - df_t['CLOSE'].shift(1)) / df_t['CLOSE'].shift(1)
#df['discrete_return'] = np.log(df['OPEN']/df['OPEN'].shift(1)) # opposed to closing prices, to avoid look-ahead bias.

print(df_t.describe())
df_t.head()

# %%
#Weighted Moving Average
def calculate_wma(data, window):
    weights = np.arange(1, window + 1)
    wma = data.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

# %%
# Define the window size for WMA calculation
window_size = 15

# Calculate WMA with the specified window size
df_t['WMA'] = calculate_wma(df_t['CLOSE'], window_size)

df_t['WMA_signal'] = df_t['CLOSE'] - df_t['WMA']

# %%
#Need to validate this code 
#Relative Strength Index (RSI)
window_size = 14

# Calculate price changes
Price_Change = df_t['CLOSE'].diff()

# Calculate gains and losses
Gain = np.where(Price_Change > 0, Price_Change, 0)
Loss = np.where(Price_Change < 0, abs(Price_Change), 0)

# Calculate average gain and average loss over the period
Avg_Gain = pd.Series(Gain).rolling(window=window_size, min_periods=1).mean()
Avg_Loss = pd.Series(Loss).rolling(window=window_size, min_periods=1).mean()

# Calculate Relative Strength (RS)
RS = Avg_Gain / Avg_Loss

# Calculate RSI
df_t['RSI'] = 100 - (100 / (1 + RS))

# %%
# Define period for WPR calculation
window_size = 14

# Calculate highest high and lowest low over the period
Highest_High = df_t['HIGH'].rolling(window=window_size).max()
Lowest_Low = df_t['LOW'].rolling(window=window_size).min()

# Calculate Williams %R
df_t['WPR'] = (Highest_High - df_t['CLOSE']) / (Highest_High - Lowest_Low) * -100


# %%
def calculate_bollinger_bands(df, window=20, num_std_dev=2): #20,2 Typiclly used 
    # Calculate the rolling mean and standard deviation
    rolling_mean = df['Typical Price'].rolling(window=window).mean()
    rolling_std = df['Typical Price'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    
    return upper_band, lower_band

# %%
# Create a new column for the closing price
df_t['Typical Price'] = (df_t['LOW'] + df_t['HIGH'] + df_t['CLOSE']) / 3.0

# Calculate Bollinger Bands
upper_band, lower_band = calculate_bollinger_bands(df_t)

# Add the diff to the DataFrame
df_t['Bollinger Diff'] = upper_band - lower_band

# %%
#Moving Average Convergence Divergence (MACD)
# Define periods for short-term and long-term EMAs
short_period = 12
long_period = 26
signal_line_span = 9

# Calculate short-term EMA
short_ema = df_t['CLOSE'].ewm(span=short_period, adjust=False).mean()

# Calculate long-term EMA
long_ema = df_t['CLOSE'].ewm(span=long_period, adjust=False).mean()

# Calculate MACD line
macd_line = short_ema - long_ema

# Calculate Signal line (typically 9-period EMA of MACD line)
signal_line = macd_line.ewm(span=signal_line_span, adjust=False).mean()

# Calculate MACD signal
df_t['macd_signal'] = macd_line - signal_line

# %%
# Transformation Function
# Technical analysis indicators need to be rescaled before being fed to the models.
# The process is conducted using a version of min-max normalization technique which produces outputs in range from ‐1 to 1.
# This technique was chosen for two reasons: it is intuitive as the machine learning models produce output 
# variable that is also ranging from ‐1 to 1 and because it causes the input data to be more comparable. 
# X'(t) = (X(t) - min(x)) / (max(x) - min(x))*2 -1

def feature_transform(x):
    max_x = np.max(x)
    min_x = np.min(x)

    x_transformed = (x - min_x)/(max_x - min_x)*2 -1

    return x_transformed

# %%
df_t = df_t.drop(df_t.index[0:20])

# %%
#feature transform
df_t.iloc[:, 7:] = df_t.iloc[:, 7:].apply(feature_transform)


# %%
df_t.drop(df_t.columns[1:6], axis=1, inplace=True)

# %%
df_t.drop('Typical Price', axis=1, inplace=True)

# %%
df_t.drop('WMA', axis=1, inplace=True)

# %%
df_t

# %%
def get_test_data():
    return (df_t)


