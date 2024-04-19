
# For Data
import yfinance as yf


# Fetch historical data for the indecis
#Training Sets
sp500_data = yf.download('^GSPC', start='2002-01-01', end='2023-01-01')
dax_data = yf.download('^GDAXI', start='2002-01-01', end='2023-01-01')
nikkei_data = yf.download('^N225', start='2002-01-01', end='2023-01-01')
bovespa_data = yf.download('^BVSP', start='2002-01-01', end='2023-01-01')

#Validation Sets
sp500_validation_set = yf.download('^GSPC', start='2023-01-02', end='2024-04-18')
dax_validation_set = yf.download('^GDAXI', start='2023-01-02', end='2024-04-18')
nikkei_validation_set = yf.download('^N225', start='2023-01-02', end='2024-04-18')
bovespa_validation_set = yf.download('^BVSP', start='2023-01-02', end='2024-04-18')

def check_nan_values(index, data):
    """
    Check for NaN values in the given DataFrame.

    Parameters:
        data (DataFrame): The DataFrame to check for NaN values.

    Returns:
        DataFrame: A DataFrame indicating whether each column contains NaN values.
    """
    nan_values_df = data.isna().any()
    print(f'NaN values in {index}:')
    print(nan_values_df)
    return nan_values_df

#Create a dictionary of the imported dfs

df_dict_data = {
    'S&P 500 - USA': {'training_set': sp500_data, 'validation_set': sp500_validation_set},
    'DAX - Germany': {'training_set': dax_data, 'validation_set': dax_validation_set},
    'N225 - Japan': {'training_set': nikkei_data, 'validation_set': nikkei_validation_set},
    'Bovespa - Brazil': {'training_set': bovespa_data, 'validation_set': bovespa_validation_set}
}

df_training_dict = {
    'S&P 500 - USA': sp500_data,
    'DAX - Germany': dax_data,
    'N225 - Japan': nikkei_data,
    'Bovespa - Brazil': bovespa_data
}

df_validation_dict = {
    'S&P 500 - USA': sp500_validation_set,
    'DAX - Germany':  dax_validation_set,
    'N225 - Japan':  nikkei_validation_set,
    'Bovespa - Brazil':  bovespa_validation_set
}

# If this file is run directly, execute some additional code
if __name__ == "__main__":
    # Loop through each dataframe and its corresponding validation set in the dictionary
    def nan_values_in_data (df_dict_data):
        for key, value in df_dict_data.items():
            print(f"Index: {key}")
            print("training data:")
            check_nan_values(key, value['training_set'])
            print()
            print("Validation data:")
            check_nan_values(key, value['validation_set'])
            print()