
#Files from src
from Functions import features_paiplot, count_outliers_iqr_df, technical_indicators, feature_transform, correlation, rolling_correlation, check_distribution, is_normal
from DataLoading import df_dict_data, df_training_dict, df_validation_dict


#access the dictionary from DataLoading
df_dict_data = df_dict_data.copy()
df_training_dict = df_training_dict.copy()
df_validation_dict = df_validation_dict.copy()

def create_features(df_training_set:dict):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        value = technical_indicators(value.copy())
        value.iloc[:,6:] = feature_transform(value.iloc[:,6:].copy())
        value = value.dropna()
        df_training_set[key] = value
    return df_training_dict

def correlations(df_training_set:dict, column_name='Adj Close'):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        correlation(value, column_name, key)

def rolling_correlations(df_training_set:dict, wf_ratio=3, wf_splits=60, column_name='Adj Close'):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        rolling_correlation(value, column_name, key, wf_ratio, wf_splits)        

def distribution(df_training_set:dict, column_name='Adj Close'):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
            
            if key == 'N225 - Japan':  # Nikkei 225
                currency = 'JPY'
            elif key == 'Bovespa - Brazil':  # Bovespa Index
                currency = 'BRL'
            elif key == 'DAX - Germany': # DAx
                currency = 'EUR'
            else:
                currency = 'USD'  # Default to USD if unknown index
    check_distribution(value, key, currency, column_name)
    

def pair_plot(df_training_set):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        features_paiplot(value, key)

def check_normality_outliers(df_training_set):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        print(f'Is data normal? Index{key} {is_normal(value)}')
        print(f'Index {key} Number of outliers in each feature: {count_outliers_iqr_df(value)}')

def choose_columns(df_training_set, chosen_features):
    # Loop through each dataframe and its corresponding validation set in the dictionary
    for key, value in df_training_set.items():
        value_f = value.copy() 
        value_f = value_f.loc[:, chosen_features]
        df_training_set[key] = value_f
    return df_training_dict

if __name__ == "__main__":
    #check distribution
    distribution(df_training_dict)

    #create featuers and transform
    df_training_dict = create_features(df_training_dict)

    #print correlations & pairplot
    correlations(df_training_dict)
    rolling_correlations(df_training_dict)

    #choosing features:
    chosen_features = ['WMA', 'MACD', 'RSI', '%D_fast', '%D_slow', 'Bollinger Diff', 
                        'WPR', 'OBV', 'ROC', 'ATR', 'MFI', 'Chaikin_Oscillator', 'Adj Close']
    df_training_dict = choose_columns(df_training_dict, chosen_features)

    #print correlations & pairplot
    correlations(df_training_dict)
    pair_plot(df_training_dict)

    #check outliers & Normality
    check_normality_outliers(df_training_dict)

#choosing features:
chosen_features = ['WMA', 'MACD', 'RSI', '%D_fast', '%D_slow', 'Bollinger Diff', 
                    'WPR', 'OBV', 'ROC', 'ATR', 'MFI', 'Chaikin_Oscillator', 'Adj Close']