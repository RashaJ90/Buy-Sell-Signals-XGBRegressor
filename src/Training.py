import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta

#for models

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from pmdarima.arima import auto_arima

#for gridsearch
from sklearn.model_selection import GridSearchCV



#for Data
import yfinance as yf

#for Data Distribution
from scipy import stats

# for visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import seaborn as sns

from Models import TimeSeriesModels, Best_Params

#print optimization results:
for model_type, model_dict in Best_Params.items():
    print(f"{model_type}:")
    for key, model_df in model_dict.items():
        print(f"  {key}:")
        print(model_df)

        

Model = TimeSeriesModels(svr_kernel='poly', C=100, gamma='scale', degree=4)

# Initialize an empty list to store predictions & adjust the df index (should cut the first unpredected 200)
predictions = Model.svr_train(splits_f, 'Adj Close')

#Split the data
out_sample_size = len(sp500_d) // (best_ratio + best_splits_num)
in_sample_size = best_ratio * out_sample_size

sp500_d = truncate_before_wf(sp500_d, in_sample_size, out_sample_size)
splits = walk_forward_validation(sp500_d, in_sample_size, out_sample_size)

# Adjust the predected df index (should cut the first unpredected 1300)
index_dropped = best_ratio * out_sample_size #first idicies that are participating in the insample bur not predicted
index_predict = out_sample_size #number of out of samples in each split

#create features
splits_f = apply_feature_creation_to_splits(splits, chosen_features)

sp500_d_includes_results = sp500_d.copy()
sp500_d_includes_results = sp500_d_includes_results.iloc[index_dropped:, :]
sp500_d_includes_results['Predictions'] = predictions

best_ratio = 3
best_splits_num = 60
nan_window = 19

df_training_dict = df_training_dict.copy()
Best_Params = models_params_optimization(df_training_dict, best_ratio = 3, best_splits_num = 60)

#print optimization results:
for model_type, model_dict in Best_Params.items():
    print(f"{model_type}:")
    for key, model_df in model_dict.items():
        print(f"  {key}:")
        print(model_df)