import numpy as np
import pandas as pd

#for models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

#Files from src
import Functions
from Functions import apply_feature_creation_to_splits, truncate_before_wf, walk_forward_validation
from DataLoading import df_training_dict
from EDA import chosen_features


class TimeSeriesModels:
    def __init__(self, k=5, svr_kernel='rbf', C=1.0, gamma=0.1, degree=3, rf_n_estimators=100, rf_max_features=4, 
                 rf_samples_split=2, rf_samples_leaf=None, rf_max_depth=None, gbm_n_estimators=100, gbm_criterion='squared_error',
                 gbm_loss='squared_error', gbm_n_features='sqrt'):

        self.k = k
        self.svr_kernel = svr_kernel
        self.gamma = gamma
        self.degree = degree
        self.C = C
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_features = rf_max_features
        self.rf_samples_split = rf_samples_split
        self.rf_samples_leaf = rf_samples_leaf
        self.rf_max_depth = rf_max_depth
        self.gbm_n_estimators = gbm_n_estimators
        self.gbm_criterion = gbm_criterion
        self.gbm_loss = gbm_loss
        self.gbm_n_features = gbm_n_features

        # Initialize models
        self.knn_model = KNeighborsRegressor(n_neighbors=self.k)
        self.svr_model = SVR(kernel=self.svr_kernel, gamma=self.gamma, degree=self.degree, C=self.C)
        self.rf_model = RandomForestRegressor(n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth, min_samples_split=self.rf_samples_split, 
                         max_features=self.rf_max_features, min_samples_leaf=self.rf_samples_leaf, bootstrap=False, warm_start=True)
        self.gbm_model = GradientBoostingRegressor(n_estimators=self.gbm_n_estimators, 
                                                   criterion=self.gbm_criterion, loss=self.gbm_loss
                                                   ,max_features=self.gbm_n_features)
        self.lr_model = LinearRegression()

    def knn_fit(self, dfX, vY):
        self.knn_model.fit(dfX, vY)

    def svr_fit(self, dfX, vY):
        self.svr_model.fit(dfX, vY)

    def rf_fit(self, dfX, vY):
        self.rf_model.fit(dfX, vY)

    def gbm_fit(self, dfX, vY):
        self.gbm_model.fit(dfX, vY)
    
    def lr_fit(self, dfX, vY):
        self.lr_model.fit(dfX, vY)

    def knn_predict(self, dfX):
        return self.knn_model.predict(dfX)

    def svr_predict(self, dfX):
        return self.svr_model.predict(dfX)

    def rf_predict(self, dfX):
        return self.rf_model.predict(dfX)

    def gbm_predict(self, dfX):
        return self.gbm_model.predict(dfX)
    
    def lr_predict(self, dfX):
        return self.lr_model.predict(dfX)

    def knn_score(self, dfX, vY):
        return self.knn_model.score(dfX, vY)

    def svr_score(self, dfX, vY):
        return self.svr_model.score(dfX, vY)

    def rf_score(self, dfX, vY):
        return self.rf_model.score(dfX, vY)

    def gbm_score(self, dfX, vY):
        return self.gbm_model.score(dfX, vY)
    
    def lr_score(self, dfX, vY):
        return self.lr_model.score(dfX, vY)

    def knn_train(self, splits, col_drop):
        predictions = []
        
        for i, (in_sample, out_of_sample) in enumerate(splits):
            X_train = in_sample.drop(columns=[col_drop])
            y_train = in_sample[col_drop]
            self.knn_model.fit(X_train, y_train)

            X_test = out_of_sample.drop(columns=[col_drop])
            # Create an array of NaN values with the same length as the NaN window
            nan_values = np.full(19, np.nan)
            prediction = self.knn_model.predict(X_test)
            # Concatenate NaN values with the non-NaN part of the predictions
            aligned_predictions = np.concatenate([nan_values, prediction])
            predictions.extend(aligned_predictions)
        return (predictions)
    
    def svr_train(self, splits, col_drop):
        predictions = []
        for i, (in_sample, out_of_sample) in enumerate(splits):
            X_train = in_sample.drop(columns=[col_drop])
            y_train = in_sample[col_drop]
            self.svr_model.fit(X_train, y_train)

            X_test = out_of_sample.drop(columns=[col_drop])
            # Create an array of NaN values with the same length as the NaN window
            nan_values = np.full(19, np.nan)
            prediction = self.svr_model.predict(X_test)
            # Concatenate NaN values with the non-NaN part of the predictions
            aligned_predictions = np.concatenate([nan_values, prediction])
            predictions.extend(aligned_predictions)
        return (predictions)
    
    def rf_train(self, splits, col_drop):
        predictions = []
        for i, (in_sample, out_of_sample) in enumerate(splits):
            X_train = in_sample.drop(columns=[col_drop])
            y_train = in_sample[col_drop]
            self.rf_model.fit(X_train, y_train)

            X_test = out_of_sample.drop(columns=[col_drop])
            # Create an array of NaN values with the same length as the NaN window
            nan_values = np.full(19, np.nan)
            prediction = self.rf_model.predict(X_test)
            # Concatenate NaN values with the non-NaN part of the predictions
            aligned_predictions = np.concatenate([nan_values, prediction])
            predictions.extend(aligned_predictions)
        return (predictions)

    def gbm_train(self, splits, col_drop):
        predictions = []
        for i, (in_sample, out_of_sample) in enumerate(splits):
            X_train = in_sample.drop(columns=[col_drop])
            y_train = in_sample[col_drop]
            self.gbm_model.fit(X_train, y_train)

            X_test = out_of_sample.drop(columns=[col_drop])
            # Create an array of NaN values with the same length as the NaN window
            nan_values = np.full(19, np.nan)
            prediction = self.gbm_model.predict(X_test)
            # Concatenate NaN values with the non-NaN part of the predictions
            aligned_predictions = np.concatenate([nan_values, prediction])
            predictions.extend(aligned_predictions)
        return (predictions)
    
    def lr_train(self, splits, col_drop):
        predictions = []
        for i, (in_sample, out_of_sample) in enumerate(splits):
            X_train = in_sample.drop(columns=[col_drop])
            y_train = in_sample[col_drop]
            self.lr_model.fit(X_train, y_train)

            X_test = out_of_sample.drop(columns=[col_drop])
            # Create an array of NaN values with the same length as the NaN window
            nan_values = np.full(19, np.nan)
            prediction = self.lr_model.predict(X_test)
            # Concatenate NaN values with the non-NaN part of the predictions
            aligned_predictions = np.concatenate([nan_values, prediction])
            predictions.extend(aligned_predictions)
        return (predictions)
    
    def evaluate(self, y_true, y_pred):
        self.mae = mean_absolute_error(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(y_true, y_pred)
        self.mape = mean_absolute_percentage_error(y_true, y_pred)

        return (self.mae, self.mse, self.rmse, self.r2, self.mape)

def svr_opt(df, splits_f, index_dropped):
    lC      = [1, 10, 100]
    lKernel = ['poly', 'rbf','linear']
    lgamma      = ['scale', 'auto', 0.1, 0.01, 0.001]
    ldegree = [3, 4, 5, 6, 7, 8, 9]

    # Creating the Data Frame

    #===========================Fill This===========================#
    # 1. Calculate the number of combinations.
    # 2. Create a nested loop to create the combinations between the parameters.
    # 3. Store the combinations as the columns of a data frame.

    # For Advanced Python users: Use iteration tools for create the cartesian product
    numComb = len(lKernel) * len(lC) * len(lgamma) * len(ldegree)
    dData   = {'kernel': [], 'C': [], 'gamma':[], 'degree':[]}

    for ii, kernel in enumerate(lKernel):
        for jj, paramC in enumerate(lC):
            for kk, gamma in enumerate(lgamma):
                for cc, degree in enumerate(ldegree):
                    dData['kernel'].append(kernel)
                    dData['C'].append(paramC)
                    dData['gamma'].append(gamma)
                    dData['degree'].append(degree)
    #===============================================================#

    dfModelScore = pd.DataFrame(data = dData)
    
    #Define the Model Object
    Model = TimeSeriesModels()

    # Initialize an empty list to store predictions
    vY = df['Adj Close'].values
    vY = vY[index_dropped:]

    for ii in range(numComb):
        kernel    = dfModelScore.loc[ii, 'kernel']
        paramC          = dfModelScore.loc[ii, 'C']
        gamma          = dfModelScore.loc[ii, 'gamma']
        degree          = dfModelScore.loc[ii, 'degree']


        print(f'Processing model {ii + 1:03d} out of {numComb}')

        Model = TimeSeriesModels(svr_kernel=kernel, C=paramC, gamma=gamma, degree=degree)
        predictions = Model.svr_train(splits_f, 'Adj Close')
        
        #contains true values in one column and predicted in the other without nans, in order to calculate eroors over all of the predicted period
        mY = pd.DataFrame(columns=['vY', 'Predictions'])
        mY['vY'] = vY
        mY['Predictions'] =  predictions
        mY = Functions.drop_nan(mY)
        # Calculate evaluation metrics
        #the calculation for eaach model was performed on the last split of the walk forward
        mae, mse, rmse, r2, mape = Model.evaluate(mY['vY'], mY['Predictions'])
        
        # Update the 'R2' column in dfModelScore with the calculated R2 score
        dfModelScore.loc[ii, 'MAE'] = mae
        dfModelScore.loc[ii, 'MSE'] = mse
        dfModelScore.loc[ii, 'RMSE'] = rmse
        dfModelScore.loc[ii, 'R2'] = r2
        dfModelScore.loc[ii, 'MAPE'] = mape

    #Find the best model
    # Calculate the mean of MAE, MSE, RMSE, and MAPE for each model
    dfModelScore['MeanScore'] = dfModelScore[['MAE', 'RMSE', 'MAPE']].mean(axis=1)

    # Find the index of the model with the lowest mean score
    best_model_index = dfModelScore['MeanScore'].idxmin()

    # Find the index of the model with the higher r2
    best_model_index_r2 = dfModelScore['R2'].idxmax()

    # Get the parameters of the best model
    best_model_params = dfModelScore.loc[best_model_index, ['kernel', 'C', 'gamma', 'degree']]
    best_model_mean_val = dfModelScore.loc[best_model_index, 'MeanScore']
    # Get the parameters of the best model
    best_model_params_r2 = dfModelScore.loc[best_model_index_r2, ['kernel', 'C', 'gamma', 'degree']]
    best_model_r2_val = dfModelScore.loc[best_model_index_r2, 'R2']

    best_model = pd.DataFrame({
        'Mean(MAE RMSE MAPE)': [best_model_mean_val],
        'Mean Params': [best_model_params],
        'R2': [best_model_r2_val],
        'R2 Params': [best_model_params_r2]
    })
    
    return(best_model)

def knn_opt(df, splits_f, index_dropped):
    dData   = {'K': []}
    dfModelScore = pd.DataFrame(data = dData)

    Model = TimeSeriesModels()

    # Initialize an empty list to store predictions
    vY = df['Adj Close'].values
    vY = vY[index_dropped:]
    for kk in range(10):
        print(f'Processing model {kk + 1:03d} out of {10}')

        Model = TimeSeriesModels(k=kk + 1)
        predictions = Model.knn_train(splits_f, 'Adj Close')
        
        #contains true values in one column and predicted in the other without nans, in order to calculate eroors over all of the predicted period
        mY = pd.DataFrame(columns=['vY', 'Predictions'])
        mY['vY'] = vY
        mY['Predictions'] =  predictions
        mY = Functions.drop_nan(mY)
        # Calculate evaluation metrics
        #the calculation for eaach model was performed on the last split of the walk forward
        mae, mse, rmse, r2, mape = Model.evaluate(mY['vY'], mY['Predictions'])
        
        # Update the 'R2' column in dfModelScore with the calculated R2 score
        dfModelScore.loc[kk, 'k'] = kk + 1
        dfModelScore.loc[kk, 'MAE'] = mae
        dfModelScore.loc[kk, 'MSE'] = mse
        dfModelScore.loc[kk, 'RMSE'] = rmse
        dfModelScore.loc[kk, 'R2'] = r2
        dfModelScore.loc[kk, 'MAPE'] = mape

    #===============================================================#
    # Calculate the mean of MAE, MSE, RMSE, and MAPE for each model
    dfModelScore['MeanScore'] = dfModelScore[['MAE', 'RMSE', 'MAPE']].mean(axis=1)

    # Find the index of the model with the lowest mean score
    best_model_index = dfModelScore['MeanScore'].idxmin()

    # Find the index of the model with the higher r2
    best_model_index_r2 = dfModelScore['R2'].idxmax()

    # Get the parameters of the best model
    best_model_params = dfModelScore.loc[best_model_index, ['k']]
    best_model_mean_val = dfModelScore.loc[best_model_index, 'MeanScore']
    # Get the parameters of the best model
    best_model_params_r2 = dfModelScore.loc[best_model_index_r2, ['k']]
    best_model_r2_val = dfModelScore.loc[best_model_index_r2, 'R2']

    best_model = pd.DataFrame({
        'Mean(MAE RMSE MAPE)': [best_model_mean_val],
        'Mean Params': [best_model_params],
        'R2': [best_model_r2_val],
        'R2 Params': [best_model_params_r2]
    })
    
    return(best_model)

def rf_opt(df, splits_f, index_dropped):
    m_features = ['sqrt', 'log2', 4, 5, 6, 7]
    max_depth = [10, 20, 30]
    n_estimators = [50, 100, 200, 300] #number of threes in the forest (100 default)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    # Creating the Data Frame
    #===========================Fill This===========================#
    # 1. Calculate the number of combinations.
    # 2. Create a nested loop to create the combinations between the parameters.
    # 3. Store the combinations as the columns of a data frame.

    # For Advanced Python users: Use iteration tools for create the cartesian product
    numComb = len(m_features) * len(max_depth) * len(n_estimators) * len(min_samples_split) * len(min_samples_leaf)
    dData   = {'max_features': [], 'max_depth': [], 'n_estimators':[], 'min_samples_split':[], 'min_samples_leaf':[]}

    for ii, feature in enumerate(m_features):
        for jj, cri in enumerate(max_depth):
            for kk, est in enumerate(n_estimators):
                for ll, ssplit in enumerate(min_samples_split):
                    for bb, sleaf in enumerate(min_samples_leaf):
                        dData['max_features'].append(feature)
                        dData['max_depth'].append(cri)
                        dData['n_estimators'].append(est)
                        dData['min_samples_split'].append(ssplit)
                        dData['min_samples_leaf'].append(sleaf)
    #===============================================================#

    dfModelScore = pd.DataFrame(data = dData)

    Model = TimeSeriesModels()

    # Initialize an empty list to store predictions 
    vY = df['Adj Close'].values
    vY = vY[index_dropped:]
    for ii in range(numComb):
        rf_n_feature = dfModelScore.loc[ii, 'max_features']
        rf_m_depth = dfModelScore.loc[ii, 'max_depth']
        rf_n_est = dfModelScore.loc[ii, 'n_estimators']
        rf_s_leaf = dfModelScore.loc[ii, 'min_samples_leaf']
        rf_s_split = dfModelScore.loc[ii, 'min_samples_split']

        print(f'Processing model {ii + 1:03d} out of {numComb}')
        Model = TimeSeriesModels(rf_n_estimators=rf_n_est, rf_max_features=rf_n_feature, rf_samples_split=rf_s_split, rf_max_depth=rf_m_depth, rf_samples_leaf=rf_s_leaf)
        predictions = Model.rf_train(splits_f, 'Adj Close')
        
        #contains true values in one column and predicted in the other without nans, in order to calculate eroors over all of the predicted period
        mY = pd.DataFrame(columns=['vY', 'Predictions'])
        mY['vY'] = vY
        mY['Predictions'] =  predictions
        mY = Functions.drop_nan(mY)
        # Calculate evaluation metrics
        #the calculation for eaach model was performed on the last split of the walk forward
        mae, mse, rmse, r2, mape = Model.evaluate(mY['vY'], mY['Predictions'])
        
        # Update the 'R2' column in dfModelScore with the calculated R2 score
        dfModelScore.loc[ii, 'MAE'] = mae
        dfModelScore.loc[ii, 'MSE'] = mse
        dfModelScore.loc[ii, 'RMSE'] = rmse
        dfModelScore.loc[ii, 'R2'] = r2
        dfModelScore.loc[ii, 'MAPE'] = mape

    #===============================================================#

    # Calculate the mean of MAE, MSE, RMSE, and MAPE for each model
    dfModelScore['MeanScore'] = dfModelScore[['MAE', 'RMSE', 'MAPE']].mean(axis=1)

    # Find the index of the model with the lowest mean score
    best_model_index = dfModelScore['MeanScore'].idxmin()

    # Find the index of the model with the higher r2
    best_model_index_r2 = dfModelScore['R2'].idxmax()

    # Get the parameters of the best model
    best_model_params = dfModelScore.loc[best_model_index, ['max_features', 'max_depth', 'n_estimators', 'min_samples_leaf', 'min_samples_split']]
    best_model_mean_val = dfModelScore.loc[best_model_index, 'MeanScore']
    # Get the parameters of the best model
    best_model_params_r2 = dfModelScore.loc[best_model_index_r2, ['max_features', 'max_depth', 'n_estimators', 'min_samples_leaf', 'min_samples_split']]
    best_model_r2_val = dfModelScore.loc[best_model_index_r2, 'R2']

    best_model = pd.DataFrame({
        'Mean(MAE RMSE MAPE)': [best_model_mean_val],
        'Mean Params': [best_model_params],
        'R2': [best_model_r2_val],
        'R2 Params': [best_model_params_r2]
    })
    
    return(best_model)

def gbm_opt(df, splits_f, index_dropped):
    L_loss = ['squared_error', 'absolute_error', 'huber']
    L_criterion = ['friedman_mse', 'squared_error']
    L_n_estimators = [100, 150, 200] #number of threes in the forest (100 default)
    L_m_features = ['sqrt', 'log2', 5, 7]

    # Creating the Data Frame

    #===========================Fill This===========================#
    # 1. Calculate the number of combinations.
    # 2. Create a nested loop to create the combinations between the parameters.
    # 3. Store the combinations as the columns of a data frame.

    # For Advanced Python users: Use iteration tools for create the cartesian product
    numComb = len(L_loss) * len(L_criterion) * len(L_n_estimators) *len(L_m_features)
    dData   = {'loss': [], 'criterion': [], 'n_estimators':[], 'm_features':[]}

    for ii, lss in enumerate(L_loss):
        for jj, cri in enumerate(L_criterion):
            for kk, est in enumerate(L_n_estimators):
                for kk, feature in enumerate(L_m_features):
                    dData['loss'].append(lss)
                    dData['criterion'].append(cri)
                    dData['n_estimators'].append(est)
                    dData['m_features'].append(feature)
    #===============================================================#

    dfModelScore = pd.DataFrame(data = dData)

    Model = TimeSeriesModels()

    # Initialize an empty list to store predictions
    vY = df['Adj Close'].values
    vY = vY[index_dropped:]
    for ii in range(numComb):
        gbm_lo = dfModelScore.loc[ii, 'loss']
        gbm_cri = dfModelScore.loc[ii, 'criterion']
        gbm_n_est = dfModelScore.loc[ii, 'n_estimators']
        gbm_feat = dfModelScore.loc[ii, 'm_features']

        print(f'Processing model {ii + 1:03d} out of {numComb}')
        Model = TimeSeriesModels(gbm_n_estimators=gbm_n_est, gbm_criterion=gbm_cri, gbm_loss=gbm_lo
                                                    ,gbm_n_features=gbm_feat)
        predictions = Model.gbm_train(splits_f, 'Adj Close')

        #contains true values in one column and predicted in the other without nans, in order to calculate eroors over all of the predicted period
        mY = pd.DataFrame(columns=['vY', 'Predictions'])
        mY['vY'] = vY
        mY['Predictions'] =  predictions
        mY = Functions.drop_nan(mY)
        # Calculate evaluation metrics
        #the calculation for eaach model was performed on the last split of the walk forward
        mae, mse, rmse, r2, mape = Model.evaluate(mY['vY'], mY['Predictions'])
        
        # Update the 'R2' column in dfModelScore with the calculated R2 score
        dfModelScore.loc[ii, 'MAE'] = mae
        dfModelScore.loc[ii, 'MSE'] = mse
        dfModelScore.loc[ii, 'RMSE'] = rmse
        dfModelScore.loc[ii, 'R2'] = r2
        dfModelScore.loc[ii, 'MAPE'] = mape

    #===============================================================#
    # Calculate the mean of MAE, MSE, RMSE, and MAPE for each model
    dfModelScore['MeanScore'] = dfModelScore[['MAE', 'RMSE', 'MAPE']].mean(axis=1)

    # Find the index of the model with the lowest mean score
    best_model_index = dfModelScore['MeanScore'].idxmin()

    # Find the index of the model with the higher r2
    best_model_index_r2 = dfModelScore['R2'].idxmax()

    # Get the parameters of the best model
    best_model_params = dfModelScore.loc[best_model_index, ['loss', 'criterion', 'n_estimators', 'm_features']]
    best_model_mean_val = dfModelScore.loc[best_model_index, 'MeanScore']
    # Get the parameters of the best model
    best_model_params_r2 = dfModelScore.loc[best_model_index_r2, ['loss', 'criterion', 'n_estimators', 'm_features']]
    best_model_r2_val = dfModelScore.loc[best_model_index_r2, 'R2']

    best_model = pd.DataFrame({
        'Mean(MAE RMSE MAPE)': [best_model_mean_val],
        'Mean Params': [best_model_params],
        'R2': [best_model_r2_val],
        'R2 Params': [best_model_params_r2]
    })
    
    return(best_model)

def models_params_optimization(df_training_set, best_ratio = 3, best_splits_num = 60):
    Best_Params = {}
    for key, value in df_training_set.items():
        data_ind = value.copy()
        #Split the data
        out_sample_size = len(data_ind) // (best_ratio + best_splits_num)
        in_sample_size = best_ratio * out_sample_size
        index_dropped = best_ratio * out_sample_size #first idicies that are participating in the insample bur not predicted

        data_ind = truncate_before_wf(data_ind, in_sample_size, out_sample_size)
        splits = walk_forward_validation(data_ind, in_sample_size, out_sample_size)
        #create features
        splits_f = apply_feature_creation_to_splits(splits, chosen_features)

        #run the optimizations
        best_model_svr = svr_opt(data_ind, splits_f, index_dropped)
        best_model_knn = knn_opt(data_ind, splits_f, index_dropped)
        best_model_rf = rf_opt(data_ind, splits_f, index_dropped)
        best_model_gbm = gbm_opt(data_ind, splits_f, index_dropped)

        Best_Params[key] = {
            'SVR_opt': best_model_svr,
            'KNN_opt': best_model_knn,
            'RF_opt': best_model_rf,
            'GBM_opt': best_model_gbm
        }
    return Best_Params

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