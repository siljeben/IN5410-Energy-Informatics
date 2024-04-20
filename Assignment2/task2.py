import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from plot_functions import plot_timeseries

train_df = pd.read_csv('data/TrainData.csv')
time_train = train_df["TIMESTAMP"]
pred_df = pd.read_csv('data/WeatherForecastInput.csv')
solution_df = pd.read_csv('data/Solution.csv')

feature_cols_zonal = ["U10"]
feature_cols_meridional = ["V10"]
feature_cols_windspeed = ["WS10"]
label = "POWER"

# from TrainData; extracting zonal component U10 and the meridional component V10 of the wind forecast to calculate the wind direction, windspeed and power 
X_train_cols_meridional = train_df.loc[:, feature_cols_meridional].values
X_train_cols_zonal = train_df.loc[:, feature_cols_zonal].values

X_train_wind_direction = np.arctan2(X_train_cols_meridional, X_train_cols_zonal)
X_train_cols_windspeed = train_df.loc[:, feature_cols_windspeed]
X_train = np.concatenate((X_train_wind_direction, X_train_cols_windspeed), axis=1)

y_train = train_df.loc[:, label]

# from WeaterForecastInput; extracting zonal component U10 and the meridional component V10 of the wind forecast to calculate the wind direction, windspeed and power 
X_pred_cols_meridional = pred_df.loc[:, feature_cols_meridional].values
X_pred_cols_zonal = pred_df.loc[:, feature_cols_zonal].values

X_pred_wind_direction = np.arctan2(X_pred_cols_meridional, X_pred_cols_zonal)
X_pred_cols_windspeed = pred_df.loc[:, feature_cols_windspeed]
X_pred = np.concatenate((X_pred_wind_direction, X_pred_cols_windspeed), axis=1)

y_sol = solution_df.loc[:, label]

# handle missing values
imputer = SimpleImputer(strategy="mean")

X_train_imputed = imputer.fit_transform(X_train)
X_pred_imputed = imputer.transform(X_pred)

lr_model = LinearRegression()
lr_model.fit(X_train_imputed, y_train)
lr_y_pred = lr_model.predict(X_pred_imputed)
lr_error = np.sqrt(mean_squared_error(y_sol, lr_y_pred))

print(f"RMSE for the model: {lr_error}") 
#0.2155597198828209

lr_pred_task1 = pd.read_csv('predictions/task1/ForecastTemplate1-LR.csv')['POWER'].values

plot_timeseries(time_train, [y_sol, lr_pred_task1, lr_y_pred], ['True power output', 'Linear Regression Forecast', 'Multiple Linear Regression Forecast'], 'Wind Power Prediction Accuracy Assessment', 'Power output')

""" Save to csv file
lr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': lr_y_pred})
lr_result_df.to_csv('predictions/task1/ForecastTemplate1-LR.csv', index=False)
"""