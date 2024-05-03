import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from plot_functions import plot_timeseries
from data_processing import components_to_angle
from data_processing import convert_str_to_datetime

train_df = pd.read_csv("data/TrainData.csv")
time_train = train_df["TIMESTAMP"]
pred_df = pd.read_csv("data/WeatherForecastInput.csv")
solution_df = pd.read_csv("data/Solution.csv")

label = "POWER"

# from TrainData; extracting zonal component U10 and the meridional component V10 of the wind forecast to calculate the wind direction, windspeed and power
train_df = components_to_angle(train_df)

X_train_wind_direction = train_df["A10"].values
X_train_cols_windspeed = train_df["WS10"].values
X_train = np.stack((X_train_wind_direction, X_train_cols_windspeed), axis=1)

y_train = train_df.loc[:, label]

# from WeatherForecastInput; extracting zonal component U10 and the meridional component V10 of the wind forecast to calculate the wind direction, windspeed and power
pred_df = components_to_angle(pred_df)
time_test = pred_df["TIMESTAMP"].apply(convert_str_to_datetime).dt.strftime('%m-%d')

X_pred_wind_direction = pred_df["A10"].values
X_pred_cols_windspeed = pred_df["WS10"].values
X_pred = np.stack((X_pred_wind_direction, X_pred_cols_windspeed), axis=1)

y_sol = solution_df.loc[:, label]

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
mlr_y_pred = mlr_model.predict(X_pred)
mlr_error = np.sqrt(mean_squared_error(y_sol, mlr_y_pred))

print(f"RMSE for the model: {mlr_error}")
# 0.21494171563903242

lr_pred_task1 = pd.read_csv("predictions/task1/ForecastTemplate1-LR.csv")[
    "POWER"
].values

plot_timeseries(
    time_test,
    [y_sol, lr_pred_task1, mlr_y_pred],
    ["True power output", "LR model", "MLR model"],
    "Wind Power Predictions LR + MLR",
    "Power output [normalized]",
    savepath="plots/task2/WindPowerPredictionsLR_MLR.svg",
    figsize=[30, 10],
    linewidth=1,
)

""" Save to csv file
lr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': lr_y_pred})
lr_result_df.to_csv('predictions/task1/ForecastTemplate1-LR.csv', index=False)
"""
