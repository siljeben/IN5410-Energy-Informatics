import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from plot_functions import plot_timeseries
from data_processing import (
    get_sliding_window_input_output,
    get_sliding_window_input_test_data,
)

# Load data
pred_df = pd.read_csv("data/WeatherForecastInput.csv")
solution_df = pd.read_csv("data/Solution.csv")

# Extract features and target
train_df = pd.read_csv("data/TrainData.csv")
time_train = train_df["TIMESTAMP"].values
power_train = train_df["POWER"].values


test_pred = pred_df["TIMESTAMP"].values

y_sol = solution_df["POWER"].values

# SVR model
window_size_svr = 83
X_test_svr = get_sliding_window_input_test_data(
    y_sol, power_train, window_size=window_size_svr
)
X_train_svr, y_train_svr = get_sliding_window_input_output(
    power_train, window_size=window_size_svr
)

svr_model = SVR()
svr_model.fit(X_train_svr, y_train_svr)
svr_y_pred = svr_model.predict(X_test_svr)
svr_error = np.sqrt(mean_squared_error(y_sol, svr_y_pred))

# Linear Regression model
window_size_lr = 100
X_test_lr = get_sliding_window_input_test_data(
    y_sol, power_train, window_size=window_size_lr
)
X_train_lr, y_train_lr = get_sliding_window_input_output(
    power_train, window_size=window_size_lr
)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
lr_y_pred = lr_model.predict(X_test_lr)
lr_error = np.sqrt(mean_squared_error(y_sol, lr_y_pred))

print(
    f"Errors for the methods;\n"
    f"SVR: {svr_error}, window_size= {window_size_svr}\n "
    f"LR: {lr_error}, window_size= {window_size_lr}\n "
)

# SVR: 0.1282092712257549, window_size= 83
# LR: 0.12223004615190537, window_size= 100

plot_timeseries(
    test_pred,
    [y_sol, svr_y_pred],
    ["True power output", "Predicted power output"],
    "SVR model",
    "Power output",
)

plot_timeseries(
    test_pred,
    [y_sol, lr_y_pred],
    ["True power output", "Predicted power output"],
    "LR model",
    "Power output",
)

""" Save to csv file
svr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': svr_y_pred})
svr_result_df.to_csv('predictions/task3/ForecastTemplate3-SVR.csv', index=False)

lr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': lr_y_pred})
lr_result_df.to_csv('predictions/task3/ForecastTemplate3-LR.csv', index=False)
"""
