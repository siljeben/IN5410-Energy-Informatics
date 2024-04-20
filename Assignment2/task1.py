import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from plot_functions import plot_timeseries

train_df = pd.read_csv('data/TrainData.csv')
time_train = train_df["TIMESTAMP"]
pred_df = pd.read_csv('data/WeatherForecastInput.csv')
solution_df = pd.read_csv('data/Solution.csv')

feature_cols = ["WS10"]
label = "POWER"

X_train = train_df.loc[:, feature_cols]
y_train = train_df.loc[:, label]

X_test = pred_df.loc[:, feature_cols]
y_sol = solution_df.loc[:, label]

svr_model = SVR()
svr_model.fit(X_train,y_train)
svr_y_pred = svr_model.predict(X_test)
svr_error = np.sqrt(mean_squared_error(y_sol, svr_y_pred))

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_y_pred = lr_model.predict(X_test)
lr_error = np.sqrt(mean_squared_error(y_sol, lr_y_pred))

knn_model = KNeighborsRegressor()
knn_model.fit(X_train,y_train)
knn_y_pred = knn_model.predict(X_test)
knn_error = np.sqrt(mean_squared_error(y_sol, knn_y_pred))

print(f"Errors for the methods;\n SVR: {svr_error}\n LR: {lr_error}\n kNN: {knn_error}")
#SVR=0.21374359746589766, LR=0.21638408562354403, kNN=0.23486489675998057

plot_timeseries(time_train, [y_sol, svr_y_pred], ['True power output', 'Predicted power output'], 'SVR model', 'Power output')

plot_timeseries(time_train, [y_sol, lr_y_pred], ['True power output', 'Predicted power output'], 'LR model', 'Power output')

plot_timeseries(time_train, [y_sol, knn_y_pred], ['True power output', 'Predicted power output'], 'KNN model', 'Power output')

""" Save to csv file
svr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': svr_y_pred})
svr_result_df.to_csv('predictions/task1/ForecastTemplate1-SVR.csv', index=False)

lr_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': lr_y_pred})
lr_result_df.to_csv('predictions/task1/ForecastTemplate1-LR.csv', index=False)

knn_result_df = pd.DataFrame({'TIMESTAMP': pred_df['TIMESTAMP'], 'POWER': knn_y_pred})
knn_result_df.to_csv('predictions/task1/ForecastTemplate1-kNN.csv', index=False)
"""