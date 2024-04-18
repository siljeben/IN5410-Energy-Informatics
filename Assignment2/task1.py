import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('data/TrainData.csv')
pred_df = pd.read_csv('data/WeatherForecastInput.csv')
solution_df = pd.read_csv('data/Solution.csv')

feature_cols = ["WS10"]
label = "POWER"

X_train = train_df.loc[:, feature_cols]
y_train = train_df.loc[:, label]

X_test = pred_df.loc[:, feature_cols]
y = solution_df.loc[:, label]

svr_model = SVR()
svr_model.fit(X_train,y_train)
svr_y_pred = svr_model.predict(X_test)
svr_error = np.sqrt(mean_squared_error(y, svr_y_pred))

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_y_pred = lr_model.predict(X_test)
lr_error = np.sqrt(mean_squared_error(y, lr_y_pred))

knn_model = KNeighborsRegressor()
knn_model.fit(X_train,y_train)
knn_y_pred = knn_model.predict(X_test)
knn_error = np.sqrt(mean_squared_error(y, knn_y_pred))

print(f"Errors for the methods; SVR: {svr_error}, LR: {lr_error}, kNN: {knn_error}")