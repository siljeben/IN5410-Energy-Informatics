import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from plot_functions import plot_timeseries
from data_processing import sliding_window_input_output

train_df = pd.read_csv('data/TrainData.csv')
time_train = train_df["TIMESTAMP"].values
power_train = train_df["POWER"].values.astype(np.float32)

test_df = pd.read_csv('data/WeatherForecastInput.csv')
time_test = test_df["TIMESTAMP"].values
power_test = pd.read_csv('data/Solution.csv')["POWER"].values.astype(np.float32)

window_size = 3

X_train, y_train = sliding_window_input_output(power_train, window_size=window_size)
y_train = y_train.reshape(-1, 1)
y_test = power_test.reshape(-1, 1)

train_x, validation_x, train_y, validation_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

trainloader = torch.utils.data.DataLoader(list(zip(train_x, train_y)), batch_size=32, shuffle=True)
validationloader = torch.utils.data.DataLoader(list(zip(validation_x, validation_y)), batch_size=32, shuffle=False)

from task1_nn import Net

torch.manual_seed(42)
net = Net(window_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

epochs = 30

for epoch in range(epochs):
    for x, y in trainloader:
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    val_loss = 0
    for x, y in validationloader:
        net.eval()
        output = net(x)
        val_loss += criterion(output, y)
        net.train()
    val_loss /= len(validationloader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val. Loss: {val_loss.item()}')


def predict_test(model):
    y_out = power_train[-window_size:].tolist()
    for i in range(len(power_test)):
        x = torch.tensor(y_out[-window_size:]).reshape(1, -1)
        y = model(x)
        y_out.append(y.item())
    return np.array(y_out)[window_size:].reshape(-1, 1)

test_pred = predict_test(net)
print(test_pred)
test_loss = criterion(test_pred, torch.tensor(y_test).reshape(-1, 1))
print(np.sqrt(test_loss.item()))
