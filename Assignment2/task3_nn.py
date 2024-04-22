import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from plot_functions import plot_timeseries
from data_processing import (
    get_sliding_window_input_output,
    get_sliding_window_input_test_data,
)
from neural_net_models import Net, RNN, train_model

window_size = 144
epochs = 25
lr = 1e-3

train_df = pd.read_csv("data/TrainData.csv")
time_train = train_df["TIMESTAMP"].values
power_train = train_df["POWER"].values.astype(np.float32)

test_df = pd.read_csv("data/WeatherForecastInput.csv")
time_test = test_df["TIMESTAMP"].values
power_test = pd.read_csv("data/Solution.csv")["POWER"].values.astype(np.float32)

X_train, y_train = get_sliding_window_input_output(power_train, window_size=window_size)
y_train = y_train.reshape(-1, 1)

X_test = get_sliding_window_input_test_data(
    power_test, power_train, window_size=window_size
)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = power_test.reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_x, validation_x, train_y, validation_y = train_test_split(
    X_train, y_train, test_size=0.2, random_state=69
)

trainloader = torch.utils.data.DataLoader(
    list(zip(train_x, train_y)), batch_size=32, shuffle=True
)
validationloader = torch.utils.data.DataLoader(
    list(zip(validation_x, validation_y)), batch_size=32, shuffle=False
)

torch.manual_seed(42)


def train_ann():
    # ANN model
    net = Net(window_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=1e-6)

    train_model(epochs, net, criterion, optimizer, trainloader, validationloader)

    test_pred = net(X_test)
    test_loss = criterion(test_pred, y_test)
    print("ANN model RMSE: " + str(np.sqrt(test_loss.item())))

    plot_timeseries(
        time_test,
        [y_test.numpy(), test_pred.detach().numpy()],
        ["True power output", "Predicted power output"],
        "Test data, ANN model",
        "Power output",
    )


# RNN model


def train_rnn():
    rnn = RNN(window_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(rnn.parameters(), lr=lr, weight_decay=1e-6)

    train_model(epochs, rnn, criterion, optimizer, trainloader, validationloader)

    test_pred = rnn(X_test)
    test_loss = criterion(test_pred, y_test)
    print("ANN model RMSE: " + str(np.sqrt(test_loss.item())))

    plot_timeseries(
        time_test,
        [y_test.numpy(), test_pred.detach().numpy()],
        ["True power output", "Predicted power output"],
        "Test data, RNN model",
        "Power output",
    )


train_ann()
train_rnn()
