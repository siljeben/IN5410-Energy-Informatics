# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from plot_functions import plot_timeseries, speed_power_plot

# %%
train_df = pd.read_csv('data/TrainData.csv')
time_train = train_df["TIMESTAMP"]
train_x_chron = train_df.WS10.to_numpy(dtype=np.float32).reshape(-1, 1)
train_y_chron = train_df.POWER.to_numpy(dtype=np.float32).reshape(-1, 1)
train_x, validation_x, train_y, validation_y = train_test_split(train_x_chron, train_y_chron, test_size=0.2, random_state=42)

# %%
test_df_x = pd.read_csv('data/WeatherForecastInput.csv')
time_test = test_df_x.TIMESTAMP
test_x = test_df_x.WS10.to_numpy(dtype=np.float32).reshape(-1, 1)
test_df_y = pd.read_csv('data/Solution.csv')
test_y = test_df_y.POWER.to_numpy(dtype=np.float32).reshape(-1, 1)

# %%
trainloader = torch.utils.data.DataLoader(list(zip(train_x, train_y)), batch_size=32, shuffle=True)
validationloader = torch.utils.data.DataLoader(list(zip(validation_x, validation_y)), batch_size=32, shuffle=False)
testloader = torch.utils.data.DataLoader(list(zip(test_x, test_y)), batch_size=32, shuffle=False)

# %%
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))
        return x

# %%
torch.manual_seed(42)
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

# %%
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

# %%
test_pred = net(torch.tensor(test_x))
test_loss = criterion(test_pred, torch.tensor(test_y))
print(np.sqrt(test_loss.item()))
# 0.21303973094683254

# %%
train_pred = net(torch.tensor(train_x_chron))

# %%

plot_timeseries(time_train[:500], [train_y_chron[:500], train_pred[:500].detach().numpy()], ['True output', 'Predicted output'], 'Train data', 'Power output')

# %%

plot_timeseries(time_test, [test_y, test_pred.detach().numpy()], ['True power output', 'Predicted power output'], 'Test data', 'Power output')

# %%
model = lambda x: net(torch.tensor(x.astype(np.float32).reshape(-1, 1))).detach().numpy()
speed_power_plot(test_x, test_y, model)

""" Save to csv file
nn_result_df = pd.DataFrame({'TIMESTAMP': time_test, 'POWER': test_pred.detach().numpy().flatten()})
nn_result_df.to_csv('predictions/task1/ForecastTemplate1-NN.csv', index=False)
"""