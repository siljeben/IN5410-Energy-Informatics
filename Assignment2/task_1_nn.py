# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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
    print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item()}')

# %%
test_pred = net(torch.tensor(test_x))
test_loss = criterion(test_pred, torch.tensor(test_y))
np.sqrt(test_loss.item())

# %%
train_pred = net(torch.tensor(train_x_chron))

# %%
# make big plot
plt.figure(figsize=(15, 7))
plt.plot(train_y_chron[:500], 'r', label='True power output')
plt.plot(train_pred[:500].detach().numpy(), 'b', label='Predicted power output')
plt.legend()
plt.title('Train data')
plt.ylabel('Power output')
plt.xticks(np.arange(0, 500, 50), time_train[:500:50], rotation=45)
plt.show()

# %%
# make big plot
plt.figure(figsize=(15, 7))

plt.plot(test_y, 'r', label='True power output, test data')
plt.plot(test_pred.detach().numpy(), 'b', label='Predicted power output, test data')
plt.legend()
plt.title('Test data')
plt.ylabel('Power output')
plt.xticks(np.linspace(0, 720, 15), time_test[::48], rotation=45)
plt.show()

# %%
plt.plot(test_x, test_y, 'ro')
xsp = np.linspace(0,8, 100).reshape(-1, 1).astype(np.float32)
plt.plot(xsp, net(torch.tensor(xsp)).detach().numpy(), 'b')
plt.xlabel('Wind speed')
plt.ylabel('Power output')
plt.title('Test data over the wind speed')
plt.show()


