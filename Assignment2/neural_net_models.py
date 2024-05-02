import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class ANN_Model(nn.Module):
    def __init__(self, window_size=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(window_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))
        return x


class RNN_Model(nn.Module):
    hidden_dim = 16

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.RNN(1, self.hidden_dim, 2, batch_first=True)
        self.lin = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(2))
        x = torch.sigmoid(self.lin(x[:, -1, :]))
        return x


def train_model(
    epochs: int,
    net: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    validationloader: torch.utils.data.DataLoader,
    plot: bool = False,
):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainloader)
        val_loss = 0
        for x, y in validationloader:
            net.eval()
            output = net(x)
            val_loss += criterion(output, y).item()
            net.train()
        val_loss /= len(validationloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Val. Loss: {val_loss}")

    if plot:
        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.show()
