import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, window_size=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(window_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x


class RNN(nn.Module):
    def __init__(self, window_size=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.RNN(window_size, 64, 2, batch_first=True)
        self.lin = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = torch.sigmoid(self.lin(x))
        return x


def train_model(
    epochs: int,
    net: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    validationloader: torch.utils.data.DataLoader,
):
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
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val. Loss: {val_loss.item()}"
        )
