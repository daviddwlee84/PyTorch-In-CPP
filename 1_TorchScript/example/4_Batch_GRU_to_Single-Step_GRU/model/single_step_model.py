import torch
import torch.nn as nn


class SingleStepLSTMRegression(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_size: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.lstm = nn.GRU(
            feature_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        x, h = self.lstm(x, h)
        x = self.linear(x.squeeze(1))
        return x, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
