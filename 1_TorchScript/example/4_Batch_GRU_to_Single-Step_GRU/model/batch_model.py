import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class PaddedLSTMRegression(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.criterion = F.smooth_l1_loss
        self.batch_first = batch_first

        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.lstm = nn.GRU(
            feature_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
        )

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        pack = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        x = self.batch_norm(pack.data)
        x = PackedSequence(
            x, pack.batch_sizes, pack.sorted_indices, pack.unsorted_indices
        )
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)
        x, _ = self.lstm(x)

        x = self.linear(x)
        return x.squeeze(-1)
