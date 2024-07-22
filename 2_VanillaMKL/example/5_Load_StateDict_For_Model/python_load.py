import json
from typing import Tuple
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

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("x:", x, x.shape)
        print("h:", h, h.shape)
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        print("Batch Norm:", x, x.shape)
        x, h = self.lstm(x, h)
        print("GRU x:", x, x.shape)
        print("GRU h:", h, h.shape)
        x = self.linear(x.squeeze(1))
        print("Linear x:", x, x.shape)
        return x, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def convert_state_dict(state_dict: dict) -> dict:
    for key, value in state_dict.items():
        if isinstance(value, list):
            state_dict[key] = torch.tensor(value)
        else:
            state_dict[key] = torch.tensor([value])
    return state_dict


if __name__ == "__main__":
    # Instantiate the original model and load pretrained weights
    feature_dim = 5
    hidden_size = 10
    num_layers = 2

    with open("state_dict.json", "r") as fp:
        raw_state_dict = json.load(fp)

    state_dict = convert_state_dict(raw_state_dict)

    # Instantiate the new model
    single_step_model = SingleStepLSTMRegression(feature_dim, hidden_size, num_layers)

    # Load the state dict into the new model
    single_step_model.load_state_dict(state_dict)

    # Mush make this evaluation mode, otherwise you might get error when forwarding batch norm
    single_step_model.eval()

    x = torch.ones(feature_dim).unsqueeze(0).unsqueeze(0)
    h = torch.zeros(num_layers, 1, hidden_size)
    output = single_step_model(x, h)
    print(output[0])
