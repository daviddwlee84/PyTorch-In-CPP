from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell
    """

    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.weight_ih = nn.Parameter(torch.zeros(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.zeros(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_dim))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_gates = F.linear(x, self.weight_ih, self.bias_ih)
        print("x_gates:", x_gates, x_gates.shape)
        h_gates = F.linear(h, self.weight_hh, self.bias_hh)
        print("h_gates:", h_gates, h_gates.shape)

        # https://pytorch.org/docs/stable/generated/torch.chunk.html
        x_r, x_z, x_n = x_gates.chunk(3, 2)
        h_r, h_z, h_n = h_gates.chunk(3, 1)

        reset_gate = torch.sigmoid(x_r + h_r)
        update_gate = torch.sigmoid(x_z + h_z)
        new_hidden = torch.tanh(x_n + reset_gate * h_n)
        h = (1 - update_gate) * new_hidden + update_gate * h

        print("Reset Gate", reset_gate, reset_gate.shape)
        print("Update Gate", update_gate, update_gate.shape)
        print("New Hidden", new_hidden, new_hidden.shape)

        print("Output", h, h.shape)

        return h


class MultiLayerGRU(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0
    ):
        super(MultiLayerGRU, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = (
            input_dim,
            hidden_dim,
            num_layers,
        )
        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GRUCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_hidden = x
        for i in range(self.num_layers):
            h[i] = self.layers[i](output_hidden, h[i])
            output_hidden = h[i].unsqueeze(0)
            if i < self.num_layers - 1:
                output_hidden = self.dropout(output_hidden)
        return output_hidden, h


class SingleStepLSTMRegressionFromScratch(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int):
        super(SingleStepLSTMRegressionFromScratch, self).__init__()
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.gru = MultiLayerGRU(feature_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("x:", x, x.shape)
        print("h:", h, h.shape)
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        print("Batch Norm:", x, x.shape)
        x, h = self.gru(x, h)
        print("GRU x:", x, x.shape)
        print("GRU h:", h, h.shape)
        x = self.linear(x[:, -1])
        print("Linear x:", x, x.shape)
        return x, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_dim)

    def custom_load_state_dict(self, state_dict: dict) -> None:
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.startswith("lstm"):
                layer_idx = int(name.split("_l")[1][0])
                gate_slices = {
                    "weight_ih": (slice(None), slice(None)),
                    "weight_hh": (slice(None), slice(None)),
                    "bias_ih": slice(None),
                    "bias_hh": slice(None),
                }
                if "weight_ih" in name or "weight_hh" in name:
                    gate_slices["weight_ih"] = (
                        slice(0, 3 * self.gru.hidden_dim),
                        (
                            slice(0, self.gru.input_dim)
                            if layer_idx == 0
                            else slice(0, self.gru.hidden_dim)
                        ),
                    )
                    gate_slices["weight_hh"] = (
                        slice(0, 3 * self.gru.hidden_dim),
                        slice(0, self.gru.hidden_dim),
                    )
                elif "bias_ih" in name or "bias_hh" in name:
                    gate_slices["bias_ih"] = slice(0, 3 * self.gru.hidden_dim)
                    gate_slices["bias_hh"] = slice(0, 3 * self.gru.hidden_dim)

                for gate_name, slices in gate_slices.items():
                    if gate_name in name:
                        param_name = f"gru.layers.{layer_idx}.{gate_name}"
                        if param_name in own_state:
                            own_state[param_name].copy_(param[slices])
            elif name.startswith("linear"):
                param_name = f'linear.{name.split(".")[1]}'
                if param_name in own_state:
                    own_state[param_name].copy_(param)
        self.load_state_dict(own_state)


def _debug_shape(model: nn.Module) -> None:
    for name, param in model.state_dict().items():
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    import json
    from python_load import convert_state_dict, SingleStepLSTMRegression

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

    from_scratch_model = SingleStepLSTMRegressionFromScratch(
        feature_dim, hidden_size, num_layers
    )

    print("=== Single Step Model (ideal) ===")
    _debug_shape(single_step_model)
    print("=== From Scratch Model ===")
    _debug_shape(from_scratch_model)

    from_scratch_model.custom_load_state_dict(state_dict)
    from_scratch_model.eval()

    from_scratch_output = from_scratch_model(x, h)
    print(from_scratch_output[0])

    print(output[0] == from_scratch_output[0])
