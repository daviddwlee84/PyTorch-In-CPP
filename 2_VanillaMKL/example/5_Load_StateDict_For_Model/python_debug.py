from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.relevance_whh, self.relevance_wxh, self.relevance_b = (
            self.create_gate_parameters()
        )
        self.update_whh, self.update_wxh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_wxh, self.candidate_b = (
            self.create_gate_parameters()
        )

    def create_gate_parameters(self) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(input_weights)
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights, input_weights, bias

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_hiddens = []
        for i in range(x.shape[1]):
            relevance_gate = F.sigmoid(
                (h @ self.relevance_whh)
                + (x[:, i] @ self.relevance_wxh)
                + self.relevance_b
            )
            update_gate = F.sigmoid(
                (h @ self.update_whh) + (x[:, i] @ self.update_wxh) + self.update_b
            )
            candidate_hidden = F.tanh(
                ((relevance_gate * h) @ self.candidate_whh)
                + (x[:, i] @ self.candidate_wxh)
                + self.candidate_b
            )
            h = (update_gate * candidate_hidden) + ((1 - update_gate) * h)
            output_hiddens.append(h.unsqueeze(1))
        return torch.concat(output_hiddens, dim=1)


"""
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
        self.linear = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_hidden = self.layers[0](x, h[0])
        new_hidden = [output_hidden[:, -1].unsqueeze(0)]
        for i in range(1, self.num_layers):
            output_hidden = self.layers[i](self.dropout(output_hidden), h[i])
            new_hidden.append(output_hidden[:, -1].unsqueeze(0))
        print("GRU x:", output_hidden[:, -1].unsqueeze(1))
        print("GRU h:", h)
        return self.linear(self.dropout(output_hidden[:, -1].unsqueeze(1))).squeeze(
            1
        ), torch.concat(new_hidden, dim=0)


class SingleStepLSTMRegressionFromScratch(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int):
        super(SingleStepLSTMRegressionFromScratch, self).__init__()
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.gru = MultiLayerGRU(feature_dim, hidden_size, num_layers)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("x:", x)
        print("h:", h)
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        print("Batch Norm:", x)
        x, h = self.gru(x, h)
        print("Linear x:", x)
        return x, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_dim)
"""


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

    def forward(self, x, h):
        output_hidden, h[0] = self.layers[0](x, h[0])
        for i in range(1, self.num_layers):
            output_hidden, h[i] = self.layers[i](self.dropout(output_hidden), h[i])
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
        x = self.batch_norm(x.squeeze(1)).unsqueeze(1)
        x, h = self.gru(x, h)
        x = self.linear(x[:, -1])
        return x, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_dim)

    # def custom_load_state_dict(self, state_dict: dict) -> None:
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         import ipdb; ipdb.set_trace()
    #         if name.startswith("lstm"):
    #             layer_idx = int(name.split("_l")[1][0])
    #             if "weight_ih" in name:
    #                 param_name = f"gru.layers.{layer_idx}.relevance_wxh"
    #             elif "weight_hh" in name:
    #                 param_name = f"gru.layers.{layer_idx}.relevance_whh"
    #             elif "bias_ih" in name:
    #                 param_name = f"gru.layers.{layer_idx}.relevance_b"
    #             elif "bias_hh" in name:
    #                 param_name = f"gru.layers.{layer_idx}.update_b"
    #             if param_name in own_state:
    #                 own_state[param_name].copy_(param)
    #         elif name.startswith("linear"):
    #             param_name = f'linear.{name.split(".")[1]}'
    #             if param_name in own_state:
    #                 own_state[param_name].copy_(param)
    #     self.load_state_dict(own_state)

    # def custom_load_state_dict(self, state_dict: dict) -> None:
    #     """
    #     odict_keys(['batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'gru.layers.0.relevance_whh', 'gru.layers.0.relevance_wxh', 'gru.layers.0.relevance_b', 'gru.layers.0.update_whh', 'gru.layers.0.update_wxh', 'gru.layers.0.update_b', 'gru.layers.0.candidate_whh', 'gru.layers.0.candidate_wxh', 'gru.layers.0.candidate_b', 'gru.layers.1.relevance_whh', 'gru.layers.1.relevance_wxh', 'gru.layers.1.relevance_b', 'gru.layers.1.update_whh', 'gru.layers.1.update_wxh', 'gru.layers.1.update_b', 'gru.layers.1.candidate_whh', 'gru.layers.1.candidate_wxh', 'gru.layers.1.candidate_b', 'linear.weight', 'linear.bias'])
    #     dict_keys(['batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'lstm.weight_ih_l1', 'lstm.weight_hh_l1', 'lstm.bias_ih_l1', 'lstm.bias_hh_l1', 'linear.weight', 'linear.bias'])
    #     """
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name.startswith("lstm"):
    #             layer_idx = int(name.split("_l")[1][0])
    #             if "weight_ih" in name:
    #                 if "l0" in name:
    #                     param_name = f"gru.layers.{layer_idx}.relevance_wxh"
    #                 elif "l1" in name:
    #                     param_name = f"gru.layers.{layer_idx}.update_wxh"
    #             elif "weight_hh" in name:
    #                 if "l0" in name:
    #                     param_name = f"gru.layers.{layer_idx}.relevance_whh"
    #                 elif "l1" in name:
    #                     param_name = f"gru.layers.{layer_idx}.update_whh"
    #             elif "bias_ih" in name:
    #                 if "l0" in name:
    #                     param_name = f"gru.layers.{layer_idx}.relevance_b"
    #                 elif "l1" in name:
    #                     param_name = f"gru.layers.{layer_idx}.update_b"
    #             elif "bias_hh" in name:
    #                 param_name = f"gru.layers.{layer_idx}.candidate_b"
    #             import ipdb; ipdb.set_trace()
    #             if param_name in own_state:
    #                 own_state[param_name].copy_(param)
    #         elif name.startswith("linear"):
    #             param_name = f'linear.{name.split(".")[1]}'
    #             if param_name in own_state:
    #                 own_state[param_name].copy_(param)
    #     self.load_state_dict(own_state)

    def custom_load_state_dict(self, state_dict: dict) -> None:
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.startswith("lstm"):
                layer_idx = int(name.split("_l")[1][0])
                param_slices = [
                    slice(0, self.gru.hidden_dim),
                    slice(self.gru.hidden_dim, 2 * self.gru.hidden_dim),
                    slice(2 * self.gru.hidden_dim, 3 * self.gru.hidden_dim),
                ]
                if "weight_ih" in name:
                    for i, gate_name in enumerate(
                        ["relevance_wxh", "update_wxh", "candidate_wxh"]
                    ):
                        param_name = f"gru.layers.{layer_idx}.{gate_name}"
                        if param_name in own_state:
                            own_state[param_name].copy_(param[param_slices[i], :])
                elif "weight_hh" in name:
                    for i, gate_name in enumerate(
                        ["relevance_whh", "update_whh", "candidate_whh"]
                    ):
                        param_name = f"gru.layers.{layer_idx}.{gate_name}"
                        if param_name in own_state:
                            own_state[param_name].copy_(param[param_slices[i], :])
                elif "bias_ih" in name:
                    for i, gate_name in enumerate(
                        ["relevance_b", "update_b", "candidate_b"]
                    ):
                        param_name = f"gru.layers.{layer_idx}.{gate_name}"
                        if param_name in own_state:
                            own_state[param_name].copy_(param[param_slices[i]])
                elif "bias_hh" in name:
                    # PyTorch's GRU concatenates all biases, we only need the candidate_b for our custom model
                    param_name = f"gru.layers.{layer_idx}.candidate_b"
                    if param_name in own_state:
                        own_state[param_name].copy_(param)
            elif name.startswith("linear"):
                param_name = f'linear.{name.split(".")[1]}'
                if param_name in own_state:
                    own_state[param_name].copy_(param)
        self.load_state_dict(own_state)


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
    from_scratch_model.custom_load_state_dict(state_dict)
    from_scratch_model.eval()

    from_scratch_output = from_scratch_model(x, h)
    print(from_scratch_output[0])

    # BUG: Output is not correct
    print(output[0] == from_scratch_output[0])
