import torch


if __name__ == "__main__":
    import json
    from model import dump_state_dict_to_json_str
    from python_load import convert_state_dict, SingleStepLSTMRegression
    from python_debug import SingleStepLSTMRegressionFromScratch, _debug_shape

    # Instantiate the original model and load pretrained weights
    feature_dim = 5
    hidden_size = 11
    num_layers = 3

    # Instantiate the new model
    single_step_model = SingleStepLSTMRegression(feature_dim, hidden_size, num_layers)

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

    # Might have a very tiny precision loss
    # state_dict_json = dump_state_dict_to_json_str(single_step_model.state_dict())
    # state_dict = convert_state_dict(json.loads(state_dict_json))
    state_dict = single_step_model.state_dict()

    from_scratch_model.custom_load_state_dict(state_dict)
    from_scratch_model.eval()

    from_scratch_output = from_scratch_model(x, h)
    print(from_scratch_output[0])

    # Sometimes can be false (while all close is true)
    print(output[0] == from_scratch_output[0])
    # https://lib.yanxishe.com/document/PyTorch1-7-1/api/torch.allclose
    print(torch.allclose(output[0], from_scratch_output[0]))
