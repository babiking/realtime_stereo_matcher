import torch
import numpy as np
from thop import profile


def get_random_sample_data(n=1, c=64, h=480, w=640):
    return torch.randn(size=(n, c, h, w), dtype=torch.float32)


def print_model_capacity(module, inputs=None):
    if inputs is None:
        inputs = (get_random_sample_data(),)

    n_macs, n_params = profile(module, inputs=inputs)

    print(f"module={module.__class__.__name__}.")
    print("{:<30}  {:<8}G".format("number_of_operations:", np.round(n_macs / 1e9), 5))
    print(
        "{:<30}  {:<8}M".format("number_of_parameters: ", np.round(n_params / 1e6, 5))
    )