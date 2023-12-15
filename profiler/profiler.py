import torch
import numpy as np
from thop import profile


def get_random_sample_data(n=1, c=64, h=480, w=640):
    return torch.randn(size=(n, c, h, w), dtype=torch.float32)


def get_model_capacity(module, inputs=None, verbose=True):
    if inputs is None:
        inputs = (get_random_sample_data(),)

    n_macs, n_params = profile(module, inputs=inputs)

    if verbose:
        print(f"module={module}.")
        print("{:<30}  {:<8}G".format("#operations:", np.round(n_macs / 1e9), 5))
        print("{:<30}  {:<8}M".format("#parameters: ", np.round(n_params / 1e6, 5)))
    return n_macs, n_params
