import torch
import numpy as np
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def get_random_sample_data(n=1, c=64, h=480, w=640):
    return torch.randn(size=(n, c, h, w), dtype=torch.float32)


def get_model_capacity(module, inputs=None, verbose=True):
    if inputs is None:
        inputs = (get_random_sample_data(),)

    n_macs, n_params = profile(module, inputs=inputs)

    n_flops = FlopCountAnalysis(module, inputs=inputs).total()

    # n_params = parameter_count_table(module)

    if verbose:
        # print(f"module={module}.")
        print("{:<30}  {:<8}G".format("#GMACS: ", np.round(n_macs / 1e9), 5))
        print("{:<30}  {:<8}G".format("#GFLOPS: ", np.round(n_flops / 1e9, 5)))
        print("{:<30}  {:<8}M".format("#PARAMS: ", np.round(n_params / 1e6, 5)))
    return n_macs, n_flops, n_params
