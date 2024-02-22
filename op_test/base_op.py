import torch
import torch.nn as nn


class BaseOp(nn.Module):
    def get_random_inputs(self, n=1, c=64, h=120, w=160, *args, **kwargs):
        raise NotImplementedError
    
    def get_output_number(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError