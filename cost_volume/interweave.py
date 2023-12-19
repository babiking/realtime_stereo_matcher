import torch
import torch.nn as nn


class TorchInterweaveCost(nn.Module):
    # reference: https://github.com/cogsys-tuebingen/mobilestereonet
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, left, right):
        n, c, h, w = left.shape

        volume = torch.zeros(
            size=[n, 2 * c, h, w],
            dtype=left.dtype,
            device=left.device,
        )
        volume[:, ::2, :, :] = left
        volume[:, 1::2, :, :] = right

        volume = volume.contiguous()
        return volume

    def __str__(self):
        return self.__class__.__name__
