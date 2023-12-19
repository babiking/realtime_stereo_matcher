import torch
import torch.nn as nn


class TorchInnerProductCost(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, left, right):
        """
        torch naive inner product of left and right image features to generate pairwise cost volume

        Args:
            [1] left, (N, C, H, W), left image encoded features
                - N = batch size
                - C = number of input feature channels
                - H = image height
                - W = image width
            [2] right, (N, C, H, W), right image encoded features

        Return:
            [1] volume, (N * H * W, 1, 1, W), i.e. (N * H * W, C', 1, D'), inner-product based cost volume
                - C', number of output feature channels
                - D', maximum disparity value, i.e. image width
        """
        n, c, h, w = left.shape

        # volume: (N, H, W, W)
        volume = torch.einsum("aijk,aijh->ajkh", left, right)
        # volume: (N * H * W, 1, 1, W)
        volume = volume.view([n * h * w, 1, 1, w])

        volume = volume.contiguous()
        return volume

    def __str__(self):
        return f"{self.__class__.__name__} | aijk,aijh->ajkh"
