import torch
import torch.nn as nn


class TorchInnerProductCost(nn.Module):
    def __init__(self, max_disparity, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_disparity = max_disparity

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
            [1] volume, (N, D, H, W), inner-product based cost volume
                - D, maximum disparity value
        """
        n, c, h, w = left.shape

        volume = torch.zeros(
            size=(n, self.max_disparity, h, w), dtype=left.dtype, device=left.device
        )

        # (N, H, W, W)
        # product = torch.einsum("aijk,aijh->ajkh", left, right).view([n * h, 1, w, w])

        for i in range(self.max_disparity):
            if i == 0:
                volume[:, 0, :, :] = torch.sum(left * right, dim=1)
            else:
                volume[:, i, :, i:] = torch.sum(left[:, :, :, i:] * right[:, :, :, :-i], dim=1)
        volume = volume.contiguous()
        return volume

    def __str__(self):
        return f"{self.__class__.__name__} | aijk,aijh->ajkh"
