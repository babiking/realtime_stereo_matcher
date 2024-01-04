import torch
import torch.nn as nn


class TorchConcatenateCost(nn.Module):
    def __init__(self, max_disparity, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_disparity = max_disparity

    def forward(self, left, right):
        """
        torch naive concatenate left and right image features to generate cost volume

        Args:
            [1] left, (N, C, H, W), left image features
            [2] right, (N, C, H, W), right image features
            [3] max_disparity, Scalar/Int, max disparity value

        Return:
            [1] volume, (N, 2 * C, H, W, max_disparity), cost volume
                - volume[:, :C, y, x, d], left feature vector at pixel (x, y) aligned with disparity level d
                - volume[:, :C, y, x, d], right feature vector at pixel (x, y) aligned with disparity level d
        """
        n, c, h, w = left.shape

        volume = torch.zeros(
            size=(n, 2 * c, h, w, self.max_disparity),
            dtype=left.dtype,
            device=left.device,
        )

        for i in range(self.max_disparity):
            if i == 0:
                volume[:, :c, :, :, i] = left
                volume[:, c:, :, :, i] = right
            else:
                volume[:, :c, :, i:, i] = left[:, :, :, i:]
                volume[:, c:, :, i:, i] = right[:, :, :, :-i]
        volume = volume.contiguous()
        return volume
